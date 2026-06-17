// Copyright (c) 2023 Horizon Robotics.All Rights Reserved.
//
// The material in this file is confidential and contains trade secrets
// of Horizon Robotics Inc. This is proprietary information owned by
// Horizon Robotics Inc. No part of this work may be disclosed,
// reproduced, copied, transmitted, or used in any way for any purpose,
// without the express written permission of Horizon Robotics Inc.

#include "method/qat_bevfusion_multitask_post_process_method.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>

#include "base/common_def.h"
#include "hobot/dnn/hb_dnn.h"
#include "method/method_factory.h"
#include "plugin/workflow_plugin.h"
#include "rapidjson/document.h"
#include "utils/stop_watch.h"
#include "utils/tensor_utils.h"
#include "utils/utils.h"

DEFINE_AND_REGISTER_METHOD(QATBevFusionMultitaskPostProcessMethod);

namespace {

float Sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

/**
 * 反量化 scale：按 scaleLen 取 scaleData[idx]；idx >= scaleLen 或仅 1 个 scale 时
 * 广播 scaleData[0]（HBM 常见）。检测 cls/bbox/ref 与 OCC 共用此规则。
 */
float QuantScaleAt(const hbDNNTensor *tensor, int32_t idx) {
  const float *sd = tensor->properties.scale.scaleData;
  if (sd == nullptr) {
    return 1.0f;
  }
  const int32_t slen = tensor->properties.scale.scaleLen;
  if (slen <= 0) {
    return 1.0f;
  }
  if (idx >= 0 && idx < slen) {
    return sd[idx];
  }
  return sd[0];
}

/** Row (i) / col (j) slice uses BPU stride in bytes (same convention as BevFormer). */
float ReadQuant2d(const hbDNNTensor &t, int i, int j) {
  const auto *base =
      reinterpret_cast<const char *>(t.sysMem.virAddr);
  const int64_t off = static_cast<int64_t>(i) * t.properties.stride[1] +
                      static_cast<int64_t>(j) * t.properties.stride[2];
  const float sc = QuantScaleAt(&t, j);
  switch (t.properties.tensorType) {
    case HB_DNN_TENSOR_TYPE_S32: {
      int32_t v = *reinterpret_cast<const int32_t *>(base + off);
      return quanti_scale(v, sc);
    }
    case HB_DNN_TENSOR_TYPE_S16: {
      int16_t v = *reinterpret_cast<const int16_t *>(base + off);
      return quanti_scale(static_cast<int32_t>(v), sc);
    }
    case HB_DNN_TENSOR_TYPE_S8: {
      int8_t v = *reinterpret_cast<const int8_t *>(base + off);
      return quanti_scale(static_cast<int32_t>(v), sc);
    }
    default:
      VLOG(EXAMPLE_SYSTEM)
          << "BevFusion det tensor type " << t.properties.tensorType
          << " not handled; expect S32/S16/S8.";
      return 0.0f;
  }
}

std::vector<float> Flatten(const std::vector<std::vector<float>> &vec) {
  std::vector<float> flat;
  for (const auto &sub_vec : vec) {
    flat.insert(flat.end(), sub_vec.begin(), sub_vec.end());
  }
  return flat;
}

std::pair<std::vector<float>, std::vector<int>> TopK(
    const std::vector<float> &vec, int k) {
  std::vector<int> indices(vec.size());
  std::iota(indices.begin(), indices.end(), 0);
  int kk = std::min(k, static_cast<int>(vec.size()));
  std::partial_sort(indices.begin(), indices.begin() + kk, indices.end(),
                    [&vec](int a, int b) { return vec[a] > vec[b]; });

  std::vector<float> topk_values(kk);
  std::vector<int> topk_indices(kk);
  for (int i = 0; i < kk; ++i) {
    topk_values[i] = vec[indices[i]];
    topk_indices[i] = indices[i];
  }
  return {topk_values, topk_indices};
}

std::vector<std::vector<float>> DenormalizeBbox(
    const std::vector<std::vector<float>> &normalized_bboxes) {
  size_t num_bboxes = normalized_bboxes.size();
  std::vector<std::vector<float>> denormalized_bboxes(num_bboxes,
                                                      std::vector<float>(10));

  for (size_t i = 0; i < num_bboxes; ++i) {
    const auto &bbox = normalized_bboxes[i];
    float rot_sine = bbox[6];
    float rot_cosine = bbox[7];
    float rot = std::atan2(rot_sine, rot_cosine);

    float cx = bbox[0];
    float cy = bbox[1];
    float cz = bbox[4];

    float w = std::exp(bbox[2]);
    float bl = std::exp(bbox[3]);
    float h = std::exp(bbox[5]);

    float vx = 0.0f;
    float vy = 0.0f;
    if (bbox.size() > 8) {
      vx = bbox[8];
      vy = bbox[9];
    }

    denormalized_bboxes[i] = {cx, cy, cz, w, bl, h, rot, vx, vy};
  }

  return denormalized_bboxes;
}

/**
 * BEVOCCHead2D / multitask: layout [N,H,W,Dz,C] (e.g. [1,200,200,16,18]).
 * INT8 or UINT8 logits; per-voxel argmax in C, then BEV collapse (max over Z).
 */
void OccArgmaxNdhwcPacked(hbDNNTensor *tensor, Parsing3d<uint32_t> *seg3d,
                          Parsing<uint8_t> *lidar_seg, int32_t resize_h,
                          int32_t resize_w, float scale_h, float scale_w) {
  auto &vs = tensor->properties.validShape;
  int32_t seg_h = vs.dimensionSize[1];
  int32_t seg_w = vs.dimensionSize[2];
  int32_t seg_z = vs.dimensionSize[3];
  int32_t seg_c = vs.dimensionSize[4];

  int32_t seg_w_aligned =
      tensor->properties.stride[1] / tensor->properties.stride[2];
  int32_t seg_z_aligned =
      tensor->properties.stride[2] / tensor->properties.stride[3];
  int32_t seg_c_aligned =
      tensor->properties.stride[3] / tensor->properties.stride[4];

  const char *raw_base = reinterpret_cast<const char *>(tensor->sysMem.virAddr);

  auto occ_val_at = [&](int32_t pos) -> int32_t {
    if (tensor->properties.tensorType == HB_DNN_TENSOR_TYPE_U8) {
      return static_cast<int32_t>(
          *reinterpret_cast<const uint8_t *>(raw_base + pos));
    }
    return static_cast<int32_t>(
        *reinterpret_cast<const int8_t *>(raw_base + pos));
  };

  int32_t seg_k_hwz = seg_h * seg_w * seg_z;
  seg3d->num_classes = static_cast<uint32_t>(seg_c);
  seg3d->h = seg_h;
  seg3d->w = seg_w;
  seg3d->z = seg_z;
  seg3d->seg.resize(static_cast<size_t>(seg_k_hwz));

  for (int32_t h = 0; h < seg_h; ++h) {
    int32_t h_idx =
        h * seg_w_aligned * seg_z_aligned * seg_c_aligned;
    int32_t h_k = h * seg_w * seg_z;
    for (int32_t w = 0; w < seg_w; ++w) {
      int32_t hw_idx = h_idx + w * seg_z_aligned * seg_c_aligned;
      int32_t hw_k = h_k + w * seg_z;
      for (int32_t z = 0; z < seg_z; ++z) {
        int32_t idx = hw_idx + z * seg_c_aligned;
        float max_val = std::numeric_limits<float>::lowest();
        int32_t top_index = -1;
        for (int c = 0; c < seg_c; ++c) {
          float sc = QuantScaleAt(tensor, c);
          float v = quanti_scale(occ_val_at(idx + c), sc);
          if (v > max_val) {
            max_val = v;
            top_index = c;
          }
        }
        seg3d->seg[static_cast<size_t>(hw_k + z)] =
            static_cast<uint32_t>(std::max(0, top_index));
      }
    }
  }

  // BEV collapse: per (h,w,c) keep max logit over z; then upsample + argmax.
  std::vector<float> data(static_cast<size_t>(seg_c * seg_h * seg_w));
  for (int32_t c = 0; c < seg_c; ++c) {
    float sc = QuantScaleAt(tensor, c);
    for (int32_t hh = 0; hh < seg_h; ++hh) {
      for (int32_t ww = 0; ww < seg_w; ++ww) {
        float best = std::numeric_limits<float>::lowest();
        for (int32_t zz = 0; zz < seg_z; ++zz) {
          int32_t pos = hh * seg_w_aligned * seg_z_aligned * seg_c_aligned +
                        ww * seg_z_aligned * seg_c_aligned +
                        zz * seg_c_aligned + c;
          float v = quanti_scale(occ_val_at(pos), sc);
          if (v > best) {
            best = v;
          }
        }
        data[static_cast<size_t>(c * seg_h * seg_w + hh * seg_w + ww)] = best;
      }
    }
  }

  auto coord = GetOriginalCoordinateFromResizedCoordinate(false);
  float sh = scale_h > 0.f ? scale_h
                           : static_cast<float>(resize_h) /
                                 static_cast<float>(std::max(seg_h, 1));
  float sw = scale_w > 0.f ? scale_w
                           : static_cast<float>(resize_w) /
                                 static_cast<float>(std::max(seg_w, 1));

  int32_t kHW = resize_h * resize_w;
  std::vector<float> out_data(static_cast<size_t>(kHW) * seg_c);
  bilinear_upsample(out_data.data(), data.data(), seg_c, resize_h, resize_w,
                    seg_h, seg_w, sh, sw, coord);

  lidar_seg->num_classes = seg_c;
  lidar_seg->height = resize_h;
  lidar_seg->width = resize_w;
  lidar_seg->seg.resize(static_cast<size_t>(kHW));

  for (int32_t k = 0; k < kHW; k++) {
    float *data_tmp = out_data.data() + k;
    float max_val = std::numeric_limits<float>::lowest();
    int32_t top_index = -1;
    for (int c = 0; c < seg_c; c++) {
      float v = data_tmp[static_cast<size_t>(c) * static_cast<size_t>(kHW)];
      if (v > max_val) {
        max_val = v;
        top_index = c;
      }
    }
    lidar_seg->seg[static_cast<size_t>(k)] =
        static_cast<uint8_t>(std::max(0, top_index));
  }
}

void OccArgmaxNhwc(hbDNNTensor *tensor, Parsing<uint8_t> *lidar_seg,
                   int32_t resize_h, int32_t resize_w, float scale_h,
                   float scale_w, bool use_int32) {
  auto &vs = tensor->properties.validShape;
  int32_t seg_h = vs.dimensionSize[1];
  int32_t seg_w = vs.dimensionSize[2];
  int32_t seg_channel = vs.dimensionSize[3];

  auto align_shape = properies2alignshape(tensor->properties);
  int32_t align_c = align_shape[3];

  std::vector<float> data;
  data.reserve(static_cast<size_t>(seg_h * seg_w * seg_channel));

  if (use_int32 ||
      tensor->properties.tensorType == HB_DNN_TENSOR_TYPE_S32) {
    int32_t *seg_data = reinterpret_cast<int32_t *>(tensor->sysMem.virAddr);
    for (int32_t c = 0; c < seg_channel; c++) {
      for (int32_t hh = 0; hh < seg_h; hh++) {
        for (int32_t ww = 0; ww < seg_w; ww++) {
          int32_t pos =
              hh * seg_w * align_c + ww * align_c + c;
          float sc = QuantScaleAt(tensor, c);
          data.push_back(quanti_scale(seg_data[pos], sc));
        }
      }
    }
  } else {
    int8_t *seg_data = reinterpret_cast<int8_t *>(tensor->sysMem.virAddr);
    for (int32_t c = 0; c < seg_channel; c++) {
      for (int32_t hh = 0; hh < seg_h; hh++) {
        for (int32_t ww = 0; ww < seg_w; ww++) {
          int32_t pos =
              hh * seg_w * align_c + ww * align_c + c;
          float sc = QuantScaleAt(tensor, c);
          data.push_back(
              quanti_scale(static_cast<int32_t>(seg_data[pos]), sc));
        }
      }
    }
  }

  auto coord = GetOriginalCoordinateFromResizedCoordinate(false);

  float sh = scale_h > 0.f ? scale_h
                           : static_cast<float>(resize_h) /
                                 static_cast<float>(std::max(seg_h, 1));
  float sw = scale_w > 0.f ? scale_w
                           : static_cast<float>(resize_w) /
                                 static_cast<float>(std::max(seg_w, 1));

  int32_t kHW = resize_h * resize_w;
  std::vector<float> out_data(static_cast<size_t>(kHW) * seg_channel);
  bilinear_upsample(out_data.data(), data.data(), seg_channel, resize_h,
                    resize_w, seg_h, seg_w, sh, sw, coord);

  lidar_seg->num_classes = seg_channel;
  lidar_seg->height = resize_h;
  lidar_seg->width = resize_w;
  lidar_seg->seg.resize(static_cast<size_t>(kHW));

  for (int32_t k = 0; k < kHW; k++) {
    float *data_tmp = out_data.data() + k;
    float max_val = std::numeric_limits<float>::lowest();
    int32_t top_index = -1;
    for (int c = 0; c < seg_channel; c++) {
      float v = data_tmp[c * kHW];
      if (v > max_val) {
        max_val = v;
        top_index = c;
      }
    }
    lidar_seg->seg[static_cast<size_t>(k)] =
        static_cast<uint8_t>(std::max(0, top_index));
  }
}

}  // namespace

int QATBevFusionMultitaskPostProcessMethod::InitFromJsonString(
    const std::string &config) {
  VLOG(EXAMPLE_DEBUG) << "QATBevFusionMultitaskPostProcessMethod Json string:"
                      << config.data();

  rapidjson::Document document;
  document.Parse(config.data());
  if (document.HasParseError()) {
    VLOG(EXAMPLE_SYSTEM) << "Parsing config file failed";
    return -1;
  }

  if (document.HasMember("topk")) {
    topk_ = document["topk"].GetInt();
  }
  if (document.HasMember("score_threshold")) {
    score_threshold_ = document["score_threshold"].GetFloat();
  }
  if (document.HasMember("det_output_base")) {
    det_output_base_ = document["det_output_base"].GetInt();
  }
  if (document.HasMember("occ_tensor_idx")) {
    occ_tensor_idx_ = document["occ_tensor_idx"].GetInt();
  }
  if (document.HasMember("occ_resize_height")) {
    occ_resize_height_ = document["occ_resize_height"].GetInt();
  }
  if (document.HasMember("occ_resize_width")) {
    occ_resize_width_ = document["occ_resize_width"].GetInt();
  }
  if (document.HasMember("occ_scale_height")) {
    occ_scale_height_ = document["occ_scale_height"].GetFloat();
  }
  if (document.HasMember("occ_scale_width")) {
    occ_scale_width_ = document["occ_scale_width"].GetFloat();
  }
  if (document.HasMember("occ_use_int32")) {
    occ_use_int32_ = document["occ_use_int32"].GetBool();
  }

  if (document.HasMember("bev_range")) {
    auto bev_range = document["bev_range"].GetArray();
    bev_range_.resize(bev_range.Size());
    for (rapidjson::SizeType i = 0; i < bev_range.Size(); i++) {
      bev_range_[i] = bev_range[i].GetFloat();
    }
  } else {
    bev_range_ = {-51.2f, -51.2f, -5.0f, 51.2f, 51.2f, 3.0f};
  }

  if (document.HasMember("post_center_range")) {
    auto post_center_range = document["post_center_range"].GetArray();
    post_center_range_.resize(post_center_range.Size());
    for (rapidjson::SizeType i = 0; i < post_center_range.Size(); i++) {
      post_center_range_[i] = post_center_range[i].GetFloat();
    }
  } else {
    post_center_range_ = {-61.2f, -61.2f, -10.0f, 61.2f, 61.2f, 10.0f};
  }

  if (document.HasMember("ori_shape")) {
    auto ori_value = document["ori_shape"].GetArray();
    ori_shape_.resize(ori_value.Size());
    for (rapidjson::SizeType i = 0; i < ori_value.Size(); i++) {
      ori_shape_[i] = ori_value[i].GetInt();
    }
  } else {
    ori_shape_ = {900, 1600};
  }

  if (document.HasMember("resize_shape")) {
    auto resize_value = document["resize_shape"].GetArray();
    resize_shape_.resize(resize_value.Size());
    for (rapidjson::SizeType i = 0; i < resize_value.Size(); i++) {
      resize_shape_[i] = resize_value[i].GetInt();
    }
  } else {
    resize_shape_ = {512, 960};
  }

  return 0;
}

PerceptionPtr QATBevFusionMultitaskPostProcessMethod::DoProcess(
    ImageTensor *image_tensor, TensorVectorPtr &output_tensor) {
  auto perception = std::shared_ptr<Perception>(new Perception);
  PostProcess(output_tensor->tensors, image_tensor, perception.get());
  return perception;
}

int QATBevFusionMultitaskPostProcessMethod::PostProcess(
    std::vector<hbDNNTensor> &tensors, ImageTensor *image_tensor,
    Perception *perception) {
  perception->type = Perception::LIDARMULTITASK;

  const int base = det_output_base_;
  const int occ_idx = occ_tensor_idx_;
  if (base + 4 > static_cast<int>(tensors.size()) ||
      occ_idx >= static_cast<int>(tensors.size()) || occ_idx < 0) {
    VLOG(EXAMPLE_SYSTEM) << "Invalid tensor layout: outputs="
                         << tensors.size() << " det_base=" << base
                         << " occ_idx=" << occ_idx;
    return -1;
  }

  const uint64_t post_t0 = Stopwatch::CurrentTs();
  for (size_t i = 0; i < tensors.size(); i++) {
    hbUCPMemFlush(&(tensors[i].sysMem), HB_SYS_MEM_CACHE_INVALIDATE);
  }

  image_tensor->ori_image_height = ori_shape_[0];
  image_tensor->ori_image_width = ori_shape_[1];
  image_tensor->resize_height = resize_shape_[0];
  image_tensor->resize_width = resize_shape_[1];
  image_tensor->is_pad_resize = true;

  hbDNNTensor &bev_embed = tensors[base + 0];
  hbDNNTensor &outputs_classes = tensors[base + 1];
  hbDNNTensor &reference_out = tensors[base + 2];
  hbDNNTensor &bbox_outputs = tensors[base + 3];
  (void)bev_embed;

  int32_t num_boxes = bbox_outputs.properties.validShape.dimensionSize[1];
  int32_t bbox_dim = bbox_outputs.properties.validShape.dimensionSize[2];
  int32_t ref_dim = reference_out.properties.validShape.dimensionSize[2];
  (void)ref_dim;

  int32_t num_cls = outputs_classes.properties.validShape.dimensionSize[2];

  int32_t class_dim_stride = outputs_classes.properties.stride[1] /
                             outputs_classes.properties.stride[2];
  int32_t ref_dim_stride = reference_out.properties.stride[1] /
                         reference_out.properties.stride[2];
  int32_t bbox_dim_stride = bbox_outputs.properties.stride[1] /
                            bbox_outputs.properties.stride[2];
  (void)class_dim_stride;
  (void)ref_dim_stride;
  (void)bbox_dim_stride;

  if (outputs_classes.properties.tensorType != HB_DNN_TENSOR_TYPE_S32 &&
      outputs_classes.properties.tensorType != HB_DNN_TENSOR_TYPE_S16 &&
      outputs_classes.properties.tensorType != HB_DNN_TENSOR_TYPE_S8) {
    VLOG(EXAMPLE_SYSTEM) << "Unexpected cls tensor type "
                         << outputs_classes.properties.tensorType;
  }
  if (bbox_outputs.properties.tensorType != HB_DNN_TENSOR_TYPE_S32 &&
      bbox_outputs.properties.tensorType != HB_DNN_TENSOR_TYPE_S16 &&
      bbox_outputs.properties.tensorType != HB_DNN_TENSOR_TYPE_S8) {
    VLOG(EXAMPLE_SYSTEM) << "Unexpected bbox tensor type "
                         << bbox_outputs.properties.tensorType;
  }

  bool is_temporal = WorkflowPlugin::GetInstance()->IsTemporalModel();
  if (is_temporal) {
    VLOG(EXAMPLE_DEBUG)
        << "Temporal flag is on; BevFusion multitask path ignores it.";
  }

  std::vector<std::vector<float>> all_cls_scores(
      num_boxes, std::vector<float>(static_cast<size_t>(num_cls)));
  std::vector<std::vector<float>> all_bbox_preds(
      num_boxes, std::vector<float>(bbox_dim));

  for (int i = 0; i < num_boxes; ++i) {
    for (int j = 0; j < num_cls; ++j) {
      float score = ReadQuant2d(outputs_classes, i, j);
      all_cls_scores[static_cast<size_t>(i)][static_cast<size_t>(j)] =
          Sigmoid(score);
    }

    std::vector<float> tmp(static_cast<size_t>(bbox_dim_stride));
    for (int j = 0; j < bbox_dim; ++j) {
      tmp[static_cast<size_t>(j)] = ReadQuant2d(bbox_outputs, i, j);
    }

    tmp[0] += ReadQuant2d(reference_out, i, 0);
    tmp[1] += ReadQuant2d(reference_out, i, 1);
    tmp[0] = 1.0f / (1.0f + std::exp(-tmp[0]));
    tmp[1] = 1.0f / (1.0f + std::exp(-tmp[1]));

    tmp[4] += ReadQuant2d(reference_out, i, 2);
    tmp[4] = 1.0f / (1.0f + std::exp(-tmp[4]));

    tmp[0] = tmp[0] * (bev_range_[3] - bev_range_[0]) + bev_range_[0];
    tmp[1] = tmp[1] * (bev_range_[4] - bev_range_[1]) + bev_range_[1];
    tmp[4] = tmp[4] * (bev_range_[5] - bev_range_[2]) + bev_range_[2];

    for (int j = 0; j < bbox_dim; ++j) {
      all_bbox_preds[static_cast<size_t>(i)][static_cast<size_t>(j)] =
          tmp[static_cast<size_t>(j)];
    }
  }

  std::vector<float> flat_scores = Flatten(all_cls_scores);
  auto top_res = TopK(flat_scores, topk_);
  auto indices_topk = top_res.second;
  int eff_k = static_cast<int>(indices_topk.size());

  std::vector<float> final_scores(static_cast<size_t>(eff_k));
  std::vector<int> final_labels(static_cast<size_t>(eff_k));
  std::vector<int> bbox_indices(static_cast<size_t>(eff_k));

  for (int i = 0; i < eff_k; ++i) {
    final_scores[static_cast<size_t>(i)] = flat_scores[static_cast<size_t>(
        indices_topk[static_cast<size_t>(i)])];
    final_labels[static_cast<size_t>(i)] =
        indices_topk[static_cast<size_t>(i)] % num_cls;
    bbox_indices[static_cast<size_t>(i)] =
        indices_topk[static_cast<size_t>(i)] / num_cls;
  }

  std::vector<std::vector<float>> selected_bboxes(
      static_cast<size_t>(eff_k), std::vector<float>(all_bbox_preds[0].size()));
  for (int i = 0; i < eff_k; ++i) {
    selected_bboxes[static_cast<size_t>(i)] =
        all_bbox_preds[static_cast<size_t>(
            bbox_indices[static_cast<size_t>(i)])];
  }

  auto final_box_preds = DenormalizeBbox(selected_bboxes);

  std::vector<bool> thresh_mask(static_cast<size_t>(eff_k), false);
  if (score_threshold_ > 0) {
    for (int i = 0; i < eff_k; ++i) {
      thresh_mask[static_cast<size_t>(i)] =
          final_scores[static_cast<size_t>(i)] > score_threshold_;
    }
    float tmp_score = score_threshold_;
    while (std::none_of(thresh_mask.begin(), thresh_mask.end(),
                        [](bool v) { return v; })) {
      tmp_score *= 0.9f;
      if (tmp_score < 0.01f) {
        std::fill(thresh_mask.begin(), thresh_mask.end(), true);
        break;
      }
      for (int i = 0; i < eff_k; ++i) {
        thresh_mask[static_cast<size_t>(i)] =
            final_scores[static_cast<size_t>(i)] >= tmp_score;
      }
    }
  } else {
    std::fill(thresh_mask.begin(), thresh_mask.end(), true);
  }

  std::vector<std::vector<float>> boxes3d;
  std::vector<float> scores;
  std::vector<int32_t> labels;

  for (int32_t i = 0; i < eff_k; ++i) {
    const auto &box = final_box_preds[static_cast<size_t>(i)];
    bool in_range{true};
    for (int32_t j = 0; j < 3; ++j) {
      if (box[static_cast<size_t>(j)] < post_center_range_[static_cast<size_t>(j)] ||
          box[static_cast<size_t>(j)] >
              post_center_range_[static_cast<size_t>(j + 3)]) {
        in_range = false;
        break;
      }
    }
    if (in_range && thresh_mask[static_cast<size_t>(i)]) {
      boxes3d.push_back(final_box_preds[static_cast<size_t>(i)]);
      scores.push_back(final_scores[static_cast<size_t>(i)]);
      labels.push_back(final_labels[static_cast<size_t>(i)]);
    }
  }

  perception->lidar3d.reserve(boxes3d.size());
  for (size_t i = 0; i < boxes3d.size(); ++i) {
    const auto &box = boxes3d[i];
    Lidar3D lb(box[0], box[1], box[2], box[3], box[4], box[5], box[6], box[7],
               box[8]);
    perception->lidar3d.emplace_back(LidarDetection3D{
        lb, scores[static_cast<size_t>(i)], labels[static_cast<size_t>(i)]});
  }

  hbDNNTensor &occ_tensor = tensors[occ_idx];
  uint32_t occ_rank = occ_tensor.properties.validShape.numDimensions;
  if (occ_rank >= 5U &&
      (occ_tensor.properties.tensorType == HB_DNN_TENSOR_TYPE_S8 ||
       occ_tensor.properties.tensorType == HB_DNN_TENSOR_TYPE_U8)) {
    VLOG(EXAMPLE_DEBUG) << "OCC output NDHWC packed (rank=" << occ_rank
                        << ", type=" << occ_tensor.properties.tensorType << ").";
    OccArgmaxNdhwcPacked(&occ_tensor, &perception->seg3d, &perception->lidarSeg,
                         occ_resize_height_, occ_resize_width_, occ_scale_height_,
                         occ_scale_width_);
  } else {
    perception->seg3d.seg.clear();
    perception->seg3d.h = perception->seg3d.w = perception->seg3d.z = 0;
    OccArgmaxNhwc(&occ_tensor, &perception->lidarSeg, occ_resize_height_,
                  occ_resize_width_, occ_scale_height_, occ_scale_width_,
                  occ_use_int32_);
  }

  const uint64_t post_us = Stopwatch::CurrentTs() - post_t0;
  VLOG(EXAMPLE_REPORT) << std::fixed << std::setprecision(3)
                      << "BevFusion multitask latency frame_id="
                      << image_tensor->frame_id << " pre_ms="
                      << image_tensor->pre_duration / 1000.0 << " infer_ms="
                      << image_tensor->infer_duration / 1000.0 << " post_ms="
                      << post_us / 1000.0;
  return 0;
}
