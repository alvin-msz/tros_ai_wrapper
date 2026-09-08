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
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <string>
#include <sys/stat.h>

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

static void EnsureDir(const std::string &dir) {
  if (dir.empty()) return;
  mkdir(dir.c_str(), 0755);
}

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

int32_t OccElemBytes(int32_t tensor_type) {
  switch (tensor_type) {
    case HB_DNN_TENSOR_TYPE_S32:
    case HB_DNN_TENSOR_TYPE_U32:
    case HB_DNN_TENSOR_TYPE_F32:
      return 4;
    case HB_DNN_TENSOR_TYPE_S16:
    case HB_DNN_TENSOR_TYPE_U16:
    case HB_DNN_TENSOR_TYPE_F16:
      return 2;
    default:
      return 1;
  }
}

int32_t OccReadClassId(const hbDNNTensor *tensor, int64_t byte_off) {
  const char *p =
      reinterpret_cast<const char *>(tensor->sysMem.virAddr) + byte_off;
  switch (tensor->properties.tensorType) {
    case HB_DNN_TENSOR_TYPE_S32:
      return *reinterpret_cast<const int32_t *>(p);
    case HB_DNN_TENSOR_TYPE_S16:
      return static_cast<int32_t>(*reinterpret_cast<const int16_t *>(p));
    case HB_DNN_TENSOR_TYPE_U8:
      return static_cast<int32_t>(*reinterpret_cast<const uint8_t *>(p));
    case HB_DNN_TENSOR_TYPE_S8:
    default:
      return static_cast<int32_t>(*reinterpret_cast<const int8_t *>(p));
  }
}

std::string OccTensorDesc(const hbDNNTensor &t) {
  std::ostringstream oss;
  const auto &vs = t.properties.validShape;
  oss << "rank=" << vs.numDimensions << " shape=[";
  for (int32_t i = 0; i < vs.numDimensions; ++i) {
    if (i > 0) {
      oss << ",";
    }
    oss << vs.dimensionSize[i];
  }
  oss << "] type=" << t.properties.tensorType
      << " aligned=" << t.properties.alignedByteSize << " stride=[";
  for (int32_t i = 0; i < vs.numDimensions; ++i) {
    if (i > 0) {
      oss << ",";
    }
    oss << t.properties.stride[i];
  }
  oss << "]";
  return oss.str();
}

int64_t OccShapeVolume(const hbDNNTensor &t) {
  const auto &vs = t.properties.validShape;
  int64_t vol = 1;
  for (int32_t i = 0; i < vs.numDimensions; ++i) {
    const int32_t di = vs.dimensionSize[i];
    if (di > 0) {
      vol *= static_cast<int64_t>(di);
    }
  }
  return vol;
}

bool OccLooksLikeLogits(const hbDNNTensor &t, int32_t num_classes) {
  const auto &vs = t.properties.validShape;
  if (vs.numDimensions < 5) {
    return false;
  }
  const int32_t last = vs.dimensionSize[vs.numDimensions - 1];
  const int32_t d1 = vs.dimensionSize[1];
  return last == num_classes && d1 > 1 && vs.dimensionSize[2] > 1;
}

bool OccLooksLikeGrid(const hbDNNTensor &t) {
  const int64_t vol = OccShapeVolume(t);
  // Occ3D 200x200x16 class ids, or 200x200x16x18 logits.
  return vol == 640000 || vol == 11520000;
}

int ResolveOccTensorIdx(const std::vector<hbDNNTensor> &tensors,
                        int preferred) {
  auto usable = [](const hbDNNTensor &t) {
    return OccLooksLikeGrid(t) && t.properties.alignedByteSize > 0 &&
           t.sysMem.virAddr != nullptr;
  };
  if (preferred >= 0 && preferred < static_cast<int>(tensors.size()) &&
      usable(tensors[preferred])) {
    return preferred;
  }
  int best = -1;
  int64_t best_vol = 0;
  for (int i = 0; i < static_cast<int>(tensors.size()); ++i) {
    if (!usable(tensors[i])) {
      continue;
    }
    const int64_t vol = OccShapeVolume(tensors[i]);
    if (vol > best_vol) {
      best_vol = vol;
      best = i;
    }
  }
  if (best >= 0) {
    return best;
  }
  if (preferred >= 0 && preferred < static_cast<int>(tensors.size())) {
    return preferred;
  }
  return tensors.empty() ? -1 : static_cast<int>(tensors.size()) - 1;
}

struct OccHwzLayout {
  int32_t h = 0;
  int32_t w = 0;
  int32_t z = 0;
  int64_t sh = 0;
  int64_t sw = 0;
  int64_t sz = 0;
  bool ok = false;
};

OccHwzLayout InferOccClassIdLayout(const hbDNNTensor &t) {
  const auto &vs = t.properties.validShape;
  const int32_t *d = vs.dimensionSize;
  const int64_t *st = t.properties.stride;
  const uint32_t rank = vs.numDimensions;
  OccHwzLayout L;

  auto set_from = [&](int hi, int wi, int zi) {
    L.h = d[hi];
    L.w = d[wi];
    L.z = d[zi];
    L.sh = st[hi];
    L.sw = st[wi];
    L.sz = st[zi];
    L.ok = (L.h > 0 && L.w > 0 && L.z > 0);
  };

  if (rank == 5) {
    if (d[1] <= 1 && d[2] > 1 && d[3] > 1) {
      set_from(2, 3, 4);  // [N,1,H,W,Z]
    } else if (d[4] <= 1) {
      set_from(1, 2, 3);  // [N,H,W,Z,1]
    } else {
      set_from(1, 2, 3);
    }
  } else if (rank == 4) {
    // NCHW [N,Z,H,W] when Z is the small axis.
    if (d[1] <= 32 && d[2] >= 64 && d[3] >= 64 && d[1] < d[2] &&
        d[1] < d[3]) {
      set_from(2, 3, 1);
    } else {
      set_from(1, 2, 3);  // [N,H,W,Z]
    }
  } else if (rank == 3) {
    if (d[0] <= 32 && d[1] >= 64 && d[2] >= 64) {
      set_from(1, 2, 0);  // [Z,H,W]
    } else {
      set_from(0, 1, 2);  // [H,W,Z]
    }
  } else if (rank == 2 && OccShapeVolume(t) == 640000) {
    L.h = 200;
    L.w = 200;
    L.z = 16;
    const int32_t es = OccElemBytes(t.properties.tensorType);
    L.sz = es;
    L.sw = static_cast<int64_t>(L.z) * es;
    L.sh = static_cast<int64_t>(L.w) * L.z * es;
    L.ok = true;
  }

  if (L.ok && (L.sh == 0 || L.sw == 0 || L.sz == 0)) {
    const int32_t es = OccElemBytes(t.properties.tensorType);
    L.sz = es;
    L.sw = static_cast<int64_t>(L.z) * es;
    L.sh = static_cast<int64_t>(L.w) * L.z * es;
  }
  return L;
}

/**
 * BPU argmax already applied: class ids, not logits — do not dequant.
 * Accepts [N,H,W,Z], [N,H,W,Z,1], [N,1,H,W,Z], NCHW [N,Z,H,W], rank-3 HWZ.
 */
void OccCopyNdhwClassIds(hbDNNTensor *tensor, Parsing3d<uint32_t> *seg3d,
                         Parsing<uint8_t> *lidar_seg, int32_t resize_h,
                         int32_t resize_w, int32_t num_classes) {
  const OccHwzLayout layout = InferOccClassIdLayout(*tensor);
  if (!layout.ok || tensor->sysMem.virAddr == nullptr ||
      tensor->properties.alignedByteSize <= 0) {
    VLOG(EXAMPLE_SYSTEM) << "OCC class-id copy skipped: "
                         << OccTensorDesc(*tensor)
                         << " layout_ok=" << layout.ok;
    return;
  }

  const int32_t seg_h = layout.h;
  const int32_t seg_w = layout.w;
  const int32_t seg_z = layout.z;
  const int32_t seg_k_hwz = seg_h * seg_w * seg_z;
  seg3d->num_classes = static_cast<uint32_t>(std::max(num_classes, 1));
  seg3d->h = seg_h;
  seg3d->w = seg_w;
  seg3d->z = seg_z;
  seg3d->seg.resize(static_cast<size_t>(seg_k_hwz));

  auto class_at = [&](int32_t h, int32_t w, int32_t z) -> int32_t {
    const int64_t off = static_cast<int64_t>(h) * layout.sh +
                        static_cast<int64_t>(w) * layout.sw +
                        static_cast<int64_t>(z) * layout.sz;
    return OccReadClassId(tensor, off);
  };

  for (int32_t h = 0; h < seg_h; ++h) {
    const int32_t h_k = h * seg_w * seg_z;
    for (int32_t w = 0; w < seg_w; ++w) {
      const int32_t hw_k = h_k + w * seg_z;
      for (int32_t z = 0; z < seg_z; ++z) {
        const int32_t label = std::max(0, class_at(h, w, z));
        seg3d->seg[static_cast<size_t>(hw_k + z)] =
            static_cast<uint32_t>(label);
      }
    }
  }

  const int32_t free_index = std::max(0, num_classes - 1);
  std::vector<uint8_t> bev(static_cast<size_t>(seg_h * seg_w),
                           static_cast<uint8_t>(free_index));
  for (int32_t h = 0; h < seg_h; ++h) {
    for (int32_t w = 0; w < seg_w; ++w) {
      uint8_t lab = static_cast<uint8_t>(free_index);
      for (int32_t z = seg_z - 1; z >= 0; --z) {
        const int32_t id = class_at(h, w, z);
        if (id != free_index) {
          lab = static_cast<uint8_t>(std::max(0, std::min(id, 255)));
          break;
        }
      }
      bev[static_cast<size_t>(h * seg_w + w)] = lab;
    }
  }

  const int32_t out_h = std::max(resize_h, 1);
  const int32_t out_w = std::max(resize_w, 1);
  lidar_seg->num_classes = num_classes;
  lidar_seg->height = out_h;
  lidar_seg->width = out_w;
  lidar_seg->seg.resize(static_cast<size_t>(out_h * out_w));
  for (int32_t rh = 0; rh < out_h; ++rh) {
    const int32_t sh = std::min(seg_h - 1, rh * seg_h / out_h);
    for (int32_t rw = 0; rw < out_w; ++rw) {
      const int32_t sw = std::min(seg_w - 1, rw * seg_w / out_w);
      lidar_seg->seg[static_cast<size_t>(rh * out_w + rw)] =
          bev[static_cast<size_t>(sh * seg_w + sw)];
    }
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
  if (document.HasMember("occ_num_classes")) {
    occ_num_classes_ = document["occ_num_classes"].GetInt();
  }

  if (document.HasMember("eval_output_dir")) {
    eval_output_dir_ = document["eval_output_dir"].GetString();
    EnsureDir(eval_output_dir_);
    VLOG(EXAMPLE_SYSTEM) << "Multitask eval output dir: " << eval_output_dir_;
  }

  if (document.HasMember("eval_occ_prefix")) {
    eval_occ_prefix_ = document["eval_occ_prefix"].GetString();
  }

  if (document.HasMember("eval_det_prefix")) {
    eval_det_prefix_ = document["eval_det_prefix"].GetString();
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
  if (base + 4 > static_cast<int>(tensors.size())) {
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

  const int resolved_occ_idx = ResolveOccTensorIdx(tensors, occ_idx);
  if (resolved_occ_idx < 0 ||
      resolved_occ_idx >= static_cast<int>(tensors.size())) {
    VLOG(EXAMPLE_SYSTEM) << "No OCC output tensor. outputs="
                         << tensors.size() << " occ_idx=" << occ_idx;
    perception->seg3d.seg.clear();
    perception->seg3d.h = perception->seg3d.w = perception->seg3d.z = 0;
  } else {
    if (resolved_occ_idx != occ_idx) {
      VLOG(EXAMPLE_SYSTEM) << "OCC tensor idx " << occ_idx
                           << " is not a 200x200x16 grid; using index "
                           << resolved_occ_idx;
    }
    for (size_t i = 0; i < tensors.size(); ++i) {
      VLOG(EXAMPLE_DEBUG) << "output[" << i << "] "
                          << OccTensorDesc(tensors[i]);
    }

    hbDNNTensor &occ_tensor = tensors[resolved_occ_idx];
    VLOG(EXAMPLE_SYSTEM) << "OCC tensor idx=" << resolved_occ_idx << " "
                         << OccTensorDesc(occ_tensor);

    if (occ_tensor.properties.alignedByteSize <= 0 ||
        occ_tensor.sysMem.virAddr == nullptr) {
      VLOG(EXAMPLE_SYSTEM)
          << "OCC output is empty (alignedByteSize=0). The HBM ArgMax node "
             "was not mapped to BPU. Rebuild with quantized INT32 ArgMax "
             "(no DeQuant / .to(int8) after argmax).";
      perception->seg3d.seg.clear();
      perception->seg3d.h = perception->seg3d.w = perception->seg3d.z = 0;
    } else if (OccLooksLikeLogits(occ_tensor, occ_num_classes_)) {
      VLOG(EXAMPLE_DEBUG) << "OCC logits NDHWC, CPU argmax.";
      OccArgmaxNdhwcPacked(&occ_tensor, &perception->seg3d,
                           &perception->lidarSeg, occ_resize_height_,
                           occ_resize_width_, occ_scale_height_,
                           occ_scale_width_);
    } else if (InferOccClassIdLayout(occ_tensor).ok) {
      VLOG(EXAMPLE_DEBUG) << "OCC class ids (BPU argmax).";
      OccCopyNdhwClassIds(&occ_tensor, &perception->seg3d,
                          &perception->lidarSeg, occ_resize_height_,
                          occ_resize_width_, occ_num_classes_);
    } else {
      VLOG(EXAMPLE_SYSTEM)
          << "OCC layout not 3D class-id / logits; fallback 2D.";
      perception->seg3d.seg.clear();
      perception->seg3d.h = perception->seg3d.w = perception->seg3d.z = 0;
      OccArgmaxNhwc(&occ_tensor, &perception->lidarSeg, occ_resize_height_,
                    occ_resize_width_, occ_scale_height_, occ_scale_width_,
                    occ_use_int32_);
    }
  }

  if (!eval_output_dir_.empty()) {
    SaveOccPredBin(image_tensor, perception);
    // SaveDetPredBin(image_tensor, perception);
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

int QATBevFusionMultitaskPostProcessMethod::SaveOccPredBin(
    const ImageTensor *image_tensor, const Perception *perception) {
  if (perception->seg3d.seg.empty() || perception->seg3d.h <= 0 ||
      perception->seg3d.w <= 0 || perception->seg3d.z <= 0) {
    VLOG(EXAMPLE_SYSTEM) << "Skip OCC pred bin: empty seg3d frame_id="
                         << image_tensor->frame_id
                         << " h=" << perception->seg3d.h
                         << " w=" << perception->seg3d.w
                         << " z=" << perception->seg3d.z
                         << " elems=" << perception->seg3d.seg.size();
    return -1;
  }

  const size_t elem_count = perception->seg3d.seg.size();
  std::vector<int16_t> pred(elem_count);
  for (size_t i = 0; i < elem_count; ++i) {
    const int32_t label = static_cast<int32_t>(perception->seg3d.seg[i]);
    pred[i] = static_cast<int16_t>(std::max(0, std::min(label, 32767)));
  }

  std::ostringstream oss;
  oss << eval_output_dir_ << "/" << eval_occ_prefix_ << std::setw(6)
      << std::setfill('0') << image_tensor->frame_id << ".bin";
  std::ofstream ofs(oss.str(),
                    std::ios::out | std::ios::binary | std::ios::trunc);
  if (!ofs) {
    VLOG(EXAMPLE_SYSTEM) << "Open OCC pred bin failed: " << oss.str();
  } else {
    ofs.write(reinterpret_cast<const char *>(pred.data()),
              static_cast<std::streamsize>(pred.size() * sizeof(int16_t)));
    VLOG(EXAMPLE_DEBUG) << "Saved OCC pred bin: " << oss.str()
                        << " shape=[" << perception->seg3d.h << ","
                        << perception->seg3d.w << ","
                        << perception->seg3d.z << "] elems=" << elem_count;
  }

  if (image_tensor->frame_id == 0) {
    std::string meta_path = eval_output_dir_ + "/meta_occ.json";
    std::ofstream mf(meta_path);
    if (mf) {
      mf << "{\n"
         << "  \"h\": " << perception->seg3d.h << ",\n"
         << "  \"w\": " << perception->seg3d.w << ",\n"
         << "  \"z\": " << perception->seg3d.z << ",\n"
         << "  \"num_classes\": " << perception->seg3d.num_classes << ",\n"
         << "  \"dtype\": \"int16\",\n"
         << "  \"order\": \"HWZ_flattened\"\n"
         << "}\n";
    }
  }

  // // Save BEV segmentation (lidarSeg) if valid.
  // if (!perception->lidarSeg.seg.empty() && perception->lidarSeg.height > 0 &&
  //     perception->lidarSeg.width > 0) {
  //   const size_t elem_count = perception->lidarSeg.seg.size();
  //   std::vector<int16_t> pred(elem_count);
  //   for (size_t i = 0; i < elem_count; ++i) {
  //     pred[i] = static_cast<int16_t>(perception->lidarSeg.seg[i]);
  //   }

  //   std::ostringstream oss;
  //   oss << eval_output_dir_ << "/" << eval_occ_prefix_ << "bev_"
  //       << std::setw(6) << std::setfill('0') << image_tensor->frame_id
  //       << ".bin";
  //   std::ofstream ofs(oss.str(),
  //                     std::ios::out | std::ios::binary | std::ios::trunc);
  //   if (!ofs) {
  //     VLOG(EXAMPLE_SYSTEM) << "Open BEV seg pred bin failed: " << oss.str();
  //   } else {
  //     ofs.write(reinterpret_cast<const char *>(pred.data()),
  //               static_cast<std::streamsize>(pred.size() * sizeof(int16_t)));
  //     VLOG(EXAMPLE_DEBUG) << "Saved BEV seg pred bin: " << oss.str()
  //                         << " shape=[" << perception->lidarSeg.height << ","
  //                         << perception->lidarSeg.width
  //                         << "] elems=" << elem_count;
  //   }

  //   if (image_tensor->frame_id == 0) {
  //     std::string meta_path = eval_output_dir_ + "/meta_bev.json";
  //     std::ofstream mf(meta_path);
  //     if (mf) {
  //       mf << "{\n"
  //          << "  \"height\": " << perception->lidarSeg.height << ",\n"
  //          << "  \"width\": " << perception->lidarSeg.width << ",\n"
  //          << "  \"num_classes\": " << perception->lidarSeg.num_classes << ",\n"
  //          << "  \"dtype\": \"int16\",\n"
  //          << "  \"order\": \"HW_flattened\"\n"
  //          << "}\n";
  //     }
  //   }
  // }

  return 0;
}

int QATBevFusionMultitaskPostProcessMethod::SaveDetPredBin(
    const ImageTensor *image_tensor, const Perception *perception) {
  const auto &detections = perception->lidar3d;
  if (detections.empty()) {
    VLOG(EXAMPLE_DEBUG) << "No detections to save for frame_id="
                        << image_tensor->frame_id;
    return 0;
  }

  // Binary format: int32_t N, then N x { float score, int32_t label,
  //   float cx, float cy, float cz, float w, float l, float h,
  //   float rot, float vx, float vy }
  const int32_t num_dets = static_cast<int32_t>(detections.size());
  constexpr int32_t kFloatsPerDet = 9;
  const size_t header_sz = sizeof(int32_t);
  const size_t body_sz =
      static_cast<size_t>(num_dets) *
      (sizeof(float) + sizeof(int32_t) + kFloatsPerDet * sizeof(float));
  std::vector<char> buf(header_sz + body_sz);

  *reinterpret_cast<int32_t *>(buf.data()) = num_dets;
  char *ptr = buf.data() + header_sz;
  for (const auto &det : detections) {
    *reinterpret_cast<float *>(ptr) = det.score;
    ptr += sizeof(float);
    *reinterpret_cast<int32_t *>(ptr) = det.label;
    ptr += sizeof(int32_t);
    const auto &b = det.bbox;
    float feats[kFloatsPerDet] = {b.xs,    b.ys,     b.height, b.dim_0,
                                   b.dim_1, b.dim_2,  b.rot,    b.vel_0,
                                   b.vel_1};
    std::memcpy(ptr, feats, sizeof(feats));
    ptr += sizeof(feats);
  }

  std::ostringstream oss;
  oss << eval_output_dir_ << "/" << eval_det_prefix_ << std::setw(6)
      << std::setfill('0') << image_tensor->frame_id << ".bin";
  std::ofstream ofs(oss.str(),
                    std::ios::out | std::ios::binary | std::ios::trunc);
  if (!ofs) {
    VLOG(EXAMPLE_SYSTEM) << "Open det pred bin failed: " << oss.str();
    return -1;
  }
  ofs.write(buf.data(), static_cast<std::streamsize>(buf.size()));
  if (!ofs) {
    VLOG(EXAMPLE_SYSTEM) << "Write det pred bin failed: " << oss.str();
    return -1;
  }
  VLOG(EXAMPLE_DEBUG) << "Saved det pred bin: " << oss.str()
                      << " num_dets=" << num_dets;
  return 0;
}
