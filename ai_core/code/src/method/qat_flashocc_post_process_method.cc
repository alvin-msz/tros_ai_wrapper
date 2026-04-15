// Copyright (c) 2023 Horizon Robotics.All Rights Reserved.
//
// The material in this file is confidential and contains trade secrets
// of Horizon Robotics Inc. This is proprietary information owned by
// Horizon Robotics Inc. No part of this work may be disclosed,
// reproduced, copied, transmitted, or used in any way for any purpose,
// without the express written permission of Horizon Robotics Inc.

#include "method/qat_flashocc_post_process_method.h"

#include <cmath>
#include <iostream>
#include <queue>
#include <unordered_map>

#include "method/method_data.h"
#include "method/method_factory.h"
#include "plugin/workflow_plugin.h"
#include "rapidjson/document.h"
#include "utils/tensor_utils.h"
#include "utils/utils.h"

DEFINE_AND_REGISTER_METHOD(QATFlashoccPostProcessMethod);

int QATFlashoccPostProcessMethod::InitFromJsonString(
    const std::string &config) {
  VLOG(EXAMPLE_DEBUG) << "QATFlashoccPostProcessMethod Json string:"
                      << config.data();

  rapidjson::Document document;
  document.Parse(config.data());

  if (document.HasParseError()) {
    VLOG(EXAMPLE_SYSTEM) << "Parsing config file failed";
    return -1;
  }

  if (document.HasMember("ori_shape")) {
    auto ori_value = document["ori_shape"].GetArray();
    ori_shape_.resize(ori_value.Size());
    for (int i = 0; i < ori_value.Size(); i++) {
      ori_shape_[i] = ori_value[i].GetInt();
    }
  }

  if (document.HasMember("resize_shape")) {
    auto resize_value = document["resize_shape"].GetArray();
    resize_shape_.resize(resize_value.Size());
    for (int i = 0; i < resize_value.Size(); i++) {
      resize_shape_[i] = resize_value[i].GetInt();
    }
  }

  return 0;
}

PerceptionPtr QATFlashoccPostProcessMethod::DoProcess(
    ImageTensor *image_tensor, TensorVectorPtr &output_tensor) {
  auto perception = std::shared_ptr<Perception>(new Perception);
  PostProcess(output_tensor->tensors, image_tensor, perception.get());
  return perception;
}

int QATFlashoccPostProcessMethod::PostProcess(std::vector<hbDNNTensor> &tensors,
                                              ImageTensor *image_tensor,
                                              Perception *perception) {
  perception->type = Perception::SEG3D;
  image_tensor->ori_image_height = ori_shape_[0];
  image_tensor->ori_image_width = ori_shape_[1];
  image_tensor->resize_height = resize_shape_[0];
  image_tensor->resize_width = resize_shape_[1];

  VLOG(EXAMPLE_DEBUG) << "ori_image_height: " << image_tensor->ori_image_height;
  VLOG(EXAMPLE_DEBUG) << "ori_image_width: " << image_tensor->ori_image_width;
  VLOG(EXAMPLE_DEBUG) << "resize_height: " << image_tensor->resize_height;
  VLOG(EXAMPLE_DEBUG) << "resize_width: " << image_tensor->resize_width;

  int8_t *seg_data = reinterpret_cast<int8_t *>(tensors[0].sysMem.virAddr);
  int32_t seg_h = tensors[0].properties.validShape.dimensionSize[1];
  int32_t seg_w = tensors[0].properties.validShape.dimensionSize[2];
  int32_t seg_z = tensors[0].properties.validShape.dimensionSize[3];
  int32_t seg_channel = tensors[0].properties.validShape.dimensionSize[4];
  float *seg_scale = tensors[0].properties.scale.scaleData;

  VLOG(EXAMPLE_DEBUG) << "seg_channel: " << seg_channel;
  VLOG(EXAMPLE_DEBUG) << "seg_h: " << seg_h;
  VLOG(EXAMPLE_DEBUG) << "seg_w: " << seg_w;
  VLOG(EXAMPLE_DEBUG) << "seg_z: " << seg_z;

  int32_t seg_w_aligned =
      tensors[0].properties.stride[1] / tensors[0].properties.stride[2];
  int32_t seg_z_aligned =
      tensors[0].properties.stride[2] / tensors[0].properties.stride[3];
  int32_t seg_channel_aligned =
      tensors[0].properties.stride[3] / tensors[0].properties.stride[4];

  VLOG(EXAMPLE_DEBUG) << "seg_w_aligned: " << seg_w_aligned;
  VLOG(EXAMPLE_DEBUG) << "seg_z_aligned: " << seg_z_aligned;
  VLOG(EXAMPLE_DEBUG) << "seg_channel_aligned: " << seg_channel_aligned;

  int32_t seg_kHWZ = seg_h * seg_w * seg_z;
  perception->seg3d.num_classes = seg_channel;
  perception->seg3d.seg.resize(seg_kHWZ);
  perception->seg3d.h = seg_h;
  perception->seg3d.w = seg_w;
  perception->seg3d.z = seg_z;

  for (int32_t h = 0; h < seg_h; h++) {
    int32_t h_idx = h * seg_w_aligned * seg_z_aligned * seg_channel_aligned;
    int32_t h_k = h * seg_w * seg_z;
    for (int32_t w = 0; w < seg_w; w++) {
      int32_t hw_idx = h_idx + w * seg_z_aligned * seg_channel_aligned;
      int32_t hw_k = h_k + w * seg_z;
      for (int32_t z = 0; z < seg_z; z++) {
        int32_t idx = hw_idx + z * seg_channel_aligned;
        int32_t k = hw_k + z;
        int8_t *data_tmp = seg_data + idx;
        float max_val = std::numeric_limits<float>::lowest();
        int32_t top_index = -1;
        for (int c = 0; c < seg_channel; c++) {
          float data = quanti_scale(data_tmp[c], seg_scale[0]);
          if (data > max_val) {
            max_val = data;
            top_index = c;
          }
        }
        perception->seg3d.seg[k] = top_index;
      }
    }
  }

  return 0;
}
