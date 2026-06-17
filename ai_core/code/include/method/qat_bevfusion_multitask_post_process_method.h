// Copyright (c) 2023 Horizon Robotics.All Rights Reserved.
//
// The material in this file is confidential and contains trade secrets
// of Horizon Robotics Inc. This is proprietary information owned by
// Horizon Robotics Inc. No part of this work may be disclosed,
// reproduced, copied, transmitted, or used in any way for any purpose,
// without the express written permission of Horizon Robotics Inc.

#ifndef DNN_AI_BENCHMARK_CODE_INCLUDE_METHOD_QAT_BEVFUSION_MULTITASK_POST_PROCESS_METHOD_H_
#define DNN_AI_BENCHMARK_CODE_INCLUDE_METHOD_QAT_BEVFUSION_MULTITASK_POST_PROCESS_METHOD_H_

#include <string>
#include <vector>

#include "glog/logging.h"
#include "method/method_data.h"
#include "method/post_process_method.h"

/**
 * Post process for BevFusion multitask: BEVFormer-style detection tensors +
 * occupancy / BEV parsing tensor. Fills Perception::LIDARMULTITASK (lidar3d +
 * lidarSeg) for raw_output multitask logging.
 *
 * Tensor layout defaults follow QATBevformerPostProcessMethod (det: base..base+3)
 * plus one OCC tensor at occ_tensor_idx (NHWC int8/int32, argmax on classes).
 */
class QATBevFusionMultitaskPostProcessMethod : public PostProcessMethod {
 public:
  int InitFromJsonString(const std::string &config) override;

  PerceptionPtr DoProcess(ImageTensor *image_tensor,
                          TensorVectorPtr &output_tensor) override;

  ~QATBevFusionMultitaskPostProcessMethod() override = default;

 private:
  int PostProcess(std::vector<hbDNNTensor> &tensors, ImageTensor *image_tensor,
                  Perception *perception);

  int topk_{300};
  float score_threshold_{0.1f};
  std::vector<float> bev_range_;
  std::vector<float> post_center_range_;
  std::vector<int> ori_shape_;
  std::vector<int> resize_shape_;

  int det_output_base_{0};
  int occ_tensor_idx_{4};

  int occ_resize_height_{512};
  int occ_resize_width_{512};
  /** If zero, scale is derived from resize / tensor spatial size. */
  float occ_scale_height_{0.f};
  float occ_scale_width_{0.f};
  bool occ_use_int32_{false};
};

#endif  // DNN_AI_BENCHMARK_CODE_INCLUDE_METHOD_QAT_BEVFUSION_MULTITASK_POST_PROCESS_METHOD_H_
