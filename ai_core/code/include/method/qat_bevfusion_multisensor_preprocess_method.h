// Copyright (c) 2023 Horizon Robotics.All Rights Reserved.
//
// The material in this file is confidential and contains trade secrets
// of Horizon Robotics Inc. This is proprietary information owned by
// Horizon Robotics Inc. No part of this work may be disclosed,
// reproduced, copied, transmitted, or used in any way for any purpose,
// without the express written permission of Horizon Robotics Inc.

#ifndef DNN_AI_BENCHMARK_CODE_INCLUDE_METHOD_QAT_BEVFUSION_MULTISENSOR_PREPROCESS_METHOD_H_
#define DNN_AI_BENCHMARK_CODE_INCLUDE_METHOD_QAT_BEVFUSION_MULTISENSOR_PREPROCESS_METHOD_H_

#include <memory>
#include <string>
#include <vector>

#include "glog/logging.h"
#include "method/method_data.h"
#include "method/preprocess_method.h"

/**
 * Preprocess for BevFusion (PointPillar features + coors from lidar, remaining
 * inputs from per-sample binary dumps aligned with compiled model).
 *
 * Each line in the raw_data list file must be a directory containing:
 *   - lidar float32 point cloud bin (see lidar_bin_name)
 *   - auxiliary bins (see auxiliary_bin_names) matching model inputs 2..N-1
 */
class QATBevFusionMultiSensorPreProcessMethod : public PreProcessMethod {
 public:
  int32_t InitFromJsonString(const std::string &config) override;

  int32_t DoProcess(std::string path, int32_t input_count,
                    ImageTensor *image_tensor) override;

  ~QATBevFusionMultiSensorPreProcessMethod() override = default;

 private:
  static bool IsDirectory(const std::string &path);
  static std::string JoinPath(const std::string &dir, const std::string &name);
  static std::string ReadWholeFile(const std::string &path);
  static bool TryLoadExactAlignedBin(hbDNNTensor *tensor,
                                       const std::string &bin_path);

  std::unique_ptr<PreProcessMethod> pillar_preprocess_;
  std::string pillar_config_file_;
  std::string lidar_bin_name_{"lidar_points.bin"};
  std::vector<std::string> auxiliary_bin_names_;
};

#endif  // DNN_AI_BENCHMARK_CODE_INCLUDE_METHOD_QAT_BEVFUSION_MULTISENSOR_PREPROCESS_METHOD_H_
