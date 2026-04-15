// Copyright (c) 2023 Horizon Robotics.All Rights Reserved.
//
// The material in this file is confidential and contains trade secrets
// of Horizon Robotics Inc. This is proprietary information owned by
// Horizon Robotics Inc. No part of this work may be disclosed,
// reproduced, copied, transmitted, or used in any way for any purpose,
// without the express written permission of Horizon Robotics Inc.

#ifndef DNN_AI_BENCHMARK_CODE_INCLUDE_METHOD_QAT_FLASHOCC_POST_PROCESS_METHOD_H_
#define DNN_AI_BENCHMARK_CODE_INCLUDE_METHOD_QAT_FLASHOCC_POST_PROCESS_METHOD_H_

#include <string>
#include <vector>

#include "glog/logging.h"
#include "method/method_data.h"
#include "method/post_process_method.h"

/**
 * Method for post processing
 */
class QATFlashoccPostProcessMethod : public PostProcessMethod {
 public:
  /**
   * Init post process from json string
   * @param[in] config: config json string
   * @return 0 if success
   */
  int InitFromJsonString(const std::string &config) override;

  PerceptionPtr DoProcess(ImageTensor *image_tensor,
                          TensorVectorPtr &output_tensor) override;

  ~QATFlashoccPostProcessMethod() override = default;

 private:
  int PostProcess(std::vector<hbDNNTensor> &tensors, ImageTensor *image_tensor,
                  Perception *perception);

 private:
  std::vector<int> ori_shape_{900, 1600};
  std::vector<int> resize_shape_{540, 960};
};

#endif  // DNN_AI_BENCHMARK_CODE_INCLUDE_METHOD_QAT_FLASHOCC_POST_PROCESS_METHOD_H_
