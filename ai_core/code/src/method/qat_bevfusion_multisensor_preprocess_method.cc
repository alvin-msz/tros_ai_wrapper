// Copyright (c) 2023 Horizon Robotics.All Rights Reserved.
//
// The material in this file is confidential and contains trade secrets
// of Horizon Robotics Inc. This is proprietary information owned by
// Horizon Robotics Inc. No part of this work may be disclosed,
// reproduced, copied, transmitted, or used in any way for any purpose,
// without the express written permission of Horizon Robotics Inc.

#include "method/qat_bevfusion_multisensor_preprocess_method.h"

#include <fstream>
#include <iomanip>
#include <sstream>
#include <sys/stat.h>

#include "base/common_def.h"
#include "method/method_factory.h"
#include "rapidjson/document.h"
#include "utils/tensor_utils.h"

DEFINE_AND_REGISTER_METHOD(QATBevFusionMultiSensorPreProcessMethod);

bool QATBevFusionMultiSensorPreProcessMethod::IsDirectory(
    const std::string &path) {
  struct stat st {};
  if (stat(path.c_str(), &st) != 0) {
    return false;
  }
  return S_ISDIR(st.st_mode);
}

std::string QATBevFusionMultiSensorPreProcessMethod::JoinPath(
    const std::string &dir, const std::string &name) {
  if (dir.empty()) {
    return name;
  }
  if (!name.empty() && name.front() == '/') {
    return name;
  }
  if (dir.back() == '/') {
    return dir + name;
  }
  return dir + "/" + name;
}

std::string QATBevFusionMultiSensorPreProcessMethod::ReadWholeFile(
    const std::string &path) {
  std::ifstream ifs(path.c_str(), std::ios::in | std::ios::binary);
  if (!ifs) {
    return {};
  }
  std::ostringstream ss;
  ss << ifs.rdbuf();
  return ss.str();
}

bool QATBevFusionMultiSensorPreProcessMethod::TryLoadExactAlignedBin(
    hbDNNTensor *tensor, const std::string &bin_path) {
  std::ifstream ifs(bin_path.c_str(), std::ios::in | std::ios::binary);
  if (!ifs) {
    return false;
  }
  ifs.seekg(0, std::ios::end);
  auto len = static_cast<size_t>(ifs.tellg());
  ifs.seekg(0, std::ios::beg);
  size_t aligned = static_cast<size_t>(tensor->properties.alignedByteSize);
  if (len != aligned) {
    return false;
  }
  ifs.read(reinterpret_cast<char *>(tensor->sysMem.virAddr),
           static_cast<std::streamsize>(len));
  hbUCPMemFlush(&tensor->sysMem, HB_SYS_MEM_CACHE_CLEAN);
  return !ifs.fail() &&
         static_cast<size_t>(ifs.gcount()) == len;
}

int32_t QATBevFusionMultiSensorPreProcessMethod::InitFromJsonString(
    const std::string &config) {
  VLOG(EXAMPLE_DEBUG) << "QATBevFusionMultiSensorPreProcessMethod Json string:"
                      << config.data();

  rapidjson::Document document;
  document.Parse(config.data());
  if (document.HasParseError()) {
    VLOG(EXAMPLE_SYSTEM) << "Parsing config file failed";
    return -1;
  }

  if (document.HasMember("pillar_config_file")) {
    pillar_config_file_ = document["pillar_config_file"].GetString();
  } else {
    VLOG(EXAMPLE_SYSTEM) << "Missing pillar_config_file";
    return -1;
  }

  if (document.HasMember("lidar_bin_name")) {
    lidar_bin_name_ = document["lidar_bin_name"].GetString();
  }

  if (document.HasMember("auxiliary_bin_names")) {
    auto arr = document["auxiliary_bin_names"].GetArray();
    auxiliary_bin_names_.clear();
    for (rapidjson::SizeType i = 0; i < arr.Size(); ++i) {
      auxiliary_bin_names_.push_back(arr[i].GetString());
    }
  }

  Method *raw =
      MethodFactory::GetInstance()->GetMethod("QATCenterPointPreProcessMethod");
  if (raw == nullptr) {
    VLOG(EXAMPLE_SYSTEM) << "QATCenterPointPreProcessMethod not registered";
    return -1;
  }
  pillar_preprocess_.reset(dynamic_cast<PreProcessMethod *>(raw));
  if (pillar_preprocess_ == nullptr) {
    VLOG(EXAMPLE_SYSTEM) << "Failed to cast pillar preprocess";
    return -1;
  }

  pillar_preprocess_->SetModelHandle(dnn_handle_);
  std::string pillar_json = ReadWholeFile(pillar_config_file_);
  if (pillar_json.empty()) {
    VLOG(EXAMPLE_SYSTEM) << "Read pillar_config_file failed: "
                         << pillar_config_file_;
    return -1;
  }
  if (pillar_preprocess_->InitFromJsonString(pillar_json) != 0) {
    VLOG(EXAMPLE_SYSTEM) << "Init pillar preprocess failed";
    return -1;
  }

  return 0;
}

int32_t QATBevFusionMultiSensorPreProcessMethod::DoProcess(
    std::string path, int32_t input_count, ImageTensor *image_tensor) {
  if (input_count < 3) {
    VLOG(EXAMPLE_SYSTEM)
        << "BevFusion expects at least 3 inputs (features, coors, ...), got "
        << input_count;
    return -1;
  }
  if (static_cast<int>(auxiliary_bin_names_.size()) != input_count - 2) {
    VLOG(EXAMPLE_SYSTEM)
        << "auxiliary_bin_names size must equal input_count - 2. Expected "
        << (input_count - 2) << ", got " << auxiliary_bin_names_.size();
    return -1;
  }

  if (!IsDirectory(path)) {
    VLOG(EXAMPLE_SYSTEM)
        << "BevFusion preprocess expects a sample directory path: " << path;
    return -1;
  }

  const std::string lidar_path = JoinPath(path, lidar_bin_name_);

  if (pillar_preprocess_->DoProcess(lidar_path, input_count, image_tensor) !=
      0) {
    VLOG(EXAMPLE_SYSTEM) << "Pillar preprocess failed for " << lidar_path;
    return -1;
  }
  // Pillar path sets pre_duration after lidar read; keep that as pure preprocess
  // (no auxiliary bin I/O).
  const uint64_t pure_pre_us = image_tensor->pre_duration;

  auto &tensors = image_tensor->tensors;
  for (int i = 2; i < input_count; ++i) {
    const std::string bin_path = JoinPath(path, auxiliary_bin_names_[i - 2]);
    if (TryLoadExactAlignedBin(&tensors[i], bin_path)) {
      continue;
    }
    release_tensor(&tensors[i]);
    std::string mutable_path = bin_path;
    if (prepare_batch_tensor_and_quanti(&tensors[i], mutable_path) != 0) {
      VLOG(EXAMPLE_SYSTEM)
          << "Load auxiliary input failed (aligned bin and float quant both "
             "failed): "
          << bin_path;
      return -1;
    }
  }

  image_tensor->pre_duration = pure_pre_us;
  // VLOG(EXAMPLE_REPORT) << std::fixed << std::setprecision(3)
  //                       << "BevFusion multisensor preprocess frame_id="
  //                       << image_tensor->frame_id
  //                       << " pre_ms=" << pure_pre_us / 1000.0;
  return 0;
}
