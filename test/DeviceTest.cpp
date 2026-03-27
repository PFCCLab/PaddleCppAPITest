#include <ATen/ATen.h>
#include <c10/core/Device.h>
#include <gtest/gtest.h>

#include <string>

#include "../src/file_manager.h"

extern paddle_api_test::ThreadSafeParam g_custom_param;

namespace at {
namespace test {

using paddle_api_test::FileManerger;
using paddle_api_test::ThreadSafeParam;

class DeviceCompatTest : public ::testing::Test {
 protected:
  void SetUp() override {}
};

TEST_F(DeviceCompatTest, DeviceStr) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();
  file << "DeviceStr ";

  c10::Device cpu_device(c10::kCPU);
  auto cpu_str = cpu_device.str();

  c10::Device cpu_device_0(c10::kCPU, 0);
  auto cpu_0_str = cpu_device_0.str();

  c10::Device cuda_device_0(c10::kCUDA, 0);
  auto cuda_0_str = cuda_device_0.str();

  c10::Device cuda_device_1(c10::kCUDA, 1);
  auto cuda_1_str = cuda_device_1.str();

  // [DIFF] PyTorch输出: cpu cpu:0 cuda:0 cuda:1
  // [DIFF] PaddlePaddle输出: cpu:0 cpu:0 gpu:0 gpu:1
  file << cpu_str << " ";
  file << cpu_0_str << " ";
  file << cuda_0_str << " ";
  file << cuda_1_str << " ";

  file << "\n";
  file.saveFile();
}

TEST_F(DeviceCompatTest, HasIndex) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "HasIndex ";

  c10::Device cpu_default(c10::kCPU);
  c10::Device cpu_0(c10::kCPU, 0);
  c10::Device cuda_default(c10::kCUDA);
  c10::Device cuda_1(c10::kCUDA, 1);

  bool cpu_default_has = cpu_default.has_index();
  bool cpu_0_has = cpu_0.has_index();
  bool cuda_default_has = cuda_default.has_index();
  bool cuda_1_has = cuda_1.has_index();

  file << std::to_string(cpu_default_has ? 1 : 0) << " ";
  file << std::to_string(cpu_0_has ? 1 : 0) << " ";
  file << std::to_string(cuda_default_has ? 1 : 0) << " ";
  file << std::to_string(cuda_1_has ? 1 : 0) << " ";

  file << "\n";
  file.saveFile();
}

}  // namespace test
}  // namespace at
