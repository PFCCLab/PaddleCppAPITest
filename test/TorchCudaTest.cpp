#include <ATen/ATen.h>
#include <gtest/gtest.h>
#include <torch/csrc/api/include/torch/cuda.h>

#include <string>

#include "../src/file_manager.h"

extern paddle_api_test::ThreadSafeParam g_custom_param;

namespace at {
namespace test {

using paddle_api_test::FileManerger;
using paddle_api_test::ThreadSafeParam;

class TorchCudaTest : public ::testing::Test {
 protected:
  void SetUp() override {}
};

// device_count
TEST_F(TorchCudaTest, DeviceCount) {
  auto count = torch::cuda::device_count();
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();
  file << std::to_string(count) << " ";
  // device_count 应非负
  file << std::to_string(count >= 0 ? 1 : 0) << " ";
  file.saveFile();
}

// is_available
TEST_F(TorchCudaTest, IsAvailable) {
  bool available = torch::cuda::is_available();
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << std::to_string(available ? 1 : 0) << " ";
  file.saveFile();
}

// device_count 和 is_available 一致性
TEST_F(TorchCudaTest, ConsistencyCheck) {
  auto count = torch::cuda::device_count();
  bool available = torch::cuda::is_available();
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  // 如果 available 则 count > 0, 反之亦然
  bool consistent = (available && count > 0) || (!available && count == 0);
  file << std::to_string(consistent ? 1 : 0) << " ";
  file.saveFile();
}

// at::cuda 命名空间别名
TEST_F(TorchCudaTest, AtCudaNamespace) {
  auto count = at::cuda::device_count();
  bool available = at::cuda::is_available();
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << std::to_string(count) << " ";
  file << std::to_string(available ? 1 : 0) << " ";
  file.saveFile();
}

// synchronize（仅在 CUDA 可用时有意义，但 API 应始终可调用）
TEST_F(TorchCudaTest, Synchronize) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  if (torch::cuda::is_available()) {
    bool passed = true;
    try {
      torch::cuda::synchronize();
    } catch (...) {
      passed = false;
    }
    file << std::to_string(passed ? 1 : 0) << " ";
  } else {
    // CUDA 不可用时跳过
    file << "skip ";
  }
  file.saveFile();
}

}  // namespace test
}  // namespace at
