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

static std::string getCudaUnavailableReason() {
  try {
    auto count = torch::cuda::device_count();
    if (count <= 0) {
      return "CUDA 运行时可访问，但没有可见 GPU 设备";
    }
    return "";
  } catch (const std::exception& e) {
    return std::string("CUDA 不可用：") + e.what() +
           "（注意：仅安装 CUDA Toolkit 不足，还需要 GPU 版 Paddle/libtorch）";
  } catch (...) {
    return "CUDA 不可用：未知异常（注意：仅安装 CUDA Toolkit 不足，还需要 GPU "
           "版 Paddle/libtorch）";
  }
}

// 安全地检测 CUDA 可用性：Paddle 未编译 CUDA 时 device_count() 会抛异常
static bool isCudaAvailable() {
  try {
    return torch::cuda::device_count() > 0;
  } catch (...) {
    return false;
  }
}

// device_count
TEST_F(TorchCudaTest, DeviceCount) {
  int64_t count;
  try {
    count = torch::cuda::device_count();
  } catch (const std::exception& e) {
    GTEST_SKIP()
        << std::string("CUDA 不可用：") << e.what()
        << "（注意：仅安装 CUDA Toolkit 不足，还需要 GPU 版 Paddle/libtorch）";
  }
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
  if (!isCudaAvailable()) {
    GTEST_SKIP() << getCudaUnavailableReason();
  }
  bool available = torch::cuda::is_available();
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << std::to_string(available ? 1 : 0) << " ";
  file.saveFile();
}

// device_count 和 is_available 一致性
TEST_F(TorchCudaTest, ConsistencyCheck) {
  if (!isCudaAvailable()) {
    GTEST_SKIP() << getCudaUnavailableReason();
  }
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
  if (!isCudaAvailable()) {
    GTEST_SKIP() << getCudaUnavailableReason();
  }
  auto count = torch::cuda::device_count();
  bool available = torch::cuda::is_available();
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << std::to_string(count) << " ";
  file << std::to_string(available ? 1 : 0) << " ";
  file.saveFile();
}

// synchronize（仅在 CUDA 可用时有意义）
TEST_F(TorchCudaTest, Synchronize) {
  if (!isCudaAvailable()) {
    GTEST_SKIP() << getCudaUnavailableReason();
  }
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  bool passed = true;
  try {
    torch::cuda::synchronize();
  } catch (...) {
    passed = false;
  }
  file << std::to_string(passed ? 1 : 0) << " ";
  file.saveFile();
}

}  // namespace test
}  // namespace at
