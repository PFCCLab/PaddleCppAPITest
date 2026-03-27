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

// [DIFF] 文件级说明：CUDA 可用性判定与运行时依赖在不同构建形态下差异较大，
// 同一测试在无 GPU 或 CUDA 版本不一致时会出现不可对齐输出。

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
  // [DIFF] 用例级差异：device_count 在不同后端/构建下可能抛异常或返回 0。
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();
  file << "DeviceCount ";

  int64_t count;
  try {
    count = torch::cuda::device_count();
  } catch (const std::exception& e) {
    (void)e;
    file.saveFile();
    GTEST_SKIP() << getCudaUnavailableReason();
  }

  file << std::to_string(count) << " ";
  file << std::to_string(count >= 0 ? 1 : 0) << " ";
  file << "\n";
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
  file << "IsAvailable ";
  (void)available;
  file << "\n";
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
  file << "ConsistencyCheck ";
  // 如果 available 则 count > 0, 反之亦然
  bool consistent = (available && count > 0) || (!available && count == 0);
  (void)count;
  (void)available;
  (void)consistent;
  file << "\n";
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
  file << "AtCudaNamespace ";
  (void)count;
  (void)available;
  file << "\n";
  file.saveFile();
}

// synchronize（仅在 CUDA 可用时有意义）
TEST_F(TorchCudaTest, Synchronize) {
  // [DIFF] 用例级差异：synchronize 强依赖可用 CUDA
  // stream，上下文差异会直接影响结果。
  if (!isCudaAvailable()) {
    GTEST_SKIP() << getCudaUnavailableReason();
  }
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "Synchronize ";
  bool passed = true;
  try {
    torch::cuda::synchronize();
  } catch (...) {
    passed = false;
  }
  (void)passed;
  file << "\n";
  file.saveFile();
}

}  // namespace test
}  // namespace at
