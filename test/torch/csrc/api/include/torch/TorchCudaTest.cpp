#include <ATen/ATen.h>
#include <gtest/gtest.h>
#include <torch/csrc/api/include/torch/cuda.h>

#include <string>

#include "src/file_manager.h"

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

// device_count
TEST_F(TorchCudaTest, DeviceCount) {
  // [DIFF] 用例级差异：device_count 在不同后端/构建下可能抛异常或返回 0。
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();
  file << "DeviceCount ";

  try {
    auto count = torch::cuda::device_count();
    file << "ok ";
    file << std::to_string(count) << " ";
    file << std::to_string(count >= 0 ? 1 : 0) << " ";
  } catch (const std::exception&) {
    file << "exception ";
  } catch (...) {
    file << "unknown_exception ";
  }
  file << "\n";
  file.saveFile();
}

// is_available
TEST_F(TorchCudaTest, IsAvailable) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "IsAvailable ";
  try {
    bool available = torch::cuda::is_available();
    file << "ok ";
    file << std::to_string(available ? 1 : 0) << " ";
  } catch (const std::exception&) {
    file << "exception ";
  } catch (...) {
    file << "unknown_exception ";
  }
  file << "\n";
  file.saveFile();
}

// device_count 和 is_available 一致性
TEST_F(TorchCudaTest, ConsistencyCheck) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "ConsistencyCheck ";
  try {
    auto count = torch::cuda::device_count();
    bool available = torch::cuda::is_available();
    bool consistent = (available && count > 0) || (!available && count == 0);
    file << "ok ";
    file << std::to_string(count) << " ";
    file << std::to_string(available ? 1 : 0) << " ";
    file << std::to_string(consistent ? 1 : 0) << " ";
  } catch (const std::exception&) {
    file << "exception ";
  } catch (...) {
    file << "unknown_exception ";
  }
  file << "\n";
  file.saveFile();
}

// at::cuda 命名空间别名
TEST_F(TorchCudaTest, AtCudaNamespace) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "AtCudaNamespace ";
  try {
    auto count = torch::cuda::device_count();
    bool available = torch::cuda::is_available();
    file << "ok ";
    file << std::to_string(count) << " ";
    file << std::to_string(available ? 1 : 0) << " ";
  } catch (const std::exception&) {
    file << "exception ";
  } catch (...) {
    file << "unknown_exception ";
  }
  file << "\n";
  file.saveFile();
}

// synchronize（仅在 CUDA 可用时有意义）
TEST_F(TorchCudaTest, Synchronize) {
  // [DIFF] 用例级差异：synchronize 强依赖可用 CUDA
  // stream，上下文差异会直接影响结果。
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "Synchronize ";
  try {
    torch::cuda::synchronize();
    file << "ok ";
  } catch (const std::exception&) {
    file << "exception ";
  } catch (...) {
    file << "unknown_exception ";
  }
  file << "\n";
  file.saveFile();
}

}  // namespace test
}  // namespace at
