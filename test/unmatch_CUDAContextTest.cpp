#include <ATen/ATen.h>
#if USE_PADDLE_API != 1
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAContextLight.h>
#endif
#include <gtest/gtest.h>

#include <string>

#include "../src/file_manager.h"

extern paddle_api_test::ThreadSafeParam g_custom_param;

namespace at {
namespace test {

using paddle_api_test::FileManerger;
using paddle_api_test::ThreadSafeParam;

class CUDAContextTest : public ::testing::Test {
 protected:
  void SetUp() override {}
};

// [DIFF] 文件级说明：CUDAContext 系列接口在两端对“无 CUDA / 属性结构 /
// 流可用性” 的返回语义不同，测试需要条件分支与占位输出，因此保留在 unmatch。

// getDeviceProperties
TEST_F(CUDAContextTest, GetDeviceProperties) {
  // [DIFF] 用例级差异：无卡或运行时不可用时，两端异常/返回值语义不一致。
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();
  file << "GetDeviceProperties ";

#if USE_PADDLE_API != 1
  // PyTorch API: at::cuda::getDeviceProperties(c10::DeviceIndex device)
  try {
    auto* prop = at::cuda::getDeviceProperties(0);
    if (prop) {
      // Output pointer address as identifier
      file << reinterpret_cast<uintptr_t>(prop) << " ";
      // Output key device properties
      file << prop->major << " ";
      file << prop->minor << " ";
      file << prop->multiProcessorCount << " ";
    } else {
      file << "null ";
    }
  } catch (const std::exception& e) {
    file << "exception ";
  }
#else
// [DIFF] 问题行：Paddle 分支额外依赖 PADDLE_WITH_CUDA，
// 与 Torch 侧同名 API 的可达条件不同。
#ifdef PADDLE_WITH_CUDA
  // Paddle API: at::cuda::getDeviceProperties
  try {
    auto* prop = at::cuda::getDeviceProperties(0);
    if (prop) {
      // Output pointer address as identifier
      file << reinterpret_cast<uintptr_t>(prop) << " ";
      // Also output some key properties
      file << prop->major << " ";
      file << prop->minor << " ";
      file << prop->multiProcessorCount << " ";
    } else {
      file << "null ";
    }
  } catch (const std::exception& e) {
    file << "exception ";
  }
#else
  file << "api_not_available ";
#endif
#endif
  file << "\n";
  file.saveFile();
}

// getCurrentDeviceProperties
TEST_F(CUDAContextTest, GetCurrentDeviceProperties) {
  // [DIFF] 用例级差异：current device 属性在异常文案/可用判定上与 Torch
  // 不一致。
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "GetCurrentDeviceProperties ";

#if USE_PADDLE_API != 1
  // PyTorch API: at::cuda::getCurrentDeviceProperties
  try {
    auto* prop = at::cuda::getCurrentDeviceProperties();
    if (prop) {
      file << reinterpret_cast<uintptr_t>(prop) << " ";
      file << prop->major << " ";
      file << prop->minor << " ";
      file << prop->multiProcessorCount << " ";
    } else {
      file << "null ";
    }
  } catch (const std::exception& e) {
    file << "exception ";
  }
#else
#ifdef PADDLE_WITH_CUDA
  // Paddle API: at::cuda::getCurrentDeviceProperties
  try {
    auto* prop = at::cuda::getCurrentDeviceProperties();
    if (prop) {
      file << reinterpret_cast<uintptr_t>(prop) << " ";
      file << prop->major << " ";
      file << prop->minor << " ";
      file << prop->multiProcessorCount << " ";
    } else {
      file << "null ";
    }
  } catch (const std::exception& e) {
    file << "exception ";
  }
#else
  file << "api_not_available ";
#endif
#endif
  file << "\n";
  file.saveFile();
}

// Test getCurrentCUDAStream
TEST_F(CUDAContextTest, GetCurrentCUDAStream) {
  // [DIFF] 用例级差异：流 API 在 Paddle 分支被跳过，Torch 分支实际调用。
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "GetCurrentCUDAStream ";

#if USE_PADDLE_API != 1
  // at::cuda::getCurrentCUDAStream
  try {
    auto stream = at::cuda::getCurrentCUDAStream();
    (void)stream;
    file << "stream_available ";
  } catch (...) {
    file << "stream_not_available ";
  }
#else
  try {
    auto stream = at::cuda::getCurrentCUDAStream();
    (void)stream;
    file << "stream_available ";
  } catch (...) {
    file << "stream_not_available ";
  }
#endif
  file << "\n";
  file.saveFile();
}

}  // namespace test
}  // namespace at
