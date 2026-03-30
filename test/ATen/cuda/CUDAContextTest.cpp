#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <gtest/gtest.h>

#include <string>

#include "src/file_manager.h"

extern paddle_api_test::ThreadSafeParam g_custom_param;

namespace at {
namespace test {

using paddle_api_test::FileManerger;
using paddle_api_test::ThreadSafeParam;

class CUDAContextTest : public ::testing::Test {
 protected:
  static bool HasCudaRuntime() {
    try {
      return at::cuda::is_available();
    } catch (const std::exception&) {
      return false;
    }
  }

  static void WriteCudaRuntimeUnavailable(FileManerger* file) {
    *file << "cuda_runtime_unavailable ";
    *file << "\n";
    file->saveFile();
  }

  template <typename DevicePropT>
  static void WriteDevicePropertiesSummary(FileManerger* file,
                                           DevicePropT* prop) {
    if (prop == nullptr) {
      *file << "null ";
      return;
    }
    *file << "1 ";
    *file << prop->major << " ";
    *file << prop->minor << " ";
    *file << prop->multiProcessorCount << " ";
  }
};

TEST_F(CUDAContextTest, GetDeviceProperties) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();
  file << "GetDeviceProperties ";

  if (!HasCudaRuntime()) {
    WriteCudaRuntimeUnavailable(&file);
    return;
  }

  c10::cuda::CUDAGuard guard(c10::Device(c10::DeviceType::CUDA, 0));
  try {
    WriteDevicePropertiesSummary(&file, at::cuda::getDeviceProperties(0));
  } catch (const std::exception&) {
    file << "exception ";
  }

  file << "\n";
  file.saveFile();
}

TEST_F(CUDAContextTest, GetCurrentDeviceProperties) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "GetCurrentDeviceProperties ";

  if (!HasCudaRuntime()) {
    WriteCudaRuntimeUnavailable(&file);
    return;
  }

  c10::cuda::CUDAGuard guard(c10::Device(c10::DeviceType::CUDA, 0));
  try {
    WriteDevicePropertiesSummary(&file, at::cuda::getCurrentDeviceProperties());
  } catch (const std::exception&) {
    file << "exception ";
  }

  file << "\n";
  file.saveFile();
}

TEST_F(CUDAContextTest, GetCurrentCUDAStream) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "GetCurrentCUDAStream ";

  if (!HasCudaRuntime()) {
    WriteCudaRuntimeUnavailable(&file);
    return;
  }

  c10::cuda::CUDAGuard guard(c10::Device(c10::DeviceType::CUDA, 0));
  try {
    auto default_stream = at::cuda::getDefaultCUDAStream(0);
    at::cuda::setCurrentCUDAStream(default_stream);
    auto current_stream = at::cuda::getCurrentCUDAStream();
    auto explicit_stream = at::cuda::getCurrentCUDAStream(0);

    file << static_cast<int>(current_stream.device_index()) << " ";
    file << static_cast<int>(current_stream.device_type()) << " ";
    file << (current_stream == explicit_stream) << " ";
    file << (current_stream == default_stream) << " ";
  } catch (const std::exception&) {
    file << "exception ";
  }

  file << "\n";
  file.saveFile();
}

}  // namespace test
}  // namespace at
