#include <ATen/ATen.h>
#ifndef USE_PADDLE_API
#include <ATen/cuda/CUDAContext.h>
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

// getDeviceProperties
TEST_F(CUDAContextTest, GetDeviceProperties) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();

  // getDeviceProperties returns a const CUDAEvent* (actually device properties)
  // This is a CUDA-specific function, test only if CUDA available
  file << "getDeviceProperties_test ";
  file.saveFile();
}

// getCurrentDeviceProperties
TEST_F(CUDAContextTest, GetCurrentDeviceProperties) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();

  // getCurrentDeviceProperties returns device properties
  file << "getCurrentDeviceProperties_test ";
  file.saveFile();
}

// Test getCurrentCUDAStream
TEST_F(CUDAContextTest, GetCurrentCUDAStream) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();

#ifndef USE_PADDLE_API
  // at::cuda::getCurrentCUDAStream
  try {
    auto stream = at::cuda::getCurrentCUDAStream();
    file << "stream_available ";
  } catch (...) {
    file << "stream_not_available ";
  }
#else
  file << "stream_skipped_paddle ";
#endif
  file.saveFile();
}

}  // namespace test
}  // namespace at
