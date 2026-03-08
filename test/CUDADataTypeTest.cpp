#include <ATen/ATen.h>
#include <gtest/gtest.h>

#include <string>

#include "../src/file_manager.h"

// Only include CUDA headers when the full CUDA toolkit is available.
#if defined(__has_include) && \
    __has_include(<cuda.h>) && __has_include(<library_types.h>)
#define HAS_CUDA 1
#include <ATen/cuda/CUDADataType.h>
#include <ATen/cuda/EmptyTensor.h>
#endif

extern paddle_api_test::ThreadSafeParam g_custom_param;

namespace paddle_cuda_api_test {

using paddle_api_test::FileManerger;
using paddle_api_test::ThreadSafeParam;

class CUDADataTypeTest : public ::testing::Test {
 protected:
  void SetUp() override {}
};

// getCudaDataType
TEST_F(CUDADataTypeTest, GetCudaDataType) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();

#if !defined(HAS_CUDA)
  GTEST_SKIP() << "CUDA not available";
#else
  // Test getCudaDataType for various ScalarTypes
  file << std::to_string(at::getCudaDataType(c10::ScalarType::Float)) << " ";
  file << std::to_string(at::getCudaDataType(c10::ScalarType::Double)) << " ";
  file << std::to_string(at::getCudaDataType(c10::ScalarType::Int)) << " ";
  file << std::to_string(at::getCudaDataType(c10::ScalarType::Long)) << " ";
  file << std::to_string(at::getCudaDataType(c10::ScalarType::Half)) << " ";
  file << std::to_string(at::getCudaDataType(c10::ScalarType::Bool)) << " ";
  file << std::to_string(at::getCudaDataType(c10::ScalarType::Byte)) << " ";
  file << std::to_string(at::getCudaDataType(c10::ScalarType::Char)) << " ";
  file << std::to_string(at::getCudaDataType(c10::ScalarType::Short)) << " ";
  file.saveFile();
#endif
}

// getCudaDataType with BFloat16
TEST_F(CUDADataTypeTest, GetCudaDataTypeBFloat16) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();

#if !defined(HAS_CUDA)
  GTEST_SKIP() << "CUDA not available";
#else
  file << std::to_string(at::getCudaDataType(c10::ScalarType::BFloat16)) << " ";
  file.saveFile();
#endif
}

// getCudaDataType with Complex
TEST_F(CUDADataTypeTest, GetCudaDataTypeComplex) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();

#if !defined(HAS_CUDA)
  GTEST_SKIP() << "CUDA not available";
#else
  file << std::to_string(at::getCudaDataType(c10::ScalarType::ComplexFloat))
       << " ";
  file << std::to_string(at::getCudaDataType(c10::ScalarType::ComplexDouble))
       << " ";
  file.saveFile();
#endif
}

// empty_cuda
TEST_F(CUDADataTypeTest, EmptyCUDA) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();

#if !defined(HAS_CUDA)
  GTEST_SKIP() << "CUDA not available";
#else
  // empty_cuda with IntArrayRef size
  try {
    at::Tensor t = at::cuda::empty_cuda({2, 3, 4}, c10::ScalarType::Float, 0);
    file << "cuda_empty ";
  } catch (...) {
    file << "cuda_not_available ";
  }
  file.saveFile();
#endif
}

// empty_cuda with different dtypes
TEST_F(CUDADataTypeTest, EmptyCudaDifferentDtype) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();

#if !defined(HAS_CUDA)
  GTEST_SKIP() << "CUDA not available";
#else
  try {
    at::Tensor t = at::cuda::empty_cuda({2, 3}, c10::ScalarType::Int, 0);
    file << "cuda_empty_int ";
  } catch (...) {
    file << "cuda_not_available ";
  }
  file.saveFile();
#endif
}

}  // namespace paddle_cuda_api_test
