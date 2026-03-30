#include <ATen/ATen.h>
#include <gtest/gtest.h>

#include <string>

#include "src/file_manager.h"

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
  file << "GetCudaDataType ";

#if !defined(HAS_CUDA)
  GTEST_SKIP() << "CUDA not available";
#else
  // Both libtorch and Paddle compat headers expose ScalarTypeToCudaDataType
  // under at::cuda. The old at::getCudaDataType(...) symbol is unavailable.
  file << std::to_string(
              at::cuda::ScalarTypeToCudaDataType(c10::ScalarType::Float))
       << " ";
  file << std::to_string(
              at::cuda::ScalarTypeToCudaDataType(c10::ScalarType::Double))
       << " ";
  file << std::to_string(
              at::cuda::ScalarTypeToCudaDataType(c10::ScalarType::Int))
       << " ";
  file << std::to_string(
              at::cuda::ScalarTypeToCudaDataType(c10::ScalarType::Long))
       << " ";
  file << std::to_string(
              at::cuda::ScalarTypeToCudaDataType(c10::ScalarType::Half))
       << " ";
  // Bool is unsupported on both libtorch and Paddle compat, so record the
  // shared exception branch explicitly instead of treating it as a Paddle-only
  // diff.
  try {
    (void)at::cuda::ScalarTypeToCudaDataType(c10::ScalarType::Bool);
    file << "bool_supported ";
  } catch (...) {
    file << "bool_unsupported ";
  }
  file << std::to_string(
              at::cuda::ScalarTypeToCudaDataType(c10::ScalarType::Byte))
       << " ";
  file << std::to_string(
              at::cuda::ScalarTypeToCudaDataType(c10::ScalarType::Char))
       << " ";
  file << std::to_string(
              at::cuda::ScalarTypeToCudaDataType(c10::ScalarType::Short))
       << " ";
  file << "\n";
  file.saveFile();
#endif
}

// getCudaDataType with BFloat16
TEST_F(CUDADataTypeTest, GetCudaDataTypeBFloat16) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "GetCudaDataTypeBFloat16 ";

#if !defined(HAS_CUDA)
  GTEST_SKIP() << "CUDA not available";
#else
  file << std::to_string(
              at::cuda::ScalarTypeToCudaDataType(c10::ScalarType::BFloat16))
       << " ";
  file << "\n";
  file.saveFile();
#endif
}

// getCudaDataType with Complex
TEST_F(CUDADataTypeTest, GetCudaDataTypeComplex) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "GetCudaDataTypeComplex ";

#if !defined(HAS_CUDA)
  GTEST_SKIP() << "CUDA not available";
#else
  file << std::to_string(
              at::cuda::ScalarTypeToCudaDataType(c10::ScalarType::ComplexFloat))
       << " ";
  file << std::to_string(at::cuda::ScalarTypeToCudaDataType(
              c10::ScalarType::ComplexDouble))
       << " ";
  file << "\n";
  file.saveFile();
#endif
}

// empty_cuda
// The observed branch depends on whether the current machine can materialize a
// CUDA tensor at runtime. Both binaries run in the same environment, so
// result_cmp should see the same token on both sides.
TEST_F(CUDADataTypeTest, EmptyCUDA) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "EmptyCUDA ";

#if !defined(HAS_CUDA)
  GTEST_SKIP() << "CUDA not available";
#else
  // Both libtorch and Paddle compat headers expose empty_cuda under at::detail.
  try {
    at::Tensor t = at::detail::empty_cuda({2, 3, 4},
                                          c10::ScalarType::Float,
                                          at::Device(at::kCUDA, 0),
                                          std::nullopt);
    (void)t;
    file << "cuda_empty ";
  } catch (...) {
    file << "cuda_not_available ";
  }
  file << "\n";
  file.saveFile();
#endif
}

// empty_cuda with different dtypes
// Same as EmptyCUDA: the token reflects the shared runtime environment rather
// than a semantic mismatch in the compat API itself.
TEST_F(CUDADataTypeTest, EmptyCudaDifferentDtype) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "EmptyCudaDifferentDtype ";

#if !defined(HAS_CUDA)
  GTEST_SKIP() << "CUDA not available";
#else
  try {
    at::Tensor t = at::detail::empty_cuda(
        {2, 3}, c10::ScalarType::Int, at::Device(at::kCUDA, 0), std::nullopt);
    (void)t;
    file << "cuda_empty_int ";
  } catch (...) {
    file << "cuda_not_available ";
  }
  file << "\n";
  file.saveFile();
#endif
}

}  // namespace paddle_cuda_api_test
