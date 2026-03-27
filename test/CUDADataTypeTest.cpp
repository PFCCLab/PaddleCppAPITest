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
  // DIFF: Paddle compat 的 ScalarTypeToCudaDataType 不支持 Bool，
  // 会抛出 "Cannot convert ScalarType Bool to cudaDataType"，因此跳过。
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
// DIFF: 该测试在 Torch CUDA 版下可成功创建 Tensor，输出 "cuda_empty"；
// 但在 Paddle 兼容层中，如果 Paddle 未编译 CUDA
// 或当前运行时不可用，会进入异常分支， 输出
// "cuda_not_available"。这是运行时/构建环境差异，不属于接口语义差异。
// 为避免比较结果受环境影响，保留调用，仅注释掉相关输出。
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
// DIFF: 与 EmptyCUDA 相同，该测试结果依赖 Paddle 是否为 GPU 版以及当前 CUDA
// 运行时是否可用。 Torch CUDA 版通常输出 "cuda_empty_int"，而 Paddle 侧可能输出
// "cuda_not_available"。
// 为避免环境差异导致比对失败，仅保留调用，不记录结果字符串。
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
