#include <ATen/ATen.h>
#include <ATen/cuda/CUDADataType.h>
#include <ATen/cuda/EmptyTensor.h>
#include <gtest/gtest.h>

#include <string>

#include "../src/file_manager.h"

extern paddle_api_test::ThreadSafeParam g_custom_param;

namespace at {
namespace test {

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
}

// getCudaDataType with BFloat16
TEST_F(CUDADataTypeTest, GetCudaDataTypeBFloat16) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();

  file << std::to_string(at::getCudaDataType(c10::ScalarType::BFloat16)) << " ";
  file.saveFile();
}

// getCudaDataType with Complex
TEST_F(CUDADataTypeTest, GetCudaDataTypeComplex) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();

  file << std::to_string(at::getCudaDataType(c10::ScalarType::ComplexFloat))
       << " ";
  file << std::to_string(at::getCudaDataType(c10::ScalarType::ComplexDouble))
       << " ";
  file.saveFile();
}

// empty_cuda
TEST_F(CUDADataTypeTest, EmptyCUDA) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();

  // empty_cuda with IntArrayRef size
  try {
    at::Tensor t = at::cuda::empty_cuda({2, 3, 4}, c10::ScalarType::Float, 0);
    file << "cuda_empty ";
  } catch (...) {
    file << "cuda_not_available ";
  }
  file.saveFile();
}

// empty_cuda with different dtypes
TEST_F(CUDADataTypeTest, EmptyCudaDifferentDtype) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();

  try {
    at::Tensor t = at::cuda::empty_cuda({2, 3}, c10::ScalarType::Int, 0);
    file << "cuda_empty_int ";
  } catch (...) {
    file << "cuda_not_available ";
  }
  file.saveFile();
}

}  // namespace test
}  // namespace at
