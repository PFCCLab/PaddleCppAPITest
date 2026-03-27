#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <c10/util/accumulate.h>
#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "../src/file_manager.h"

extern paddle_api_test::ThreadSafeParam g_custom_param;

namespace at {
namespace test {

using paddle_api_test::FileManerger;
using paddle_api_test::ThreadSafeParam;

class AccumulateTypeTest : public ::testing::Test {
 protected:
  void SetUp() override {}
};

// sum_integers（container 版）
TEST_F(AccumulateTypeTest, SumIntegersContainer) {
  std::vector<int64_t> vec = {1, 2, 3, 4, 5};
  int64_t result = c10::sum_integers(vec);

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();
  file << "SumIntegersContainer ";
  file << std::to_string(result) << " ";
  file << "\n";
  file.saveFile();
}

// sum_integers（iterator 版）
TEST_F(AccumulateTypeTest, SumIntegersIterator) {
  std::vector<int64_t> vec = {10, 20, 30};
  int64_t result = c10::sum_integers(vec.begin(), vec.end());

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "SumIntegersIterator ";
  file << std::to_string(result) << " ";
  file << "\n";
  file.saveFile();
}

// sum_integers 空容器
TEST_F(AccumulateTypeTest, SumIntegersEmpty) {
  std::vector<int64_t> vec;
  int64_t result = c10::sum_integers(vec);

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "SumIntegersEmpty ";
  file << std::to_string(result) << " ";
  file << "\n";
  file.saveFile();
}

// multiply_integers（container 版）
TEST_F(AccumulateTypeTest, MultiplyIntegersContainer) {
  std::vector<int64_t> vec = {2, 3, 4};
  int64_t result = c10::multiply_integers(vec);

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "MultiplyIntegersContainer ";
  file << std::to_string(result) << " ";
  file << "\n";
  file.saveFile();
}

// multiply_integers（iterator 版）
TEST_F(AccumulateTypeTest, MultiplyIntegersIterator) {
  std::vector<int64_t> vec = {5, 6, 7};
  int64_t result = c10::multiply_integers(vec.begin(), vec.end());

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "MultiplyIntegersIterator ";
  file << std::to_string(result) << " ";
  file << "\n";
  file.saveFile();
}

// multiply_integers 空容器（应返回 1）
TEST_F(AccumulateTypeTest, MultiplyIntegersEmpty) {
  std::vector<int64_t> vec;
  int64_t result = c10::multiply_integers(vec);

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "MultiplyIntegersEmpty ";
  file << std::to_string(result) << " ";
  file << "\n";
  file.saveFile();
}

// multiply_integers 含零（应返回 0）
TEST_F(AccumulateTypeTest, MultiplyIntegersWithZero) {
  std::vector<int64_t> vec = {2, 0, 4};
  int64_t result = c10::multiply_integers(vec);

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "MultiplyIntegersWithZero ";
  file << std::to_string(result) << " ";
  file << "\n";
  file.saveFile();
}

// numelements_from_dim
TEST_F(AccumulateTypeTest, NumElementsFromDim) {
  std::vector<int64_t> dims = {2, 3, 4, 5};
  // from_dim(0) = 2*3*4*5 = 120
  // from_dim(1) = 3*4*5 = 60
  // from_dim(2) = 4*5 = 20
  // from_dim(3) = 5
  // from_dim(4) = 1 (beyond dims)

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "NumElementsFromDim ";
  file << std::to_string(c10::numelements_from_dim(0, dims)) << " ";
  file << std::to_string(c10::numelements_from_dim(1, dims)) << " ";
  file << std::to_string(c10::numelements_from_dim(2, dims)) << " ";
  file << std::to_string(c10::numelements_from_dim(3, dims)) << " ";
  file << std::to_string(c10::numelements_from_dim(5, dims)) << " ";
  file << "\n";
  file.saveFile();
}

// numelements_to_dim
TEST_F(AccumulateTypeTest, NumElementsToDim) {
  std::vector<int64_t> dims = {2, 3, 4, 5};
  // to_dim(0) = 1 (空积)
  // to_dim(1) = 2
  // to_dim(2) = 2*3 = 6
  // to_dim(3) = 2*3*4 = 24
  // to_dim(4) = 2*3*4*5 = 120

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "NumElementsToDim ";
  file << std::to_string(c10::numelements_to_dim(0, dims)) << " ";
  file << std::to_string(c10::numelements_to_dim(1, dims)) << " ";
  file << std::to_string(c10::numelements_to_dim(2, dims)) << " ";
  file << std::to_string(c10::numelements_to_dim(3, dims)) << " ";
  file << std::to_string(c10::numelements_to_dim(4, dims)) << " ";
  file << "\n";
  file.saveFile();
}

// numelements_between_dim
TEST_F(AccumulateTypeTest, NumElementsBetweenDim) {
  std::vector<int64_t> dims = {2, 3, 4, 5};
  // between(0, 2) = dims[0]*dims[1] = 2*3 = 6
  // between(1, 3) = dims[1]*dims[2] = 3*4 = 12

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "NumElementsBetweenDim ";
  file << std::to_string(c10::numelements_between_dim(0, 2, dims)) << " ";
  file << std::to_string(c10::numelements_between_dim(1, 3, dims)) << " ";
  // 交换顺序应得同样结果
  file << std::to_string(c10::numelements_between_dim(3, 1, dims)) << " ";
  file << "\n";
  file.saveFile();
}

// 大值测试
TEST_F(AccumulateTypeTest, LargeValues) {
  std::vector<int64_t> vec = {100, 100, 100};
  int64_t sum_result = c10::sum_integers(vec);
  int64_t mul_result = c10::multiply_integers(vec);

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "LargeValues ";
  file << std::to_string(sum_result) << " ";
  file << std::to_string(mul_result) << " ";
  file << "\n";
  file.saveFile();
}

// toAccumulateType(c10::ScalarType type, c10::DeviceType device) - CPU
TEST_F(AccumulateTypeTest, ToAccumulateTypeCPUDevice) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "ToAccumulateTypeCPUDevice ";

  // CPU accumulate types
  file << "BFloat16->"
       << std::to_string(static_cast<int>(at::toAccumulateType(
              c10::ScalarType::BFloat16, c10::DeviceType::CPU)))
       << " ";
  file << "Half->"
       << std::to_string(static_cast<int>(at::toAccumulateType(
              c10::ScalarType::Half, c10::DeviceType::CPU)))
       << " ";
  file << "Float->"
       << std::to_string(static_cast<int>(at::toAccumulateType(
              c10::ScalarType::Float, c10::DeviceType::CPU)))
       << " ";
  file << "Double->"
       << std::to_string(static_cast<int>(at::toAccumulateType(
              c10::ScalarType::Double, c10::DeviceType::CPU)))
       << " ";
  file << "Char->"
       << std::to_string(static_cast<int>(at::toAccumulateType(
              c10::ScalarType::Char, c10::DeviceType::CPU)))
       << " ";
  file << "Int16->"
       << std::to_string(static_cast<int>(at::toAccumulateType(
              c10::ScalarType::Short, c10::DeviceType::CPU)))
       << " ";
  file << "Int32->"
       << std::to_string(static_cast<int>(
              at::toAccumulateType(c10::ScalarType::Int, c10::DeviceType::CPU)))
       << " ";
  file << "Int64->"
       << std::to_string(static_cast<int>(at::toAccumulateType(
              c10::ScalarType::Long, c10::DeviceType::CPU)))
       << " ";
  file << "Bool->"
       << std::to_string(static_cast<int>(at::toAccumulateType(
              c10::ScalarType::Bool, c10::DeviceType::CPU)))
       << " ";
  file << "\n";
  file.saveFile();
}

// toAccumulateType(c10::ScalarType type, c10::DeviceType device) - CUDA
TEST_F(AccumulateTypeTest, ToAccumulateTypeCUDADevice) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "ToAccumulateTypeCUDADevice ";

  // CUDA accumulate types
  file << "BFloat16->"
       << std::to_string(static_cast<int>(at::toAccumulateType(
              c10::ScalarType::BFloat16, c10::DeviceType::CUDA)))
       << " ";
  file << "Half->"
       << std::to_string(static_cast<int>(at::toAccumulateType(
              c10::ScalarType::Half, c10::DeviceType::CUDA)))
       << " ";
  file << "Float->"
       << std::to_string(static_cast<int>(at::toAccumulateType(
              c10::ScalarType::Float, c10::DeviceType::CUDA)))
       << " ";
  file << "Double->"
       << std::to_string(static_cast<int>(at::toAccumulateType(
              c10::ScalarType::Double, c10::DeviceType::CUDA)))
       << " ";
  file << "Char->"
       << std::to_string(static_cast<int>(at::toAccumulateType(
              c10::ScalarType::Char, c10::DeviceType::CUDA)))
       << " ";
  file << "Int16->"
       << std::to_string(static_cast<int>(at::toAccumulateType(
              c10::ScalarType::Short, c10::DeviceType::CUDA)))
       << " ";
  file << "Int32->"
       << std::to_string(static_cast<int>(at::toAccumulateType(
              c10::ScalarType::Int, c10::DeviceType::CUDA)))
       << " ";
  file << "Int64->"
       << std::to_string(static_cast<int>(at::toAccumulateType(
              c10::ScalarType::Long, c10::DeviceType::CUDA)))
       << " ";
  file << "Bool->"
       << std::to_string(static_cast<int>(at::toAccumulateType(
              c10::ScalarType::Bool, c10::DeviceType::CUDA)))
       << " ";
  file << "\n";
  file.saveFile();
}

// toAccumulateType(c10::ScalarType type, bool is_cuda) - is_cuda = false
TEST_F(AccumulateTypeTest, ToAccumulateTypeBoolFalse) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "ToAccumulateTypeBoolFalse ";

  // is_cuda = false (equivalent to CPU)
  file << "Float->"
       << std::to_string(static_cast<int>(
              at::toAccumulateType(c10::ScalarType::Float, false)))
       << " ";
  file << "Double->"
       << std::to_string(static_cast<int>(
              at::toAccumulateType(c10::ScalarType::Double, false)))
       << " ";
  file << "Int32->"
       << std::to_string(static_cast<int>(
              at::toAccumulateType(c10::ScalarType::Int, false)))
       << " ";
  file << "\n";
  file.saveFile();
}

// toAccumulateType(c10::ScalarType type, bool is_cuda) - is_cuda = true
TEST_F(AccumulateTypeTest, ToAccumulateTypeBoolTrue) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "ToAccumulateTypeBoolTrue ";

  // is_cuda = true (equivalent to CUDA)
  file << "Float->"
       << std::to_string(static_cast<int>(
              at::toAccumulateType(c10::ScalarType::Float, true)))
       << " ";
  file << "Double->"
       << std::to_string(static_cast<int>(
              at::toAccumulateType(c10::ScalarType::Double, true)))
       << " ";
  file << "Int32->"
       << std::to_string(static_cast<int>(
              at::toAccumulateType(c10::ScalarType::Int, true)))
       << " ";
  file << "\n";
  file.saveFile();
}

}  // namespace test
}  // namespace at
