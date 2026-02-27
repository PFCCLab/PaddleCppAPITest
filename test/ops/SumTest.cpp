#include <ATen/ATen.h>
#include <ATen/core/Tensor.h>
#include <ATen/ops/sum.h>
#include <ATen/ops/zeros.h>
#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "../../src/file_manager.h"

extern paddle_api_test::ThreadSafeParam g_custom_param;

namespace at {
namespace test {

using paddle_api_test::FileManerger;
using paddle_api_test::ThreadSafeParam;

class SumTest : public ::testing::Test {
 protected:
  void SetUp() override {
    std::vector<int64_t> shape = {2, 3};
    test_tensor = at::zeros(shape, at::kFloat);
    float* data = test_tensor.data_ptr<float>();
    for (int64_t i = 0; i < 6; ++i) {
      data[i] = static_cast<float>(i + 1);
    }
  }
  at::Tensor test_tensor;
};

TEST_F(SumTest, SumAllElements) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();
  at::Tensor result = at::sum(test_tensor);
  file << std::to_string(result.dim()) << " ";
  file << std::to_string(result.numel()) << " ";
  float result_value = *result.data_ptr<float>();
  file << std::to_string(result_value) << " ";
  file.saveFile();
}

TEST_F(SumTest, SumWithDtype) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();
  at::Tensor result = at::sum(test_tensor, at::kDouble);
  file << std::to_string(result.dim()) << " ";
  file << std::to_string(static_cast<int>(result.scalar_type())) << " ";
  double result_value = *result.data_ptr<double>();
  file << std::to_string(result_value) << " ";
  file.saveFile();
}

TEST_F(SumTest, SumAlongDim0) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();
  at::Tensor result = at::sum(test_tensor, {0}, false);
  file << std::to_string(result.dim()) << " ";
  file << std::to_string(result.numel()) << " ";
  float* data = result.data_ptr<float>();
  file << std::to_string(data[0]) << " ";
  file << std::to_string(data[1]) << " ";
  file << std::to_string(data[2]) << " ";
  file.saveFile();
}

TEST_F(SumTest, SumAlongDim1) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();
  at::Tensor result = at::sum(test_tensor, {1}, false);
  file << std::to_string(result.dim()) << " ";
  file << std::to_string(result.numel()) << " ";
  float* data = result.data_ptr<float>();
  file << std::to_string(data[0]) << " ";
  file << std::to_string(data[1]) << " ";
  file.saveFile();
}

TEST_F(SumTest, SumWithKeepdim) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();
  at::Tensor result = at::sum(test_tensor, {0}, true);
  file << std::to_string(result.dim()) << " ";
  file << std::to_string(result.numel()) << " ";
  file << std::to_string(result.sizes()[0]) << " ";
  file << std::to_string(result.sizes()[1]) << " ";
  float* data = result.data_ptr<float>();
  file << std::to_string(data[0]) << " ";
  file << std::to_string(data[1]) << " ";
  file << std::to_string(data[2]) << " ";
  file.saveFile();
}

TEST_F(SumTest, SumOutFunction) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();
  at::Tensor output = at::zeros({}, at::kFloat);
  at::Tensor& result = at::sum_out(output, test_tensor);
  file << std::to_string(&result == &output) << " ";
  float result_value = *output.data_ptr<float>();
  file << std::to_string(result_value) << " ";
  file.saveFile();
}

// --------------------------------------------------------------------------
// 以下为 tensor 成员函数形式 sum() 的测试
// --------------------------------------------------------------------------

// 测试 tensor.sum()：无参数，对所有元素求和
TEST_F(SumTest, MemberSumAllElements) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();
  at::Tensor result = test_tensor.sum();
  file << std::to_string(result.dim()) << " ";
  file << std::to_string(result.numel()) << " ";
  float result_value = *result.data_ptr<float>();
  file << std::to_string(result_value) << " ";
  file.saveFile();
}

// 测试 tensor.sum(dtype)：指定输出类型
TEST_F(SumTest, MemberSumWithDtype) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();
  at::Tensor result = test_tensor.sum(at::kDouble);
  file << std::to_string(static_cast<int>(result.scalar_type())) << " ";
  double result_value = *result.data_ptr<double>();
  file << std::to_string(result_value) << " ";
  file.saveFile();
}

// 测试 tensor.sum(dim, keepdim=false)：沿 dim=0 求和并降维
TEST_F(SumTest, MemberSumAlongDim0) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();
  at::Tensor result = test_tensor.sum({0}, /*keepdim=*/false);
  file << std::to_string(result.dim()) << " ";
  file << std::to_string(result.numel()) << " ";
  float* data = result.data_ptr<float>();
  file << std::to_string(data[0]) << " ";
  file << std::to_string(data[1]) << " ";
  file << std::to_string(data[2]) << " ";
  file.saveFile();
}

// 测试 tensor.sum(dim, keepdim=true)：保留维度
TEST_F(SumTest, MemberSumKeepdim) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();
  at::Tensor result = test_tensor.sum({1}, /*keepdim=*/true);
  file << std::to_string(result.dim()) << " ";
  file << std::to_string(result.sizes()[0]) << " ";
  file << std::to_string(result.sizes()[1]) << " ";
  float* data = result.data_ptr<float>();
  file << std::to_string(data[0]) << " ";  // 1+2+3=6
  file << std::to_string(data[1]) << " ";  // 4+5+6=15
  file.saveFile();
}

// 测试 tensor.sum(dim, keepdim, dtype)：沿多维度求和并指定输出类型
TEST_F(SumTest, MemberSumMultiDimWithDtype) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();
  at::Tensor result = test_tensor.sum(at::IntArrayRef{0, 1},
                                      /*keepdim=*/false,
                                      std::make_optional(at::kDouble));
  file << std::to_string(result.dim()) << " ";
  file << std::to_string(result.numel()) << " ";
  file << std::to_string(static_cast<int>(result.scalar_type())) << " ";
  double val = *result.data_ptr<double>();
  file << std::to_string(val) << " ";  // 21
  file.saveFile();
}

}  // namespace test
}  // namespace at
