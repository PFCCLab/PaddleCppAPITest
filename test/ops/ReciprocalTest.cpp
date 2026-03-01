#include <ATen/ATen.h>
#include <ATen/core/Tensor.h>
#include <ATen/ops/reciprocal.h>
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

class ReciprocalTest : public ::testing::Test {
 protected:
  void SetUp() override {
    std::vector<int64_t> shape = {4};
    test_tensor = at::zeros(shape, at::kFloat);
    float* data = test_tensor.data_ptr<float>();
    data[0] = 1.0f;
    data[1] = 2.0f;
    data[2] = 0.5f;
    data[3] = 4.0f;
  }
  at::Tensor test_tensor;
};

static void write_reciprocal_result_to_file(FileManerger* file,
                                            const at::Tensor& result) {
  *file << std::to_string(result.dim()) << " ";
  *file << std::to_string(result.numel()) << " ";
  float* result_data = result.data_ptr<float>();
  for (int64_t i = 0; i < result.numel(); ++i) {
    *file << std::to_string(result_data[i]) << " ";
  }
}

// 测试 reciprocal() 方法
TEST_F(ReciprocalTest, BasicReciprocal) {
  at::Tensor result = test_tensor.reciprocal();
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();
  write_reciprocal_result_to_file(&file, result);

  // 验证原始 tensor 未被修改
  float* original_data = test_tensor.data_ptr<float>();
  file << std::to_string(original_data[0]) << " ";
  file << std::to_string(original_data[1]) << " ";
  file.saveFile();
}

// 测试 reciprocal_() in-place 方法
TEST_F(ReciprocalTest, InplaceReciprocal) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();

  // 保存原始数据指针
  float* original_ptr = test_tensor.data_ptr<float>();

  // 调用 in-place 版本
  at::Tensor& result = test_tensor.reciprocal_();

  // 验证返回的是同一个 tensor
  file << std::to_string(result.data_ptr<float>() == original_ptr) << " ";

  write_reciprocal_result_to_file(&file, result);
  file.saveFile();
}

// 测试不同值的 reciprocal
TEST_F(ReciprocalTest, VariousValues) {
  at::Tensor various_tensor = at::zeros({5}, at::kFloat);
  float* data = various_tensor.data_ptr<float>();
  data[0] = 10.0f;
  data[1] = 0.1f;
  data[2] = -2.0f;
  data[3] = -0.5f;
  data[4] = 100.0f;

  at::Tensor result = various_tensor.reciprocal();
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  write_reciprocal_result_to_file(&file, result);
  file.saveFile();
}

// 测试多维 tensor 的 reciprocal
TEST_F(ReciprocalTest, MultiDimensionalTensor) {
  at::Tensor multi_dim_tensor = at::zeros({2, 3}, at::kFloat);
  float* data = multi_dim_tensor.data_ptr<float>();
  data[0] = 1.0f;
  data[1] = 2.0f;
  data[2] = 4.0f;
  data[3] = 0.25f;
  data[4] = 0.5f;
  data[5] = 8.0f;

  at::Tensor result = multi_dim_tensor.reciprocal();
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();

  file << std::to_string(result.dim()) << " ";
  file << std::to_string(result.sizes()[0]) << " ";
  file << std::to_string(result.sizes()[1]) << " ";
  write_reciprocal_result_to_file(&file, result);
  file.saveFile();
}

// 测试使用 at::reciprocal 全局函数
TEST_F(ReciprocalTest, GlobalReciprocal) {
  at::Tensor result = at::reciprocal(test_tensor);
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  write_reciprocal_result_to_file(&file, result);
  file.saveFile();
}

}  // namespace test
}  // namespace at
