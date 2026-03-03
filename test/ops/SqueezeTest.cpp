#include <ATen/ATen.h>
#include <ATen/core/Tensor.h>
#include <ATen/ops/ones.h>
#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "../../src/file_manager.h"

extern paddle_api_test::ThreadSafeParam g_custom_param;

namespace at {
namespace test {

using paddle_api_test::FileManerger;
using paddle_api_test::ThreadSafeParam;

class SqueezeTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // 创建一个包含大小为1的维度的tensor: shape = {2, 1, 3, 1, 4}
    tensor_with_ones = at::ones({2, 1, 3, 1, 4}, at::kFloat);
  }
  at::Tensor tensor_with_ones;
};

// 测试 squeeze - 移除所有大小为1的维度
TEST_F(SqueezeTest, SqueezeAll) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();
  at::Tensor squeezed = tensor_with_ones.squeeze();
  file << std::to_string(squeezed.dim()) << " ";
  file << std::to_string(squeezed.numel()) << " ";
  for (int64_t i = 0; i < squeezed.dim(); ++i) {
    file << std::to_string(squeezed.sizes()[i]) << " ";
  }
  file.saveFile();
}

// 测试 squeeze - 移除指定维度
TEST_F(SqueezeTest, SqueezeDim) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();
  // 移除维度1（大小为1）
  at::Tensor squeezed_dim1 = tensor_with_ones.squeeze(1);
  file << std::to_string(squeezed_dim1.dim()) << " ";
  file << std::to_string(squeezed_dim1.numel()) << " ";
  for (int64_t i = 0; i < squeezed_dim1.dim(); ++i) {
    file << std::to_string(squeezed_dim1.sizes()[i]) << " ";
  }
  file.saveFile();
}

// 测试 squeeze_ - 原位移除所有大小为1的维度
TEST_F(SqueezeTest, SqueezeInplaceAll) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();
  // 记录原始数据指针
  void* original_ptr = tensor_with_ones.data_ptr();
  // 原位移除所有大小为1的维度
  tensor_with_ones.squeeze_();
  file << std::to_string(tensor_with_ones.dim()) << " ";
  file << std::to_string(tensor_with_ones.numel()) << " ";
  for (int64_t i = 0; i < tensor_with_ones.dim(); ++i) {
    file << std::to_string(tensor_with_ones.sizes()[i]) << " ";
  }
  // 验证是原位操作（数据指针未改变）
  file << std::to_string(tensor_with_ones.data_ptr() == original_ptr) << " ";
  file.saveFile();
}

// 测试 squeeze_ - 原位移除指定维度
TEST_F(SqueezeTest, SqueezeInplaceDim) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();
  // 记录原始数据指针
  void* original_ptr = tensor_with_ones.data_ptr();
  // 原位移除维度1
  tensor_with_ones.squeeze_(1);
  file << std::to_string(tensor_with_ones.dim()) << " ";
  file << std::to_string(tensor_with_ones.numel()) << " ";
  for (int64_t i = 0; i < tensor_with_ones.dim(); ++i) {
    file << std::to_string(tensor_with_ones.sizes()[i]) << " ";
  }
  // 验证是原位操作（数据指针未改变）
  file << std::to_string(tensor_with_ones.data_ptr() == original_ptr) << " ";
  file.saveFile();
}

}  // namespace test
}  // namespace at
