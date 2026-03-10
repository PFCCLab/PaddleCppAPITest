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

class UnsqueezeTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // 创建一个基础tensor: shape = {2, 3, 4}
    tensor = at::ones({2, 3, 4}, at::kFloat);
  }
  at::Tensor tensor;
};

// 测试 unsqueeze - 在维度0之前添加维度
TEST_F(UnsqueezeTest, UnsqueezeDim0) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();
  at::Tensor unsqueezed0 = tensor.unsqueeze(0);
  file << std::to_string(unsqueezed0.dim()) << " ";
  file << std::to_string(unsqueezed0.numel()) << " ";
  for (int64_t i = 0; i < unsqueezed0.dim(); ++i) {
    file << std::to_string(unsqueezed0.sizes()[i]) << " ";
  }
  file.saveFile();
}

// 测试 unsqueeze - 在维度2之前添加维度
TEST_F(UnsqueezeTest, UnsqueezeDim2) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();
  at::Tensor unsqueezed2 = tensor.unsqueeze(2);
  file << std::to_string(unsqueezed2.dim()) << " ";
  file << std::to_string(unsqueezed2.numel()) << " ";
  for (int64_t i = 0; i < unsqueezed2.dim(); ++i) {
    file << std::to_string(unsqueezed2.sizes()[i]) << " ";
  }
  file.saveFile();
}

// 测试 unsqueeze - 使用负索引在最后添加维度
TEST_F(UnsqueezeTest, UnsqueezeNegativeDim) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();
  at::Tensor unsqueezed_last = tensor.unsqueeze(-1);
  file << std::to_string(unsqueezed_last.dim()) << " ";
  file << std::to_string(unsqueezed_last.numel()) << " ";
  for (int64_t i = 0; i < unsqueezed_last.dim(); ++i) {
    file << std::to_string(unsqueezed_last.sizes()[i]) << " ";
  }
  file.saveFile();
}

// 测试 unsqueeze_ - 原位在维度0之前添加维度
TEST_F(UnsqueezeTest, UnsqueezeInplaceDim0) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();
  // 记录原始数据指针
  void* original_ptr = tensor.data_ptr();
  // 原位在维度0之前添加维度
  tensor.unsqueeze_(0);
  file << std::to_string(tensor.dim()) << " ";
  file << std::to_string(tensor.numel()) << " ";
  for (int64_t i = 0; i < tensor.dim(); ++i) {
    file << std::to_string(tensor.sizes()[i]) << " ";
  }
  // 验证是原位操作（数据指针未改变）
  file << std::to_string(tensor.data_ptr() == original_ptr) << " ";
  file.saveFile();
}

// 测试 unsqueeze_ - 原位使用负索引添加维度
TEST_F(UnsqueezeTest, UnsqueezeInplaceNegativeDim) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();
  // 记录原始数据指针
  void* original_ptr = tensor.data_ptr();
  // 原位在最后添加维度
  tensor.unsqueeze_(-1);
  file << std::to_string(tensor.dim()) << " ";
  file << std::to_string(tensor.numel()) << " ";
  for (int64_t i = 0; i < tensor.dim(); ++i) {
    file << std::to_string(tensor.sizes()[i]) << " ";
  }
  // 验证是原位操作（数据指针未改变）
  file << std::to_string(tensor.data_ptr() == original_ptr) << " ";
  file.saveFile();
}

}  // namespace test
}  // namespace at
