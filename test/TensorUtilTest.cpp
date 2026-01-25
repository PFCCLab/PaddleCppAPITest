#include <ATen/ATen.h>
#include <ATen/core/Tensor.h>
#include <ATen/ops/ones.h>
#include <gtest/gtest.h>
#include <torch/all.h>

#include <string>
#include <vector>

#include "../src/file_manager.h"

extern paddle_api_test::ThreadSafeParam g_custom_param;

namespace at {
namespace test {

using paddle_api_test::FileManerger;
using paddle_api_test::ThreadSafeParam;

class TensorUtilTest : public ::testing::Test {
 protected:
  void SetUp() override {
    std::vector<int64_t> shape = {2, 3, 4};
    tensor = at::ones(shape, at::kFloat);
  }

  at::Tensor tensor;
};

// 测试 toString
TEST_F(TensorUtilTest, ToString) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();
  std::string tensor_str = tensor.toString();
  file << tensor_str << " ";
  file.saveFile();
}

// 测试 is_contiguous_or_false
TEST_F(TensorUtilTest, IsContiguousOrFalse) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();
  file << std::to_string(tensor.is_contiguous_or_false()) << " ";

  // 测试非连续的tensor
  at::Tensor transposed = tensor.transpose(0, 2);
  file << std::to_string(transposed.is_contiguous_or_false()) << " ";
  file.saveFile();
}

// 测试 is_same
TEST_F(TensorUtilTest, IsSame) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();

  // Test that tensor is same as itself
  file << std::to_string(tensor.is_same(tensor)) << " ";

  // Test that two different tensors are not the same
  at::Tensor other_tensor = at::ones({2, 3, 4}, at::kFloat);
  file << std::to_string(tensor.is_same(other_tensor)) << " ";

  // Test that a shallow copy points to the same tensor
  at::Tensor shallow_copy = tensor;
  file << std::to_string(tensor.is_same(shallow_copy)) << " ";

  // Test that a view of the tensor
  at::Tensor view = tensor.view({24});
  file << std::to_string(tensor.is_same(view)) << " ";
  file.saveFile();
}

// 测试 use_count
TEST_F(TensorUtilTest, UseCount) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();

  // Get initial use count
  size_t initial_count = tensor.use_count();
  file << std::to_string(initial_count) << " ";

  // Create a copy, should increase use count
  {
    at::Tensor copy = tensor;
    size_t new_count = tensor.use_count();
    file << std::to_string(new_count) << " ";
    file << std::to_string(new_count - initial_count) << " ";  // 差值
  }

  // After copy goes out of scope, use count should decrease
  size_t final_count = tensor.use_count();
  file << std::to_string(final_count) << " ";
  file.saveFile();
}

// 测试 weak_use_count
TEST_F(TensorUtilTest, WeakUseCount) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();

  // Get initial weak use count
  size_t initial_weak_count = tensor.weak_use_count();
  file << std::to_string(initial_weak_count) << " ";
  file.saveFile();
}

// 测试 print
TEST_F(TensorUtilTest, Print) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();

  // 创建一个小的tensor用于print测试
  at::Tensor small_tensor = at::ones({2, 2}, at::kFloat);

  // 使用 captureStdout 捕获 print() 的输出
  file.captureStdout([&]() {
    tensor.print();
    small_tensor.print();
  });

  file << std::to_string(1) << " ";  // 如果执行到这里说明print()没有崩溃
  file.saveFile();
}

}  // namespace test
}  // namespace at
