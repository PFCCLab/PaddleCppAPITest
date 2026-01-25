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

class TensorAccessorTest : public ::testing::Test {
 protected:
  void SetUp() override {
    std::vector<int64_t> shape = {2, 3, 4};
    tensor = at::ones(shape, at::kFloat);
  }

  at::Tensor tensor;
};

// 测试 packed_accessor32
TEST_F(TensorAccessorTest, PackedAccessor32) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();
  auto accessor = tensor.packed_accessor32<float, 3>();
  file << std::to_string(accessor.size(0)) << " ";
  file << std::to_string(accessor.size(1)) << " ";
  file << std::to_string(accessor.size(2)) << " ";
  file << std::to_string(accessor[0][0][0]) << " ";
  file << std::to_string(accessor[1][2][3]) << " ";
  file.saveFile();
}

// 测试 packed_accessor64
TEST_F(TensorAccessorTest, PackedAccessor64) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();
  auto accessor = tensor.packed_accessor64<float, 3>();
  file << std::to_string(accessor.size(0)) << " ";
  file << std::to_string(accessor.size(1)) << " ";
  file << std::to_string(accessor.size(2)) << " ";
  file << std::to_string(accessor[0][0][0]) << " ";
  file << std::to_string(accessor[1][2][3]) << " ";
  file.saveFile();
}

// 测试 generic_packed_accessor
TEST_F(TensorAccessorTest, GenericPackedAccessor) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();
  auto accessor = tensor.generic_packed_accessor<float, 3>();
  file << std::to_string(accessor.size(0)) << " ";
  file << std::to_string(accessor.size(1)) << " ";
  file << std::to_string(accessor.size(2)) << " ";
  file << std::to_string(accessor[0][0][0]) << " ";
  file << std::to_string(accessor[1][2][3]) << " ";
  file.saveFile();
}

// 测试 is_non_overlapping_and_dense
TEST_F(TensorAccessorTest, IsNonOverlappingAndDense) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();
  file << std::to_string(tensor.is_non_overlapping_and_dense()) << " ";

  // 测试非连续的tensor
  at::Tensor transposed = tensor.transpose(0, 2);
  file << std::to_string(transposed.is_non_overlapping_and_dense()) << " ";

  // 测试连续化后的tensor
  at::Tensor contiguous = transposed.contiguous();
  file << std::to_string(contiguous.is_non_overlapping_and_dense()) << " ";
  file.saveFile();
}

// 测试 has_names
TEST_F(TensorAccessorTest, HasNames) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();
  file << std::to_string(tensor.has_names()) << " ";
  file.saveFile();
}

// 测试 print
TEST_F(TensorAccessorTest, Print) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();

  // 创建一个小的tensor用于print测试
  at::Tensor small_tensor = at::ones({2, 2}, at::kFloat);

  // print() 通常输出到标准输出，我们主要测试它不会崩溃
  // 并记录tensor的基本信息
  tensor.print();
  small_tensor.print();

  file << std::to_string(1) << " ";  // 如果执行到这里说明print()没有崩溃
  file.saveFile();
}

}  // namespace test
}  // namespace at
