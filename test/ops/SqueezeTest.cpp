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

static void write_squeeze_result_to_file(FileManerger* file,
                                         const at::Tensor& result) {
  *file << std::to_string(result.dim()) << " ";
  *file << std::to_string(result.numel()) << " ";
  for (int64_t i = 0; i < result.dim(); ++i) {
    *file << std::to_string(result.sizes()[i]) << " ";
  }
}

class SqueezeTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // 创建一个包含大小为1的维度的tensor: shape = {2, 1, 3, 1, 4}
    tensor_with_ones = at::ones({2, 1, 3, 1, 4}, at::kFloat);
  }
  at::Tensor tensor_with_ones;
};

// ========== 基础功能 ==========

// 测试 squeeze - 移除所有大小为1的维度
TEST_F(SqueezeTest, SqueezeAll) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();
  file << "SqueezeAll ";
  at::Tensor squeezed = tensor_with_ones.squeeze();
  write_squeeze_result_to_file(&file, squeezed);
  file << "\n";
  file.saveFile();
}

// 测试 squeeze - 移除指定维度
TEST_F(SqueezeTest, SqueezeDim) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "SqueezeDim ";
  // 移除维度1（大小为1）
  at::Tensor squeezed_dim1 = tensor_with_ones.squeeze(1);
  write_squeeze_result_to_file(&file, squeezed_dim1);
  file << "\n";
  file.saveFile();
}

// ========== Shape 覆盖 ==========

// 标量 squeeze
TEST_F(SqueezeTest, ScalarSqueeze) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "ScalarSqueeze ";
  at::Tensor scalar = at::ones({}, at::kFloat);
  at::Tensor result = scalar.squeeze();
  write_squeeze_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// 大 shape squeeze
TEST_F(SqueezeTest, LargeShapeSqueeze) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "LargeShapeSqueeze ";
  at::Tensor large = at::ones({100, 1, 100}, at::kFloat);
  at::Tensor result = large.squeeze();
  write_squeeze_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// 零维度 tensor squeeze
TEST_F(SqueezeTest, ZeroDimSqueeze) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "ZeroDimSqueeze ";
  at::Tensor zero_tensor = at::ones({2, 0, 1, 3}, at::kFloat);
  at::Tensor result = zero_tensor.squeeze();
  write_squeeze_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// 全一维度 squeeze
TEST_F(SqueezeTest, AllOneShapeSqueeze) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "AllOneShapeSqueeze ";
  at::Tensor t = at::ones({1, 1, 1}, at::kFloat);
  at::Tensor result = t.squeeze();
  write_squeeze_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// 无大小为1的维度 squeeze (应无变化)
TEST_F(SqueezeTest, NoSizeOneDim) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "NoSizeOneDim ";
  at::Tensor t = at::ones({2, 3, 4}, at::kFloat);
  at::Tensor result = t.squeeze();
  write_squeeze_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// ========== Dtype 覆盖 ==========

// float64
TEST_F(SqueezeTest, Float64Dtype) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "Float64Dtype ";
  at::Tensor t = at::ones({2, 1, 3}, at::kDouble);
  at::Tensor result = t.squeeze();
  file << std::to_string(static_cast<int>(result.scalar_type())) << " ";
  write_squeeze_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// int32
TEST_F(SqueezeTest, Int32Dtype) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "Int32Dtype ";
  at::Tensor t = at::ones({2, 1, 3}, at::kInt);
  at::Tensor result = t.squeeze();
  file << std::to_string(static_cast<int>(result.scalar_type())) << " ";
  write_squeeze_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// int64
TEST_F(SqueezeTest, Int64Dtype) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "Int64Dtype ";
  at::Tensor t = at::ones({2, 1, 3}, at::kLong);
  at::Tensor result = t.squeeze();
  file << std::to_string(static_cast<int>(result.scalar_type())) << " ";
  write_squeeze_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// ========== API 变体 ==========

// 负索引 dim
TEST_F(SqueezeTest, NegativeDim) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "NegativeDim ";
  // 使用负索引指定维度
  at::Tensor result = tensor_with_ones.squeeze(-2);  // 倒数第二个维度
  write_squeeze_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// squeeze 非大小为1的维度 (应无变化)
TEST_F(SqueezeTest, SqueezeNonSizeOneDim) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "SqueezeNonSizeOneDim ";
  // 尝试 squeeze 维度0（大小为2，不是1）
  at::Tensor result = tensor_with_ones.squeeze(0);
  write_squeeze_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// 原位操作 squeeze_ - 移除所有大小为1的维度
TEST_F(SqueezeTest, SqueezeInplaceAll) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "SqueezeInplaceAll ";
  at::Tensor t = tensor_with_ones.clone();
  void* original_ptr = t.data_ptr();
  t.squeeze_();
  file << std::to_string(t.data_ptr() == original_ptr) << " ";
  write_squeeze_result_to_file(&file, t);
  file << "\n";
  file.saveFile();
}

// 原位操作 squeeze_ - 移除指定维度
TEST_F(SqueezeTest, SqueezeInplaceDim) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "SqueezeInplaceDim ";
  at::Tensor t = tensor_with_ones.clone();
  void* original_ptr = t.data_ptr();
  t.squeeze_(1);
  file << std::to_string(t.data_ptr() == original_ptr) << " ";
  write_squeeze_result_to_file(&file, t);
  file << "\n";
  file.saveFile();
}

// 原位操作 squeeze_ - 负索引
TEST_F(SqueezeTest, SqueezeInplaceNegativeDim) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "SqueezeInplaceNegativeDim ";
  at::Tensor t = tensor_with_ones.clone();
  void* original_ptr = t.data_ptr();
  t.squeeze_(-2);
  file << std::to_string(t.data_ptr() == original_ptr) << " ";
  write_squeeze_result_to_file(&file, t);
  file << "\n";
  file.saveFile();
}

// ========== 异常行为 ==========

// 无效 dim (超出范围)
TEST_F(SqueezeTest, InvalidDim) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "InvalidDim ";
  try {
    // tensor_with_ones 是 5D，dim=10 越界
    at::Tensor result = tensor_with_ones.squeeze(10);
    write_squeeze_result_to_file(&file, result);
  } catch (const std::exception&) {
    file << "exception ";
  }
  file << "\n";
  file.saveFile();
}

}  // namespace test
}  // namespace at
