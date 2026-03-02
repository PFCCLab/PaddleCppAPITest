#include <ATen/ATen.h>
#include <ATen/core/Tensor.h>
#include <ATen/ops/ones.h>
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

static void write_flatten_result_to_file(FileManerger* file,
                                         const at::Tensor& result) {
  *file << std::to_string(result.dim()) << " ";
  *file << std::to_string(result.numel()) << " ";
  for (int64_t i = 0; i < result.dim(); ++i) {
    *file << std::to_string(result.sizes()[i]) << " ";
  }
}

class FlattenTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // 创建一个 2x3x4 的 tensor
    tensor = at::ones({2, 3, 4}, at::kFloat);
    float* data = tensor.data_ptr<float>();
    for (int64_t i = 0; i < 24; ++i) {
      data[i] = static_cast<float>(i);
    }
  }

  at::Tensor tensor;
};

// ========== 基础功能 ==========

// 测试 flatten 默认参数 (start_dim=0, end_dim=-1)
TEST_F(FlattenTest, FlattenDefault) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();
  file << "FlattenDefault ";
  at::Tensor result = tensor.flatten(0, -1);
  write_flatten_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// 测试 flatten 指定 start_dim 和 end_dim
TEST_F(FlattenTest, FlattenWithDims) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "FlattenWithDims ";
  // flatten dim 1 and 2: shape {2, 3, 4} -> {2, 12}
  at::Tensor result = tensor.flatten(1, 2);
  write_flatten_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// 测试 flatten 使用负数索引
TEST_F(FlattenTest, FlattenNegativeDims) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "FlattenNegativeDims ";
  // flatten from dim -2 to -1: shape {2, 3, 4} -> {2, 12}
  at::Tensor result = tensor.flatten(-2, -1);
  write_flatten_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// 测试 flatten 从 dim 0 开始
TEST_F(FlattenTest, FlattenFromStart) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "FlattenFromStart ";
  // flatten dim 0 and 1: shape {2, 3, 4} -> {6, 4}
  at::Tensor result = tensor.flatten(0, 1);
  write_flatten_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// ========== Shape 覆盖 ==========

// 标量 flatten
TEST_F(FlattenTest, ScalarFlatten) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "ScalarFlatten ";
  at::Tensor scalar = at::ones({}, at::kFloat);
  at::Tensor result = scalar.flatten(0, -1);
  write_flatten_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// 大 shape flatten
TEST_F(FlattenTest, LargeShapeFlatten) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "LargeShapeFlatten ";
  at::Tensor large = at::ones({10, 20, 30, 40}, at::kFloat);
  at::Tensor result = large.flatten(0, -1);
  file << std::to_string(result.numel()) << " ";
  file << std::to_string(result.dim()) << " ";
  file << "\n";
  file.saveFile();
}

// 零维度 tensor flatten
TEST_F(FlattenTest, ZeroDimFlatten) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "ZeroDimFlatten ";
  at::Tensor zero_tensor = at::ones({2, 0, 3}, at::kFloat);
  at::Tensor result = zero_tensor.flatten(0, -1);
  write_flatten_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// 全一维度 flatten
TEST_F(FlattenTest, AllOneShapeFlatten) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "AllOneShapeFlatten ";
  at::Tensor t = at::ones({1, 1, 1}, at::kFloat);
  at::Tensor result = t.flatten(0, -1);
  write_flatten_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// 非连续 tensor flatten
TEST_F(FlattenTest, NonContiguousFlatten) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "NonContiguousFlatten ";
  at::Tensor t = at::ones({2, 3, 4}, at::kFloat);
  at::Tensor transposed = t.transpose(0, 1);
  file << std::to_string(transposed.is_contiguous()) << " ";
  at::Tensor result = transposed.flatten(0, -1);
  write_flatten_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// ========== Dtype 覆盖 ==========

// float64
TEST_F(FlattenTest, Float64Dtype) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "Float64Dtype ";
  at::Tensor t = at::ones({2, 3, 4}, at::kDouble);
  at::Tensor result = t.flatten(0, -1);
  file << std::to_string(static_cast<int>(result.scalar_type())) << " ";
  write_flatten_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// int32
TEST_F(FlattenTest, Int32Dtype) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "Int32Dtype ";
  at::Tensor t = at::ones({2, 3, 4}, at::kInt);
  at::Tensor result = t.flatten(0, -1);
  file << std::to_string(static_cast<int>(result.scalar_type())) << " ";
  write_flatten_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// int64
TEST_F(FlattenTest, Int64Dtype) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "Int64Dtype ";
  at::Tensor t = at::ones({2, 3, 4}, at::kLong);
  at::Tensor result = t.flatten(0, -1);
  file << std::to_string(static_cast<int>(result.scalar_type())) << " ";
  write_flatten_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// ========== unflatten 测试 ==========

// 测试 unflatten
TEST_F(FlattenTest, Unflatten) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "Unflatten ";
  // 先 flatten 成 {2, 12}，然后 unflatten 回 {2, 3, 4}
  at::Tensor flattened = tensor.flatten(1, 2);
  at::Tensor result = flattened.unflatten(1, {3, 4});
  write_flatten_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// 测试 unflatten 使用负数维度
TEST_F(FlattenTest, UnflattenNegativeDim) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "UnflattenNegativeDim ";
  // 创建一个 {6, 4} 的 tensor
  at::Tensor flat_tensor = at::ones({6, 4}, at::kFloat);
  // unflatten dim -2 (即 dim 0) 成 {2, 3}
  at::Tensor result = flat_tensor.unflatten(-2, {2, 3});
  write_flatten_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// 测试 unflatten_symint
TEST_F(FlattenTest, UnflattenSymint) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "UnflattenSymint ";
  at::Tensor flattened = tensor.flatten(1, 2);
  c10::SymIntArrayRef sizes({3, 4});
  at::Tensor result = flattened.unflatten_symint(1, sizes);
  write_flatten_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// ========== 数据完整性测试 ==========

// 测试 flatten 后数据保持正确
TEST_F(FlattenTest, FlattenDataIntegrity) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "FlattenDataIntegrity ";
  at::Tensor result = tensor.flatten(0, -1);
  float* src_data = tensor.data_ptr<float>();
  float* dst_data = result.data_ptr<float>();

  // 检查数据是否一致
  bool data_equal = true;
  for (int64_t i = 0; i < 24; ++i) {
    if (src_data[i] != dst_data[i]) {
      data_equal = false;
      break;
    }
  }
  file << std::to_string(data_equal) << " ";
  file << "\n";
  file.saveFile();
}

// ========== 异常行为 ==========

// 无效 dim (start_dim > end_dim)
TEST_F(FlattenTest, InvalidDims) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "InvalidDims ";
  try {
    // start_dim > end_dim
    at::Tensor result = tensor.flatten(2, 1);
    write_flatten_result_to_file(&file, result);
  } catch (const std::exception&) {
    file << "exception ";
  }
  file << "\n";
  file.saveFile();
}

}  // namespace test
}  // namespace at
