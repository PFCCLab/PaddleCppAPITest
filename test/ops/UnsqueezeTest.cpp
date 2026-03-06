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

static void write_unsqueeze_result_to_file(FileManerger* file,
                                           const at::Tensor& result) {
  *file << std::to_string(result.dim()) << " ";
  *file << std::to_string(result.numel()) << " ";
  for (int64_t i = 0; i < result.dim(); ++i) {
    *file << std::to_string(result.sizes()[i]) << " ";
  }
}

class UnsqueezeTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // 创建一个基础tensor: shape = {2, 3, 4}
    tensor = at::ones({2, 3, 4}, at::kFloat);
  }
  at::Tensor tensor;
};

// ========== 基础功能 ==========

// 测试 unsqueeze - 在维度0之前添加维度
TEST_F(UnsqueezeTest, UnsqueezeDim0) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();
  file << "UnsqueezeDim0 ";
  at::Tensor unsqueezed0 = tensor.unsqueeze(0);
  write_unsqueeze_result_to_file(&file, unsqueezed0);
  file << "\n";
  file.saveFile();
}

// 测试 unsqueeze - 在维度2之前添加维度
TEST_F(UnsqueezeTest, UnsqueezeDim2) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "UnsqueezeDim2 ";
  at::Tensor unsqueezed2 = tensor.unsqueeze(2);
  write_unsqueeze_result_to_file(&file, unsqueezed2);
  file << "\n";
  file.saveFile();
}

// 测试 unsqueeze - 使用负索引在最后添加维度
TEST_F(UnsqueezeTest, UnsqueezeNegativeDim) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "UnsqueezeNegativeDim ";
  at::Tensor unsqueezed_last = tensor.unsqueeze(-1);
  write_unsqueeze_result_to_file(&file, unsqueezed_last);
  file << "\n";
  file.saveFile();
}

// ========== Shape 覆盖 ==========

// 标量 unsqueeze
TEST_F(UnsqueezeTest, ScalarUnsqueeze) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "ScalarUnsqueeze ";
  at::Tensor scalar = at::ones({}, at::kFloat);
  at::Tensor result = scalar.unsqueeze(0);
  write_unsqueeze_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// 大 shape unsqueeze
TEST_F(UnsqueezeTest, LargeShapeUnsqueeze) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "LargeShapeUnsqueeze ";
  at::Tensor large = at::ones({100, 100}, at::kFloat);
  at::Tensor result = large.unsqueeze(0);
  write_unsqueeze_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// 零维度 tensor unsqueeze
TEST_F(UnsqueezeTest, ZeroDimUnsqueeze) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "ZeroDimUnsqueeze ";
  at::Tensor zero_tensor = at::ones({2, 0, 3}, at::kFloat);
  at::Tensor result = zero_tensor.unsqueeze(1);
  write_unsqueeze_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// 全一维度 unsqueeze
TEST_F(UnsqueezeTest, AllOneShapeUnsqueeze) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "AllOneShapeUnsqueeze ";
  at::Tensor t = at::ones({1, 1, 1}, at::kFloat);
  at::Tensor result = t.unsqueeze(0);
  write_unsqueeze_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// ========== Dtype 覆盖 ==========

// float64
TEST_F(UnsqueezeTest, Float64Dtype) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "Float64Dtype ";
  at::Tensor t = at::ones({2, 3}, at::kDouble);
  at::Tensor result = t.unsqueeze(0);
  file << std::to_string(static_cast<int>(result.scalar_type())) << " ";
  write_unsqueeze_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// int32
TEST_F(UnsqueezeTest, Int32Dtype) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "Int32Dtype ";
  at::Tensor t = at::ones({2, 3}, at::kInt);
  at::Tensor result = t.unsqueeze(0);
  file << std::to_string(static_cast<int>(result.scalar_type())) << " ";
  write_unsqueeze_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// int64
TEST_F(UnsqueezeTest, Int64Dtype) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "Int64Dtype ";
  at::Tensor t = at::ones({2, 3}, at::kLong);
  at::Tensor result = t.unsqueeze(0);
  file << std::to_string(static_cast<int>(result.scalar_type())) << " ";
  write_unsqueeze_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// ========== API 变体 ==========

// 多次 unsqueeze
TEST_F(UnsqueezeTest, MultipleUnsqueeze) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "MultipleUnsqueeze ";
  at::Tensor result = tensor.unsqueeze(0).unsqueeze(-1);
  write_unsqueeze_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// 原位操作 unsqueeze_ - 在维度0之前添加维度
TEST_F(UnsqueezeTest, UnsqueezeInplaceDim0) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "UnsqueezeInplaceDim0 ";
  at::Tensor t = tensor.clone();
  void* original_ptr = t.data_ptr();
  t.unsqueeze_(0);
  file << std::to_string(t.data_ptr() == original_ptr) << " ";
  write_unsqueeze_result_to_file(&file, t);
  file << "\n";
  file.saveFile();
}

// 原位操作 unsqueeze_ - 使用负索引添加维度
TEST_F(UnsqueezeTest, UnsqueezeInplaceNegativeDim) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "UnsqueezeInplaceNegativeDim ";
  at::Tensor t = tensor.clone();
  void* original_ptr = t.data_ptr();
  t.unsqueeze_(-1);
  file << std::to_string(t.data_ptr() == original_ptr) << " ";
  write_unsqueeze_result_to_file(&file, t);
  file << "\n";
  file.saveFile();
}

// ========== 异常行为 ==========

// 无效 dim (超出范围)
TEST_F(UnsqueezeTest, InvalidDim) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "InvalidDim ";
  try {
    // tensor 是 3D，unsqueeze 的有效 dim 范围是 [-4, 3]
    at::Tensor result = tensor.unsqueeze(10);
    write_unsqueeze_result_to_file(&file, result);
  } catch (const std::exception&) {
    file << "exception ";
  }
  file << "\n";
  file.saveFile();
}

}  // namespace test
}  // namespace at
