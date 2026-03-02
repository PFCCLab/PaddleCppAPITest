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

static void write_narrow_result_to_file(FileManerger* file,
                                        const at::Tensor& result) {
  *file << std::to_string(result.dim()) << " ";
  *file << std::to_string(result.numel()) << " ";
  for (int64_t i = 0; i < result.dim(); ++i) {
    *file << std::to_string(result.sizes()[i]) << " ";
  }
}

class NarrowTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // 创建一个 4x5x6 的 tensor
    tensor = at::zeros({4, 5, 6}, at::kFloat);
    float* data = tensor.data_ptr<float>();
    for (int64_t i = 0; i < 120; ++i) {
      data[i] = static_cast<float>(i);
    }
  }

  at::Tensor tensor;
};

// ========== 基础功能 ==========

// 测试 narrow 在 dim 0
TEST_F(NarrowTest, NarrowDim0) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();
  file << "NarrowDim0 ";
  // narrow(dim=0, start=1, length=2): shape {4, 5, 6} -> {2, 5, 6}
  at::Tensor result = tensor.narrow(0, 1, 2);
  write_narrow_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// 测试 narrow 在 dim 1
TEST_F(NarrowTest, NarrowDim1) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "NarrowDim1 ";
  // narrow(dim=1, start=2, length=3): shape {4, 5, 6} -> {4, 3, 6}
  at::Tensor result = tensor.narrow(1, 2, 3);
  write_narrow_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// 测试 narrow 在 dim 2
TEST_F(NarrowTest, NarrowDim2) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "NarrowDim2 ";
  // narrow(dim=2, start=0, length=4): shape {4, 5, 6} -> {4, 5, 4}
  at::Tensor result = tensor.narrow(2, 0, 4);
  write_narrow_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// ========== Shape 覆盖 ==========

// 标量 narrow
TEST_F(NarrowTest, ScalarNarrow) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "ScalarNarrow ";
  at::Tensor scalar = at::ones({}, at::kFloat);
  try {
    // 标量无法 narrow
    at::Tensor result = scalar.narrow(0, 0, 1);
    write_narrow_result_to_file(&file, result);
  } catch (const std::exception&) {
    file << "exception ";
  }
  file << "\n";
  file.saveFile();
}

// 大 shape narrow
TEST_F(NarrowTest, LargeShapeNarrow) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "LargeShapeNarrow ";
  at::Tensor large = at::zeros({100, 100}, at::kFloat);
  at::Tensor result = large.narrow(0, 10, 50);
  write_narrow_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// 零维度 tensor narrow
TEST_F(NarrowTest, ZeroDimNarrow) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "ZeroDimNarrow ";
  at::Tensor zero_tensor = at::zeros({2, 0, 3}, at::kFloat);
  at::Tensor result = zero_tensor.narrow(0, 0, 2);
  write_narrow_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// 全一维度 narrow
TEST_F(NarrowTest, AllOneShapeNarrow) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "AllOneShapeNarrow ";
  at::Tensor t = at::ones({1, 1, 1}, at::kFloat);
  at::Tensor result = t.narrow(0, 0, 1);
  write_narrow_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// ========== Dtype 覆盖 ==========

// float64
TEST_F(NarrowTest, Float64Dtype) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "Float64Dtype ";
  at::Tensor t = at::zeros({4, 5}, at::kDouble);
  at::Tensor result = t.narrow(0, 1, 2);
  file << std::to_string(static_cast<int>(result.scalar_type())) << " ";
  write_narrow_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// int32
TEST_F(NarrowTest, Int32Dtype) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "Int32Dtype ";
  at::Tensor t = at::zeros({4, 5}, at::kInt);
  at::Tensor result = t.narrow(0, 1, 2);
  file << std::to_string(static_cast<int>(result.scalar_type())) << " ";
  write_narrow_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// int64
TEST_F(NarrowTest, Int64Dtype) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "Int64Dtype ";
  at::Tensor t = at::zeros({4, 5}, at::kLong);
  at::Tensor result = t.narrow(0, 1, 2);
  file << std::to_string(static_cast<int>(result.scalar_type())) << " ";
  write_narrow_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// ========== API 变体 ==========

// 测试 narrow_symint
TEST_F(NarrowTest, NarrowSymint) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "NarrowSymint ";
  c10::SymInt start(1);
  c10::SymInt length(2);
  at::Tensor result = tensor.narrow_symint(0, start, length);
  write_narrow_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// 测试 narrow_copy
TEST_F(NarrowTest, NarrowCopy) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "NarrowCopy ";
  at::Tensor result = tensor.narrow_copy(0, 1, 2);
  write_narrow_result_to_file(&file, result);
  // narrow_copy 返回的是拷贝，验证数据独立性
  file << std::to_string(result.is_contiguous()) << " ";
  file << "\n";
  file.saveFile();
}

// 测试 narrow_copy_symint
TEST_F(NarrowTest, NarrowCopySymint) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "NarrowCopySymint ";
  c10::SymInt start(0);
  c10::SymInt length(3);
  at::Tensor result = tensor.narrow_copy_symint(0, start, length);
  write_narrow_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// 测试 narrow 使用 Tensor 作为 start
TEST_F(NarrowTest, NarrowWithTensorStart) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "NarrowWithTensorStart ";
  // 创建一个标量 tensor 作为 start (0-dim tensor)
  at::Tensor start_tensor = at::zeros({}, at::kLong);
  int64_t* start_data = start_tensor.data_ptr<int64_t>();
  start_data[0] = 2;
  at::Tensor result = tensor.narrow(0, start_tensor, 2);
  write_narrow_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// 测试 narrow_symint 使用 Tensor 作为 start
TEST_F(NarrowTest, NarrowSymintWithTensorStart) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "NarrowSymintWithTensorStart ";
  at::Tensor start_tensor = at::zeros({}, at::kLong);
  int64_t* start_data = start_tensor.data_ptr<int64_t>();
  start_data[0] = 1;
  c10::SymInt length(2);
  at::Tensor result = tensor.narrow_symint(1, start_tensor, length);
  write_narrow_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// 测试多次 narrow 操作
TEST_F(NarrowTest, MultipleNarrow) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "MultipleNarrow ";
  // 连续 narrow: {4, 5, 6} -> {2, 5, 6} -> {2, 3, 6} -> {2, 3, 4}
  at::Tensor result = tensor.narrow(0, 1, 2).narrow(1, 1, 3).narrow(2, 1, 4);
  write_narrow_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// 负索引 dim
TEST_F(NarrowTest, NegativeDim) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "NegativeDim ";
  // 使用负索引 dim
  at::Tensor result = tensor.narrow(-1, 1, 4);
  write_narrow_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// ========== 异常行为 ==========

// 越界 start
TEST_F(NarrowTest, OutOfBoundStart) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "OutOfBoundStart ";
  try {
    // start=10 超出 dim 0 的大小 4
    at::Tensor result = tensor.narrow(0, 10, 1);
    write_narrow_result_to_file(&file, result);
  } catch (const std::exception&) {
    file << "exception ";
  }
  file << "\n";
  file.saveFile();
}

// 越界 length
TEST_F(NarrowTest, OutOfBoundLength) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "OutOfBoundLength ";
  try {
    // start=2, length=10 超出 dim 0 的大小 4
    at::Tensor result = tensor.narrow(0, 2, 10);
    write_narrow_result_to_file(&file, result);
  } catch (const std::exception&) {
    file << "exception ";
  }
  file << "\n";
  file.saveFile();
}

// 无效 dim
TEST_F(NarrowTest, InvalidDim) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "InvalidDim ";
  try {
    // dim=10 超出 tensor 的维度数
    at::Tensor result = tensor.narrow(10, 0, 1);
    write_narrow_result_to_file(&file, result);
  } catch (const std::exception&) {
    file << "exception ";
  }
  file << "\n";
  file.saveFile();
}

}  // namespace test
}  // namespace at
