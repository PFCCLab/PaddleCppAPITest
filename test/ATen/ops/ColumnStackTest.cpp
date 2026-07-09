#include <ATen/ATen.h>
#include <ATen/core/Tensor.h>
#include <ATen/ops/column_stack.h>
#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "src/file_manager.h"

extern paddle_api_test::ThreadSafeParam g_custom_param;

namespace at {
namespace test {

using paddle_api_test::FileManerger;
using paddle_api_test::ThreadSafeParam;

static void write_column_stack_result_to_file(FileManerger* file,
                                              const at::Tensor& result) {
  *file << std::to_string(result.dim()) << " ";
  *file << std::to_string(result.numel()) << " ";
  *file << std::to_string(static_cast<int>(result.scalar_type())) << " ";
  for (int64_t i = 0; i < result.dim(); ++i) {
    *file << std::to_string(result.sizes()[i]) << " ";
  }
}

static void write_float_values_to_file(FileManerger* file,
                                       const at::Tensor& result) {
  auto contiguous = result.contiguous();
  const float* data = contiguous.data_ptr<float>();
  for (int64_t i = 0; i < contiguous.numel(); ++i) {
    *file << std::to_string(data[i]) << " ";
  }
}

class ColumnStackTest : public ::testing::Test {
 protected:
  void SetUp() override {}
};

// ========== 基础功能 ==========

TEST_F(ColumnStackTest, Basic1D) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();
  file << "Basic1D ";
  at::Tensor v1 = at::arange(3, at::kFloat);
  at::Tensor v2 = at::arange(3, 6, at::kFloat);
  std::vector<at::Tensor> tensors = {v1, v2};
  at::Tensor result = at::column_stack(tensors);
  write_column_stack_result_to_file(&file, result);
  write_float_values_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

TEST_F(ColumnStackTest, Basic2D) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "Basic2D ";
  at::Tensor m1 = at::ones({2, 3}, at::kFloat);
  at::Tensor m2 = at::ones({2, 2}, at::kFloat);
  std::vector<at::Tensor> tensors = {m1, m2};
  at::Tensor result = at::column_stack(tensors);
  write_column_stack_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

TEST_F(ColumnStackTest, Mixed1DAnd2D) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "Mixed1DAnd2D ";
  at::Tensor vec = at::zeros({3}, at::kFloat);
  at::Tensor mat = at::zeros({3, 2}, at::kFloat);
  std::vector<at::Tensor> tensors = {vec, mat};
  at::Tensor result = at::column_stack(tensors);
  write_column_stack_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

TEST_F(ColumnStackTest, ScalarTensors) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "ScalarTensors ";
  at::Tensor s1 = at::ones({}, at::kFloat);
  at::Tensor s2 = at::ones({}, at::kFloat);
  std::vector<at::Tensor> tensors = {s1, s2};
  at::Tensor result = at::column_stack(tensors);
  write_column_stack_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

TEST_F(ColumnStackTest, ScalarAndVectorMismatch) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "ScalarAndVectorMismatch ";
  try {
    at::Tensor scalar = at::ones({}, at::kFloat);
    at::Tensor vec = at::arange(3, at::kFloat);
    std::vector<at::Tensor> tensors = {scalar, vec};
    at::Tensor result = at::column_stack(tensors);
    write_column_stack_result_to_file(&file, result);
    write_float_values_to_file(&file, result);
  } catch (const std::exception&) {
    file << "exception ";
  }
  file << "\n";
  file.saveFile();
}

TEST_F(ColumnStackTest, ScalarAndSingleRowMatrix) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "ScalarAndSingleRowMatrix ";
  at::Tensor scalar = at::full({}, 2.0f, at::kFloat);
  at::Tensor matrix = at::arange(2, at::kFloat).reshape({1, 2});
  std::vector<at::Tensor> tensors = {scalar, matrix};
  at::Tensor result = at::column_stack(tensors);
  write_column_stack_result_to_file(&file, result);
  write_float_values_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

TEST_F(ColumnStackTest, ScalarAndMatrixMismatch) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "ScalarAndMatrixMismatch ";
  try {
    at::Tensor scalar = at::ones({}, at::kFloat);
    at::Tensor matrix = at::ones({3, 2}, at::kFloat);
    std::vector<at::Tensor> tensors = {scalar, matrix};
    at::Tensor result = at::column_stack(tensors);
    write_column_stack_result_to_file(&file, result);
    write_float_values_to_file(&file, result);
  } catch (const std::exception&) {
    file << "exception ";
  }
  file << "\n";
  file.saveFile();
}

TEST_F(ColumnStackTest, SingleTensor) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "SingleTensor ";
  at::Tensor vec = at::zeros({3}, at::kFloat);
  std::vector<at::Tensor> tensors = {vec};
  at::Tensor result = at::column_stack(tensors);
  write_column_stack_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// ========== Shape 覆盖 ==========

TEST_F(ColumnStackTest, LargeShape) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "LargeShape ";
  at::Tensor t1 = at::zeros({100}, at::kFloat);
  at::Tensor t2 = at::zeros({100}, at::kFloat);
  std::vector<at::Tensor> tensors = {t1, t2};
  at::Tensor result = at::column_stack(tensors);
  write_column_stack_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

TEST_F(ColumnStackTest, ZeroDim) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "ZeroDim ";
  at::Tensor t1 = at::zeros({0}, at::kFloat);
  at::Tensor t2 = at::zeros({0}, at::kFloat);
  std::vector<at::Tensor> tensors = {t1, t2};
  at::Tensor result = at::column_stack(tensors);
  write_column_stack_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

TEST_F(ColumnStackTest, AllOneShape) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "AllOneShape ";
  at::Tensor t1 = at::ones({1}, at::kFloat);
  at::Tensor t2 = at::ones({1}, at::kFloat);
  std::vector<at::Tensor> tensors = {t1, t2};
  at::Tensor result = at::column_stack(tensors);
  write_column_stack_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// ========== Dtype 覆盖 ==========

TEST_F(ColumnStackTest, DtypeFloat) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "DtypeFloat ";
  at::Tensor t1 = at::zeros({3}, at::kFloat);
  at::Tensor t2 = at::zeros({3}, at::kFloat);
  std::vector<at::Tensor> tensors = {t1, t2};
  at::Tensor result = at::column_stack(tensors);
  write_column_stack_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

TEST_F(ColumnStackTest, DtypeDouble) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "DtypeDouble ";
  at::Tensor t1 = at::zeros({3}, at::kDouble);
  at::Tensor t2 = at::zeros({3}, at::kDouble);
  std::vector<at::Tensor> tensors = {t1, t2};
  at::Tensor result = at::column_stack(tensors);
  write_column_stack_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

TEST_F(ColumnStackTest, DtypeInt) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "DtypeInt ";
  at::Tensor t1 = at::zeros({3}, at::kInt);
  at::Tensor t2 = at::zeros({3}, at::kInt);
  std::vector<at::Tensor> tensors = {t1, t2};
  at::Tensor result = at::column_stack(tensors);
  write_column_stack_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

TEST_F(ColumnStackTest, DtypeLong) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "DtypeLong ";
  at::Tensor t1 = at::zeros({3}, at::kLong);
  at::Tensor t2 = at::zeros({3}, at::kLong);
  std::vector<at::Tensor> tensors = {t1, t2};
  at::Tensor result = at::column_stack(tensors);
  write_column_stack_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// ========== 异常行为 ==========

TEST_F(ColumnStackTest, EmptyList) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "EmptyList ";
  try {
    std::vector<at::Tensor> tensors = {};
    at::Tensor result = at::column_stack(tensors);
    write_column_stack_result_to_file(&file, result);
  } catch (const std::exception&) {
    file << "exception ";
  }
  file << "\n";
  file.saveFile();
}

TEST_F(ColumnStackTest, MismatchedRows) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "MismatchedRows ";
  try {
    at::Tensor t1 = at::zeros({3}, at::kFloat);
    at::Tensor t2 = at::zeros({4}, at::kFloat);
    std::vector<at::Tensor> tensors = {t1, t2};
    at::Tensor result = at::column_stack(tensors);
    write_column_stack_result_to_file(&file, result);
  } catch (const std::exception&) {
    file << "exception ";
  }
  file << "\n";
  file.saveFile();
}

}  // namespace test
}  // namespace at
