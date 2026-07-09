#include <ATen/ATen.h>
#include <ATen/core/Tensor.h>
#include <ATen/ops/full.h>
#include <ATen/ops/ones.h>
#include <ATen/ops/vstack.h>
#include <ATen/ops/zeros.h>
#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "src/file_manager.h"

extern paddle_api_test::ThreadSafeParam g_custom_param;

namespace at {
namespace test {

using paddle_api_test::FileManerger;
using paddle_api_test::ThreadSafeParam;

class VStackTest : public ::testing::Test {
 protected:
  void SetUp() override {}
};

static void write_vstack_result_to_file(FileManerger* file,
                                        const at::Tensor& result) {
  *file << std::to_string(result.dim()) << " ";
  *file << std::to_string(result.numel()) << " ";
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

// ========== Shape 覆盖 ==========

// 标量 (0-d tensor) -> vstack 后变成 2D
TEST_F(VStackTest, Scalar0D) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();
  file << "Scalar0D ";
  auto t1 = at::full({}, 1.0f, at::TensorOptions().dtype(at::kFloat));
  auto t2 = at::full({}, 2.0f, at::TensorOptions().dtype(at::kFloat));
  std::vector<at::Tensor> tensors = {t1, t2};
  at::Tensor result = at::vstack(tensors);
  write_vstack_result_to_file(&file, result);
  write_float_values_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// 小 shape
TEST_F(VStackTest, SmallShape2D) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "SmallShape2D ";
  auto t1 = at::ones({2, 3}, at::TensorOptions().dtype(at::kFloat));
  auto t2 = at::zeros({2, 3}, at::TensorOptions().dtype(at::kFloat));
  std::vector<at::Tensor> tensors = {t1, t2};
  at::Tensor result = at::vstack(tensors);
  write_vstack_result_to_file(&file, result);
  write_float_values_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// 1D tensor -> vstack 后变成 2D
TEST_F(VStackTest, SmallShape1D) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "SmallShape1D ";
  auto t1 = at::ones({3}, at::TensorOptions().dtype(at::kFloat));
  auto t2 = at::zeros({3}, at::TensorOptions().dtype(at::kFloat));
  std::vector<at::Tensor> tensors = {t1, t2};
  at::Tensor result = at::vstack(tensors);
  write_vstack_result_to_file(&file, result);
  write_float_values_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// 大 shape
TEST_F(VStackTest, LargeShape) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "LargeShape ";
  auto t1 = at::ones({50, 100}, at::TensorOptions().dtype(at::kFloat));
  auto t2 = at::zeros({50, 100}, at::TensorOptions().dtype(at::kFloat));
  std::vector<at::Tensor> tensors = {t1, t2};
  at::Tensor result = at::vstack(tensors);
  write_vstack_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// 边界 shape: 含零维度
TEST_F(VStackTest, ZeroDimShape) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "ZeroDimShape ";
  auto t1 = at::ones({0, 3}, at::TensorOptions().dtype(at::kFloat));
  auto t2 = at::zeros({2, 3}, at::TensorOptions().dtype(at::kFloat));
  std::vector<at::Tensor> tensors = {t1, t2};
  at::Tensor result = at::vstack(tensors);
  write_vstack_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// 边界 shape: 全一维度
TEST_F(VStackTest, AllOnesShape) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "AllOnesShape ";
  auto t1 = at::ones({1, 1}, at::TensorOptions().dtype(at::kFloat));
  auto t2 = at::zeros({1, 1}, at::TensorOptions().dtype(at::kFloat));
  std::vector<at::Tensor> tensors = {t1, t2};
  at::Tensor result = at::vstack(tensors);
  write_vstack_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// 混合维度
TEST_F(VStackTest, MixedDims) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "MixedDims ";
  auto t1 = at::ones({3}, at::TensorOptions().dtype(at::kFloat));
  auto t2 = at::zeros({1, 3}, at::TensorOptions().dtype(at::kFloat));
  std::vector<at::Tensor> tensors = {t1, t2};
  at::Tensor result = at::vstack(tensors);
  write_vstack_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// ========== Dtype 覆盖 ==========

TEST_F(VStackTest, DtypeFloat) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "DtypeFloat ";
  auto t1 = at::ones({2, 3}, at::TensorOptions().dtype(at::kFloat));
  auto t2 = at::zeros({2, 3}, at::TensorOptions().dtype(at::kFloat));
  std::vector<at::Tensor> tensors = {t1, t2};
  at::Tensor result = at::vstack(tensors);
  file << std::to_string(static_cast<int>(result.scalar_type())) << " ";
  write_vstack_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

TEST_F(VStackTest, DtypeDouble) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "DtypeDouble ";
  auto t1 = at::ones({2, 3}, at::TensorOptions().dtype(at::kDouble));
  auto t2 = at::zeros({2, 3}, at::TensorOptions().dtype(at::kDouble));
  std::vector<at::Tensor> tensors = {t1, t2};
  at::Tensor result = at::vstack(tensors);
  file << std::to_string(static_cast<int>(result.scalar_type())) << " ";
  write_vstack_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

TEST_F(VStackTest, DtypeInt32) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "DtypeInt32 ";
  auto t1 = at::ones({2, 3}, at::TensorOptions().dtype(at::kInt));
  auto t2 = at::zeros({2, 3}, at::TensorOptions().dtype(at::kInt));
  std::vector<at::Tensor> tensors = {t1, t2};
  at::Tensor result = at::vstack(tensors);
  file << std::to_string(static_cast<int>(result.scalar_type())) << " ";
  write_vstack_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

TEST_F(VStackTest, DtypeInt64) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "DtypeInt64 ";
  auto t1 = at::ones({2, 3}, at::TensorOptions().dtype(at::kLong));
  auto t2 = at::zeros({2, 3}, at::TensorOptions().dtype(at::kLong));
  std::vector<at::Tensor> tensors = {t1, t2};
  at::Tensor result = at::vstack(tensors);
  file << std::to_string(static_cast<int>(result.scalar_type())) << " ";
  write_vstack_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// ========== 异常行为 ==========

TEST_F(VStackTest, EmptyListThrows) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "EmptyListThrows ";
  std::vector<at::Tensor> tensors = {};
  try {
    at::Tensor result = at::vstack(tensors);
    file << "no_exception ";
    write_vstack_result_to_file(&file, result);
  } catch (const std::exception&) {
    file << "exception ";
  }
  file << "\n";
  file.saveFile();
}

TEST_F(VStackTest, MismatchedShapeThrows) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "MismatchedShapeThrows ";
  std::vector<at::Tensor> tensors = {
      at::ones({2, 3}, at::TensorOptions().dtype(at::kFloat)),
      at::zeros({2, 4}, at::TensorOptions().dtype(at::kFloat))};
  try {
    at::Tensor result = at::vstack(tensors);
    file << "no_exception ";
    write_vstack_result_to_file(&file, result);
    write_float_values_to_file(&file, result);
  } catch (const std::exception&) {
    file << "exception ";
  }
  file << "\n";
  file.saveFile();
}

}  // namespace test
}  // namespace at
