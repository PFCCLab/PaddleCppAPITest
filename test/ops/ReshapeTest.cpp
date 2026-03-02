#include <ATen/ATen.h>
#include <ATen/core/Tensor.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/reshape.h>
#include <ATen/ops/zeros.h>
#include <ATen/ops/zeros_like.h>
#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "../../src/file_manager.h"

extern paddle_api_test::ThreadSafeParam g_custom_param;

namespace at {
namespace test {

using paddle_api_test::FileManerger;
using paddle_api_test::ThreadSafeParam;

// 按 dtype 分发写出 tensor 内容
static void write_reshape_result_to_file(FileManerger* file,
                                         const at::Tensor& result) {
  *file << std::to_string(result.dim()) << " ";
  *file << std::to_string(result.numel()) << " ";
  for (int64_t i = 0; i < result.dim(); ++i) {
    *file << std::to_string(result.sizes()[i]) << " ";
  }
  switch (result.scalar_type()) {
    case at::kFloat: {
      float* data = result.data_ptr<float>();
      for (int64_t i = 0; i < result.numel(); ++i) {
        *file << std::to_string(data[i]) << " ";
      }
      break;
    }
    case at::kDouble: {
      double* data = result.data_ptr<double>();
      for (int64_t i = 0; i < result.numel(); ++i) {
        *file << std::to_string(data[i]) << " ";
      }
      break;
    }
    case at::kInt: {
      int32_t* data = result.data_ptr<int32_t>();
      for (int64_t i = 0; i < result.numel(); ++i) {
        *file << std::to_string(data[i]) << " ";
      }
      break;
    }
    case at::kLong: {
      int64_t* data = result.data_ptr<int64_t>();
      for (int64_t i = 0; i < result.numel(); ++i) {
        *file << std::to_string(data[i]) << " ";
      }
      break;
    }
    case at::kBool: {
      bool* data = result.data_ptr<bool>();
      for (int64_t i = 0; i < result.numel(); ++i) {
        *file << std::to_string(static_cast<int>(data[i])) << " ";
      }
      break;
    }
    default: {
      *file << "unsupported_dtype ";
      break;
    }
  }
}

class ReshapeTest : public ::testing::Test {
 protected:
  void SetUp() override {
    original_tensor = at::zeros({2, 3}, at::kFloat);
    float* data = original_tensor.data_ptr<float>();
    for (int64_t i = 0; i < 6; ++i) {
      data[i] = static_cast<float>(i);
    }
  }
  at::Tensor original_tensor;
};

// ========== 基础功能 ==========

TEST_F(ReshapeTest, Reshape2DTo1D) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();
  file << "Reshape2DTo1D ";
  at::Tensor result = at::reshape(original_tensor, {6});
  write_reshape_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

TEST_F(ReshapeTest, Reshape2DTo3D) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "Reshape2DTo3D ";
  at::Tensor result = at::reshape(original_tensor, {1, 2, 3});
  write_reshape_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

TEST_F(ReshapeTest, ReshapeAutoInferDim) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "ReshapeAutoInferDim ";
  at::Tensor result = at::reshape(original_tensor, {-1});
  write_reshape_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

TEST_F(ReshapeTest, ReshapeInferOneDim) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "ReshapeInferOneDim ";
  at::Tensor result = at::reshape(original_tensor, {3, -1});
  write_reshape_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// ========== Shape 覆盖 ==========

// 标量 reshape
TEST_F(ReshapeTest, ScalarReshape) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "ScalarReshape ";
  at::Tensor scalar = at::zeros({}, at::kFloat);
  scalar.data_ptr<float>()[0] = 42.0f;
  // 标量 reshape 为 {1}
  at::Tensor result = at::reshape(scalar, {1});
  write_reshape_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// 大 shape reshape
TEST_F(ReshapeTest, LargeShapeReshape) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "LargeShapeReshape ";
  at::Tensor large = at::zeros({100, 100}, at::kFloat);
  // {100, 100} -> {10000}
  at::Tensor result = at::reshape(large, {10000});
  file << std::to_string(result.numel()) << " ";
  file << std::to_string(result.dim()) << " ";
  file << "\n";
  file.saveFile();
}

// 零维度 tensor reshape
TEST_F(ReshapeTest, ZeroDimReshape) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "ZeroDimReshape ";
  at::Tensor zero_tensor = at::zeros({2, 0, 3}, at::kFloat);
  // {2, 0, 3} -> {0, 6}
  at::Tensor result = at::reshape(zero_tensor, {0, 6});
  write_reshape_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// 全一维度 reshape
TEST_F(ReshapeTest, AllOneShapeReshape) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "AllOneShapeReshape ";
  at::Tensor t = at::zeros({1, 1, 1}, at::kFloat);
  t.data_ptr<float>()[0] = 7.0f;
  // {1, 1, 1} -> {1}
  at::Tensor result = at::reshape(t, {1});
  write_reshape_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// 非连续 tensor reshape
TEST_F(ReshapeTest, NonContiguousReshape) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "NonContiguousReshape ";
  at::Tensor t = at::zeros({2, 3}, at::kFloat);
  float* data = t.data_ptr<float>();
  for (int64_t i = 0; i < 6; ++i) {
    data[i] = static_cast<float>(i);
  }
  at::Tensor transposed = t.transpose(0, 1);
  file << std::to_string(transposed.is_contiguous()) << " ";
  // 非连续 tensor reshape
  at::Tensor result = at::reshape(transposed, {6});
  write_reshape_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// ========== Dtype 覆盖 ==========

// float64
TEST_F(ReshapeTest, Float64Dtype) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "Float64Dtype ";
  at::Tensor t = at::zeros({2, 3}, at::kDouble);
  double* data = t.data_ptr<double>();
  for (int64_t i = 0; i < 6; ++i) {
    data[i] = static_cast<double>(i);
  }
  at::Tensor result = at::reshape(t, {6});
  write_reshape_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// int32
TEST_F(ReshapeTest, Int32Dtype) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "Int32Dtype ";
  at::Tensor t = at::zeros({2, 3}, at::kInt);
  int32_t* data = t.data_ptr<int32_t>();
  for (int64_t i = 0; i < 6; ++i) {
    data[i] = static_cast<int32_t>(i);
  }
  at::Tensor result = at::reshape(t, {6});
  write_reshape_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// int64
TEST_F(ReshapeTest, Int64Dtype) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "Int64Dtype ";
  at::Tensor t = at::zeros({2, 3}, at::kLong);
  int64_t* data = t.data_ptr<int64_t>();
  for (int64_t i = 0; i < 6; ++i) {
    data[i] = i;
  }
  at::Tensor result = at::reshape(t, {6});
  write_reshape_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// bool
TEST_F(ReshapeTest, BoolDtype) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "BoolDtype ";
  at::Tensor t = at::zeros({2, 3}, at::kBool);
  bool* data = t.data_ptr<bool>();
  for (int64_t i = 0; i < 6; ++i) {
    data[i] = (i % 2 == 0);
  }
  at::Tensor result = at::reshape(t, {6});
  write_reshape_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// ========== API 变体 ==========

// view 方法
TEST_F(ReshapeTest, ViewMethod) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "ViewMethod ";
  at::Tensor result = original_tensor.view({6});
  write_reshape_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// ========== empty_like / zeros_like ==========

TEST_F(ReshapeTest, EmptyLike) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "EmptyLike ";
  at::Tensor result = at::empty_like(original_tensor);
  file << std::to_string(result.dim()) << " ";
  file << std::to_string(result.sizes()[0]) << " ";
  file << std::to_string(result.sizes()[1]) << " ";
  file << std::to_string(static_cast<int>(result.scalar_type())) << " ";
  file << std::to_string(result.data_ptr() != nullptr) << " ";
  file << "\n";
  file.saveFile();
}

TEST_F(ReshapeTest, ZerosLike) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "ZerosLike ";
  at::Tensor result = at::zeros_like(original_tensor);
  write_reshape_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

TEST_F(ReshapeTest, EmptyLikeWithOptions) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "EmptyLikeWithOptions ";
  at::Tensor result =
      at::empty_like(original_tensor, at::TensorOptions().dtype(at::kDouble));
  file << std::to_string(result.dim()) << " ";
  file << std::to_string(result.sizes()[0]) << " ";
  file << std::to_string(result.sizes()[1]) << " ";
  file << std::to_string(static_cast<int>(result.scalar_type())) << " ";
  file << "\n";
  file.saveFile();
}

TEST_F(ReshapeTest, ZerosLikeWithOptions) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "ZerosLikeWithOptions ";
  at::Tensor result =
      at::zeros_like(original_tensor, at::TensorOptions().dtype(at::kInt));
  write_reshape_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// ========== 异常行为 ==========

// 不兼容的 shape (元素数不匹配)
TEST_F(ReshapeTest, IncompatibleShape) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "IncompatibleShape ";
  try {
    // original_tensor 有 6 个元素，但要求 reshape 成 {10}
    at::Tensor result = at::reshape(original_tensor, {10});
    write_reshape_result_to_file(&file, result);
  } catch (const std::exception&) {
    file << "exception ";
  }
  file << "\n";
  file.saveFile();
}

}  // namespace test
}  // namespace at
