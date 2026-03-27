#include <ATen/ATen.h>
#include <ATen/core/Tensor.h>
#include <ATen/ops/abs.h>
#include <ATen/ops/zeros.h>
#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <string>
#include <vector>

#include "../../src/file_manager.h"

extern paddle_api_test::ThreadSafeParam g_custom_param;

namespace at {
namespace test {

using paddle_api_test::FileManerger;
using paddle_api_test::ThreadSafeParam;

// 按 dtype 分发写出 tensor 内容: <ndim> <numel> [sizes...] [values...]
static void write_abs_result_to_file(FileManerger* file,
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

class AbsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // 基准 tensor: {4}, float32, 值为 1, -2, 0, -3.5
    test_tensor = at::zeros({4}, at::kFloat);
    float* data = test_tensor.data_ptr<float>();
    data[0] = 1.0f;
    data[1] = -2.0f;
    data[2] = 0.0f;
    data[3] = -3.5f;
  }
  at::Tensor test_tensor;
};

// ========== 基础功能 ==========

TEST_F(AbsTest, BasicAbs) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();
  file << "BasicAbs ";
  at::Tensor result = at::abs(test_tensor);
  write_abs_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// ========== Shape 覆盖 ==========

// 标量 tensor {}
TEST_F(AbsTest, ScalarTensor) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "ScalarTensor ";
  at::Tensor scalar = at::zeros({}, at::kFloat);
  scalar.data_ptr<float>()[0] = -42.0f;
  at::Tensor result = at::abs(scalar);
  write_abs_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// 小 shape {2, 3}
TEST_F(AbsTest, SmallShape) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "SmallShape ";
  at::Tensor t = at::zeros({2, 3}, at::kFloat);
  float* data = t.data_ptr<float>();
  for (int64_t i = 0; i < 6; ++i) {
    data[i] = static_cast<float>(i - 3);
  }
  at::Tensor result = at::abs(t);
  write_abs_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// 大 shape {100, 100}
TEST_F(AbsTest, LargeShape) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "LargeShape ";
  at::Tensor t = at::ones({100, 100}, at::kFloat);
  float* data = t.data_ptr<float>();
  for (int64_t i = 0; i < 10000; ++i) {
    data[i] = (i % 2 == 0) ? 1.0f : -1.0f;
  }
  at::Tensor result = at::abs(t);
  write_abs_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// 含零维度 {2, 0}
TEST_F(AbsTest, ZeroDimTensor) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "ZeroDimTensor ";
  at::Tensor t = at::zeros({2, 0}, at::kFloat);
  at::Tensor result = at::abs(t);
  write_abs_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// 全一维度 {1, 1, 1}
TEST_F(AbsTest, AllOneShape) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "AllOneShape ";
  at::Tensor t = at::zeros({1, 1, 1}, at::kFloat);
  t.data_ptr<float>()[0] = -5.0f;
  at::Tensor result = at::abs(t);
  write_abs_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// 非连续 tensor (transpose 后)
TEST_F(AbsTest, NonContiguousTensor) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "NonContiguousTensor ";
  at::Tensor t = at::zeros({2, 3}, at::kFloat);
  float* data = t.data_ptr<float>();
  for (int64_t i = 0; i < 6; ++i) {
    data[i] = static_cast<float>(i - 3);
  }
  at::Tensor transposed = t.transpose(0, 1);
  file << std::to_string(transposed.is_contiguous()) << " ";
  at::Tensor result = at::abs(transposed);
  write_abs_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// ========== Dtype 覆盖 ==========

// float64
TEST_F(AbsTest, Float64Dtype) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "Float64Dtype ";
  at::Tensor t = at::zeros({4}, at::kDouble);
  double* data = t.data_ptr<double>();
  data[0] = 1.5;
  data[1] = -2.5;
  data[2] = 0.0;
  data[3] = -3.5;
  at::Tensor result = at::abs(t);
  write_abs_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// int32
TEST_F(AbsTest, Int32Dtype) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "Int32Dtype ";
  at::Tensor t = at::zeros({4}, at::kInt);
  int32_t* data = t.data_ptr<int32_t>();
  data[0] = 10;
  data[1] = -20;
  data[2] = 0;
  data[3] = -30;
  at::Tensor result = at::abs(t);
  write_abs_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// int64
TEST_F(AbsTest, Int64Dtype) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "Int64Dtype ";
  at::Tensor t = at::zeros({4}, at::kLong);
  int64_t* data = t.data_ptr<int64_t>();
  data[0] = 100;
  data[1] = -200;
  data[2] = 0;
  data[3] = -300;
  at::Tensor result = at::abs(t);
  write_abs_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// bool
// TEST_F(AbsTest, BoolDtype) {
//   auto file_name = g_custom_param.get();
//   FileManerger file(file_name);
//   file.openAppend();
//   file << "BoolDtype ";
//   at::Tensor t = at::zeros({4}, at::kBool);
//   bool* data = t.data_ptr<bool>();
//   data[0] = true;
//   data[1] = false;
//   data[2] = true;
//   data[3] = false;
//   at::Tensor result = at::abs(t);
//   write_abs_result_to_file(&file, result);
//   file << "\n";
//   file.saveFile();
// }

// ========== 值域覆盖 ==========

// 正数
TEST_F(AbsTest, PositiveValues) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "PositiveValues ";
  at::Tensor t = at::zeros({3}, at::kFloat);
  float* data = t.data_ptr<float>();
  data[0] = 1.5f;
  data[1] = 3.0f;
  data[2] = 7.2f;
  at::Tensor result = at::abs(t);
  write_abs_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// 负数
TEST_F(AbsTest, NegativeValues) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "NegativeValues ";
  at::Tensor t = at::zeros({3}, at::kFloat);
  float* data = t.data_ptr<float>();
  data[0] = -1.5f;
  data[1] = -3.0f;
  data[2] = -7.2f;
  at::Tensor result = at::abs(t);
  write_abs_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// 含零
TEST_F(AbsTest, ZeroValues) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "ZeroValues ";
  at::Tensor t = at::zeros({3}, at::kFloat);
  float* data = t.data_ptr<float>();
  data[0] = 0.0f;
  data[1] = -0.0f;
  data[2] = 1.0f;
  at::Tensor result = at::abs(t);
  write_abs_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// 含 Inf
TEST_F(AbsTest, InfValues) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "InfValues ";
  at::Tensor t = at::zeros({3}, at::kFloat);
  float* data = t.data_ptr<float>();
  data[0] = std::numeric_limits<float>::infinity();
  data[1] = -std::numeric_limits<float>::infinity();
  data[2] = 1.0f;
  at::Tensor result = at::abs(t);
  write_abs_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// 含 NaN
TEST_F(AbsTest, NaNValues) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "NaNValues ";
  at::Tensor t = at::zeros({3}, at::kFloat);
  float* data = t.data_ptr<float>();
  data[0] = std::numeric_limits<float>::quiet_NaN();
  data[1] = -std::numeric_limits<float>::quiet_NaN();
  data[2] = 1.0f;
  at::Tensor result = at::abs(t);
  write_abs_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// 极值
TEST_F(AbsTest, ExtremeValues) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "ExtremeValues ";
  at::Tensor t = at::zeros({4}, at::kFloat);
  float* data = t.data_ptr<float>();
  data[0] = std::numeric_limits<float>::max();
  data[1] = -std::numeric_limits<float>::max();
  data[2] = std::numeric_limits<float>::min();
  data[3] = -std::numeric_limits<float>::min();
  at::Tensor result = at::abs(t);
  write_abs_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// ========== API 变体 ==========

// 原地操作 abs_()
TEST_F(AbsTest, InplaceAbs) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "InplaceAbs ";
  at::Tensor t = test_tensor.clone();
  void* original_ptr = t.data_ptr();
  t.abs_();
  file << std::to_string(t.data_ptr() == original_ptr) << " ";
  write_abs_result_to_file(&file, t);
  file << "\n";
  file.saveFile();
}

// 方法调用 t.abs()
TEST_F(AbsTest, MethodAbs) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "MethodAbs ";
  at::Tensor result = test_tensor.abs();
  write_abs_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

}  // namespace test
}  // namespace at
