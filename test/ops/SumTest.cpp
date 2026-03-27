#include <ATen/ATen.h>
#include <ATen/core/Tensor.h>
#include <ATen/ops/sum.h>
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
static void write_sum_result_to_file(FileManerger* file,
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

class SumTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // 基准 tensor: {2, 3}, float32, 值为 1~6
    std::vector<int64_t> shape = {2, 3};
    test_tensor = at::zeros(shape, at::kFloat);
    float* data = test_tensor.data_ptr<float>();
    for (int64_t i = 0; i < 6; ++i) {
      data[i] = static_cast<float>(i + 1);
    }
  }
  at::Tensor test_tensor;
};

// ========== 基础功能 ==========

// 全元素求和 (第一个用例，createFile)
TEST_F(SumTest, SumAllElements) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();
  file << "SumAllElements ";
  at::Tensor result = at::sum(test_tensor);
  write_sum_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// 指定输出 dtype
TEST_F(SumTest, SumWithDtype) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "SumWithDtype ";
  at::Tensor result = at::sum(test_tensor, at::kDouble);
  file << std::to_string(static_cast<int>(result.scalar_type())) << " ";
  write_sum_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// ========== dim / keepdim 变体 ==========

// 沿 dim 0 求和
TEST_F(SumTest, SumAlongDim0) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "SumAlongDim0 ";
  at::Tensor result = at::sum(test_tensor, {0}, false);
  write_sum_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// 沿 dim 1 求和
TEST_F(SumTest, SumAlongDim1) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "SumAlongDim1 ";
  at::Tensor result = at::sum(test_tensor, {1}, false);
  write_sum_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// keepdim=true
TEST_F(SumTest, SumWithKeepdim) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "SumWithKeepdim ";
  at::Tensor result = at::sum(test_tensor, {0}, true);
  write_sum_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// out= 变体
TEST_F(SumTest, SumOutFunction) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "SumOutFunction ";
  at::Tensor output = at::zeros({}, at::kFloat);
  at::Tensor& result = at::sum_out(output, test_tensor);
  file << std::to_string(&result == &output) << " ";
  write_sum_result_to_file(&file, output);
  file << "\n";
  file.saveFile();
}

// ========== Shape 覆盖 ==========

// 标量 tensor {}
TEST_F(SumTest, SumScalar) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "SumScalar ";
  at::Tensor scalar = at::zeros({}, at::kFloat);
  float* data = scalar.data_ptr<float>();
  data[0] = 42.0f;
  at::Tensor result = at::sum(scalar);
  write_sum_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// 大 shape {100, 100}
TEST_F(SumTest, SumLargeShape) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "SumLargeShape ";
  at::Tensor large = at::ones({100, 100}, at::kFloat);
  at::Tensor result = at::sum(large);
  write_sum_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// 含零维度 {2, 0}
TEST_F(SumTest, SumZeroDim) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "SumZeroDim ";
  at::Tensor zero_tensor = at::zeros({2, 0}, at::kFloat);
  at::Tensor result = at::sum(zero_tensor);
  write_sum_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// 全一 shape {1, 1, 1}
TEST_F(SumTest, SumAllOneShape) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "SumAllOneShape ";
  at::Tensor t = at::zeros({1, 1, 1}, at::kFloat);
  t.data_ptr<float>()[0] = 5.0f;
  at::Tensor result = at::sum(t);
  write_sum_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// 非连续 tensor (transpose 后求和)
TEST_F(SumTest, SumNonContiguous) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "SumNonContiguous ";
  // test_tensor {2,3} transpose -> {3,2}, 非连续
  at::Tensor transposed = test_tensor.transpose(0, 1);
  file << std::to_string(transposed.is_contiguous()) << " ";
  at::Tensor result = at::sum(transposed);
  write_sum_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// ========== Dtype 覆盖 ==========

// float64
TEST_F(SumTest, SumFloat64) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "SumFloat64 ";
  at::Tensor t = at::zeros({2, 3}, at::kDouble);
  double* data = t.data_ptr<double>();
  for (int64_t i = 0; i < 6; ++i) {
    data[i] = static_cast<double>(i + 1);
  }
  at::Tensor result = at::sum(t);
  write_sum_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// int32
TEST_F(SumTest, SumInt32) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "SumInt32 ";
  at::Tensor t = at::zeros({4}, at::kInt);
  int32_t* data = t.data_ptr<int32_t>();
  data[0] = 10;
  data[1] = 20;
  data[2] = 30;
  data[3] = 40;
  at::Tensor result = at::sum(t);
  write_sum_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// int64
TEST_F(SumTest, SumInt64) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "SumInt64 ";
  at::Tensor t = at::zeros({3}, at::kLong);
  int64_t* data = t.data_ptr<int64_t>();
  data[0] = 100;
  data[1] = 200;
  data[2] = 300;
  at::Tensor result = at::sum(t);
  write_sum_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// ========== 值域覆盖 ==========

// 含负数
TEST_F(SumTest, SumNegativeValues) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "SumNegativeValues ";
  at::Tensor t = at::zeros({4}, at::kFloat);
  float* data = t.data_ptr<float>();
  data[0] = -1.5f;
  data[1] = 2.5f;
  data[2] = -3.5f;
  data[3] = 4.5f;
  at::Tensor result = at::sum(t);
  write_sum_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// 含 Inf
TEST_F(SumTest, SumWithInf) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "SumWithInf ";
  at::Tensor t = at::zeros({3}, at::kFloat);
  float* data = t.data_ptr<float>();
  data[0] = 1.0f;
  data[1] = std::numeric_limits<float>::infinity();
  data[2] = 2.0f;
  at::Tensor result = at::sum(t);
  write_sum_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// 含 NaN
TEST_F(SumTest, SumWithNaN) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "SumWithNaN ";
  at::Tensor t = at::zeros({3}, at::kFloat);
  float* data = t.data_ptr<float>();
  data[0] = 1.0f;
  data[1] = std::numeric_limits<float>::quiet_NaN();
  data[2] = 2.0f;
  at::Tensor result = at::sum(t);
  write_sum_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// 极值
TEST_F(SumTest, SumExtremeValues) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "SumExtremeValues ";
  at::Tensor t = at::zeros({2}, at::kFloat);
  float* data = t.data_ptr<float>();
  data[0] = std::numeric_limits<float>::max();
  data[1] = std::numeric_limits<float>::max();
  at::Tensor result = at::sum(t);
  write_sum_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// ========== 异常行为 ==========

// 无效 dim
TEST_F(SumTest, SumInvalidDim) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "SumInvalidDim ";
  try {
    // test_tensor 是 2D，dim=5 越界
    at::Tensor result = at::sum(test_tensor, {5}, false);
    write_sum_result_to_file(&file, result);
  } catch (const std::exception& e) {
    file << "exception ";
  }
  file << "\n";
  file.saveFile();
}

// --------------------------------------------------------------------------
// 以下为 tensor 成员函数形式 sum() 的测试
// --------------------------------------------------------------------------

// 测试 tensor.sum()：无参数，对所有元素求和
TEST_F(SumTest, MemberSumAllElements) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "MemberSumAllElements ";
  at::Tensor result = test_tensor.sum();
  file << std::to_string(result.dim()) << " ";
  file << std::to_string(result.numel()) << " ";
  float result_value = *result.data_ptr<float>();
  file << std::to_string(result_value) << " ";
  file << "\n";
  file.saveFile();
}

// 测试 tensor.sum(dtype)：指定输出类型
TEST_F(SumTest, MemberSumWithDtype) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "MemberSumWithDtype ";
  at::Tensor result = test_tensor.sum(at::kDouble);
  file << std::to_string(static_cast<int>(result.scalar_type())) << " ";
  double result_value = *result.data_ptr<double>();
  file << std::to_string(result_value) << " ";
  file << "\n";
  file.saveFile();
}

// 测试 tensor.sum(dim, keepdim=false)：沿 dim=0 求和并降维
TEST_F(SumTest, MemberSumAlongDim0) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "MemberSumAlongDim0 ";
  at::Tensor result = test_tensor.sum({0}, /*keepdim=*/false);
  file << std::to_string(result.dim()) << " ";
  file << std::to_string(result.numel()) << " ";
  float* data = result.data_ptr<float>();
  file << std::to_string(data[0]) << " ";
  file << std::to_string(data[1]) << " ";
  file << std::to_string(data[2]) << " ";
  file << "\n";
  file.saveFile();
}

// 测试 tensor.sum(dim, keepdim=true)：保留维度
TEST_F(SumTest, MemberSumKeepdim) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "MemberSumKeepdim ";
  at::Tensor result = test_tensor.sum({1}, /*keepdim=*/true);
  file << std::to_string(result.dim()) << " ";
  file << std::to_string(result.sizes()[0]) << " ";
  file << std::to_string(result.sizes()[1]) << " ";
  float* data = result.data_ptr<float>();
  file << std::to_string(data[0]) << " ";  // 1+2+3=6
  file << std::to_string(data[1]) << " ";  // 4+5+6=15
  file << "\n";
  file.saveFile();
}

// 测试 tensor.sum(dim, keepdim, dtype)：沿多维度求和并指定输出类型
TEST_F(SumTest, MemberSumMultiDimWithDtype) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "MemberSumMultiDimWithDtype ";
  at::Tensor result = test_tensor.sum(at::IntArrayRef{0, 1},
                                      /*keepdim=*/false,
                                      std::make_optional(at::kDouble));
  file << std::to_string(result.dim()) << " ";
  file << std::to_string(result.numel()) << " ";
  file << std::to_string(static_cast<int>(result.scalar_type())) << " ";
  double val = *result.data_ptr<double>();
  file << std::to_string(val) << " ";  // 21
  file << "\n";
  file.saveFile();
}

}  // namespace test
}  // namespace at
