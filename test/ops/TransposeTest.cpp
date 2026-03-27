#include <ATen/ATen.h>
#include <ATen/core/Tensor.h>
#include <ATen/ops/transpose.h>
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

class TransposeTest : public ::testing::Test {
 protected:
  void SetUp() override {
    tensor = at::zeros({2, 3, 4}, at::kFloat);
    float* data = tensor.data_ptr<float>();
    for (int64_t i = 0; i < 24; ++i) {
      data[i] = static_cast<float>(i);
    }
  }

  at::Tensor tensor;
};

static void write_transpose_result_to_file(FileManerger* file,
                                           const at::Tensor& result) {
  *file << std::to_string(result.dim()) << " ";
  *file << std::to_string(result.numel()) << " ";
  for (int64_t i = 0; i < result.dim(); ++i) {
    *file << std::to_string(result.sizes()[i]) << " ";
  }
  for (int64_t i = 0; i < result.dim(); ++i) {
    *file << std::to_string(result.strides()[i]) << " ";
  }
  *file << std::to_string(result.is_contiguous()) << " ";
}

// 2D 转置
TEST_F(TransposeTest, Transpose2D) {
  at::Tensor t2d = at::zeros({3, 5}, at::kFloat);
  float* data = t2d.data_ptr<float>();
  for (int64_t i = 0; i < 15; ++i) {
    data[i] = static_cast<float>(i);
  }
  at::Tensor result = at::transpose(t2d, 0, 1);
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();
  file << "Transpose2D ";
  write_transpose_result_to_file(&file, result);
  at::Tensor cont = result.contiguous();
  float* rdata = cont.data_ptr<float>();
  for (int64_t i = 0; i < 15; ++i) {
    file << std::to_string(rdata[i]) << " ";
  }
  file << "\n";
  file.saveFile();
}

// 3D 转置：交换 dim 0 和 dim 2
TEST_F(TransposeTest, Transpose3D_Dim0_Dim2) {
  at::Tensor result = at::transpose(tensor, 0, 2);
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "Transpose3D_Dim0_Dim2 ";
  write_transpose_result_to_file(&file, result);
  at::Tensor cont = result.contiguous();
  float* rdata = cont.data_ptr<float>();
  for (int64_t i = 0; i < 24; ++i) {
    file << std::to_string(rdata[i]) << " ";
  }
  file << "\n";
  file.saveFile();
}

// 3D 转置：交换 dim 1 和 dim 2
TEST_F(TransposeTest, Transpose3D_Dim1_Dim2) {
  at::Tensor result = at::transpose(tensor, 1, 2);
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "Transpose3D_Dim1_Dim2 ";
  write_transpose_result_to_file(&file, result);
  at::Tensor cont = result.contiguous();
  float* rdata = cont.data_ptr<float>();
  for (int64_t i = 0; i < 24; ++i) {
    file << std::to_string(rdata[i]) << " ";
  }
  file << "\n";
  file.saveFile();
}

// 相同维度转置（no-op）
TEST_F(TransposeTest, TransposeSameDim) {
  at::Tensor result = at::transpose(tensor, 1, 1);
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "TransposeSameDim ";
  write_transpose_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// 成员函数版本
TEST_F(TransposeTest, TransposeMemberFunction) {
  at::Tensor result = tensor.transpose(0, 2);
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "TransposeMemberFunction ";
  write_transpose_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// Inplace 转置
TEST_F(TransposeTest, TransposeInplace) {
  at::Tensor temp = at::zeros({2, 3, 4}, at::kFloat);
  at::Tensor result = temp.transpose_(0, 2);
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "TransposeInplace ";
  write_transpose_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// Scalar 参数版本
TEST_F(TransposeTest, TransposeScalarAPI) {
  at::Tensor result = at::transpose(tensor, 0, 2);
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "TransposeScalarAPI ";
  write_transpose_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// Double 类型
TEST_F(TransposeTest, TransposeDouble) {
  at::Tensor td = at::zeros({2, 3, 4}, at::kDouble);
  double* data = td.data_ptr<double>();
  for (int64_t i = 0; i < 24; ++i) {
    data[i] = static_cast<double>(i) * 0.5;
  }
  at::Tensor result = at::transpose(td, 0, 2);
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "TransposeDouble ";
  file << std::to_string(result.dim()) << " ";
  file << std::to_string(result.numel()) << " ";
  file << std::to_string(static_cast<int>(result.scalar_type())) << " ";
  for (int64_t i = 0; i < result.dim(); ++i) {
    file << std::to_string(result.sizes()[i]) << " ";
  }
  at::Tensor cont = result.contiguous();
  double* rdata = cont.data_ptr<double>();
  for (int64_t i = 0; i < 24; ++i) {
    file << std::to_string(rdata[i]) << " ";
  }
  file << "\n";
  file.saveFile();
}

// Int 类型
TEST_F(TransposeTest, TransposeInt) {
  at::Tensor ti = at::zeros({2, 3, 4}, at::kInt);
  int* data = ti.data_ptr<int>();
  for (int64_t i = 0; i < 24; ++i) {
    data[i] = static_cast<int>(i) - 12;
  }
  at::Tensor result = at::transpose(ti, 0, 1);
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "TransposeInt ";
  file << std::to_string(result.dim()) << " ";
  file << std::to_string(result.numel()) << " ";
  file << std::to_string(static_cast<int>(result.scalar_type())) << " ";
  for (int64_t i = 0; i < result.dim(); ++i) {
    file << std::to_string(result.sizes()[i]) << " ";
  }
  at::Tensor cont = result.contiguous();
  int* rdata = cont.data_ptr<int>();
  for (int64_t i = 0; i < 24; ++i) {
    file << std::to_string(rdata[i]) << " ";
  }
  file << "\n";
  file.saveFile();
}

// Long 类型
TEST_F(TransposeTest, TransposeLong) {
  at::Tensor tl = at::zeros({2, 3, 4}, at::kLong);
  int64_t* data = tl.data_ptr<int64_t>();
  for (int64_t i = 0; i < 24; ++i) {
    data[i] = i * 100;
  }
  at::Tensor result = at::transpose(tl, 1, 2);
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "TransposeLong ";
  file << std::to_string(result.dim()) << " ";
  file << std::to_string(result.numel()) << " ";
  file << std::to_string(static_cast<int>(result.scalar_type())) << " ";
  for (int64_t i = 0; i < result.dim(); ++i) {
    file << std::to_string(result.sizes()[i]) << " ";
  }
  at::Tensor cont = result.contiguous();
  int64_t* rdata = cont.data_ptr<int64_t>();
  for (int64_t i = 0; i < 24; ++i) {
    file << std::to_string(rdata[i]) << " ";
  }
  file << "\n";
  file.saveFile();
}

// 大 shape
TEST_F(TransposeTest, TransposeLargeShape) {
  at::Tensor large = at::zeros({100, 100}, at::kFloat);
  float* data = large.data_ptr<float>();
  for (int64_t i = 0; i < 10000; ++i) {
    data[i] = static_cast<float>(i);
  }
  at::Tensor result = at::transpose(large, 0, 1);
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "TransposeLargeShape ";
  file << std::to_string(result.dim()) << " ";
  file << std::to_string(result.numel()) << " ";
  file << std::to_string(result.sizes()[0]) << " ";
  file << std::to_string(result.sizes()[1]) << " ";
  // 只输出前几个和角落元素
  at::Tensor cont = result.contiguous();
  float* rdata = cont.data_ptr<float>();
  file << std::to_string(rdata[0]) << " ";
  file << std::to_string(rdata[1]) << " ";
  file << std::to_string(rdata[100]) << " ";
  file << std::to_string(rdata[9999]) << " ";
  file << "\n";
  file.saveFile();
}

// 特殊值
TEST_F(TransposeTest, TransposeSpecialValues) {
  at::Tensor special = at::zeros({2, 3}, at::kFloat);
  float* data = special.data_ptr<float>();
  data[0] = std::numeric_limits<float>::infinity();
  data[1] = -std::numeric_limits<float>::infinity();
  data[2] = std::nanf("");
  data[3] = 0.0f;
  data[4] = -0.0f;
  data[5] = 1e38f;
  at::Tensor result = at::transpose(special, 0, 1);
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "TransposeSpecialValues ";
  file << std::to_string(result.dim()) << " ";
  file << std::to_string(result.numel()) << " ";
  at::Tensor cont = result.contiguous();
  float* rdata = cont.data_ptr<float>();
  for (int64_t i = 0; i < 6; ++i) {
    file << std::to_string(rdata[i]) << " ";
  }
  file << "\n";
  file.saveFile();
}

}  // namespace test
}  // namespace at
