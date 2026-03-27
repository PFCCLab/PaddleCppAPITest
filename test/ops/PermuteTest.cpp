#include <ATen/ATen.h>
#include <ATen/core/Tensor.h>
#include <ATen/ops/permute.h>
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

class PermuteTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // 创建一个 2x3x4 的 tensor，填充递增值
    tensor = at::zeros({2, 3, 4}, at::kFloat);
    float* data = tensor.data_ptr<float>();
    for (int64_t i = 0; i < 24; ++i) {
      data[i] = static_cast<float>(i);
    }
  }

  at::Tensor tensor;
};

static void write_permute_result_to_file(FileManerger* file,
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

static void write_permute_data_to_file(FileManerger* file,
                                       const at::Tensor& result) {
  at::Tensor cont = result.contiguous();
  float* data = cont.data_ptr<float>();
  for (int64_t i = 0; i < cont.numel(); ++i) {
    *file << std::to_string(data[i]) << " ";
  }
}

// 基本置换：{0,2,1} — shape {2,3,4} -> {2,4,3}
TEST_F(PermuteTest, BasicPermute) {
  at::Tensor result = at::permute(tensor, {0, 2, 1});
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();
  file << "BasicPermute ";
  write_permute_result_to_file(&file, result);
  write_permute_data_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// 完全逆转：{2,1,0} — shape {2,3,4} -> {4,3,2}
TEST_F(PermuteTest, PermuteReverse) {
  at::Tensor result = at::permute(tensor, {2, 1, 0});
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "PermuteReverse ";
  write_permute_result_to_file(&file, result);
  write_permute_data_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// 恒等置换：{0,1,2} — shape 不变
TEST_F(PermuteTest, PermuteIdentity) {
  at::Tensor result = at::permute(tensor, {0, 1, 2});
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "PermuteIdentity ";
  write_permute_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// 成员函数版本
TEST_F(PermuteTest, PermuteMemberFunction) {
  at::Tensor result = tensor.permute({1, 0, 2});
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "PermuteMemberFunction ";
  write_permute_result_to_file(&file, result);
  write_permute_data_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// 2D 转置
TEST_F(PermuteTest, Permute2D) {
  at::Tensor t2d = at::zeros({3, 5}, at::kFloat);
  float* data = t2d.data_ptr<float>();
  for (int64_t i = 0; i < 15; ++i) {
    data[i] = static_cast<float>(i);
  }
  at::Tensor result = at::permute(t2d, {1, 0});
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "Permute2D ";
  write_permute_result_to_file(&file, result);
  write_permute_data_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// 4D tensor 置换
TEST_F(PermuteTest, PermuteHighDim) {
  at::Tensor t4d = at::zeros({2, 3, 4, 5}, at::kFloat);
  float* data = t4d.data_ptr<float>();
  for (int64_t i = 0; i < 120; ++i) {
    data[i] = static_cast<float>(i);
  }
  at::Tensor result = at::permute(t4d, {0, 3, 1, 2});
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "PermuteHighDim ";
  write_permute_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// 大 shape：元素数 >= 10000
TEST_F(PermuteTest, PermuteLargeShape) {
  at::Tensor large = at::zeros({10, 20, 50}, at::kFloat);
  float* data = large.data_ptr<float>();
  for (int64_t i = 0; i < 10000; ++i) {
    data[i] = static_cast<float>(i);
  }
  at::Tensor result = at::permute(large, {2, 0, 1});
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "PermuteLargeShape ";
  write_permute_result_to_file(&file, result);
  // 只输出前 10 个元素避免文件过大
  at::Tensor cont = result.contiguous();
  float* rdata = cont.data_ptr<float>();
  for (int64_t i = 0; i < 10; ++i) {
    file << std::to_string(rdata[i]) << " ";
  }
  file << "\n";
  file.saveFile();
}

// Double 类型
TEST_F(PermuteTest, PermuteDouble) {
  at::Tensor td = at::zeros({2, 3, 4}, at::kDouble);
  double* data = td.data_ptr<double>();
  for (int64_t i = 0; i < 24; ++i) {
    data[i] = static_cast<double>(i) * 0.1;
  }
  at::Tensor result = at::permute(td, {2, 0, 1});
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "PermuteDouble ";
  *(&file) << std::to_string(result.dim()) << " ";
  *(&file) << std::to_string(result.numel()) << " ";
  *(&file) << std::to_string(static_cast<int>(result.scalar_type())) << " ";
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
TEST_F(PermuteTest, PermuteInt) {
  at::Tensor ti = at::zeros({2, 3, 4}, at::kInt);
  int* data = ti.data_ptr<int>();
  for (int64_t i = 0; i < 24; ++i) {
    data[i] = static_cast<int>(i) - 12;
  }
  at::Tensor result = at::permute(ti, {1, 2, 0});
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "PermuteInt ";
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
TEST_F(PermuteTest, PermuteLong) {
  at::Tensor tl = at::zeros({2, 3, 4}, at::kLong);
  int64_t* data = tl.data_ptr<int64_t>();
  for (int64_t i = 0; i < 24; ++i) {
    data[i] = i * 1000;
  }
  at::Tensor result = at::permute(tl, {2, 1, 0});
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "PermuteLong ";
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

// 非连续 tensor 再 permute
TEST_F(PermuteTest, PermuteNonContiguous) {
  // 先 transpose 产生非连续 tensor，再 permute
  at::Tensor transposed = tensor.transpose(0, 1);  // {3, 2, 4}
  at::Tensor result = at::permute(transposed.contiguous(), {2, 0, 1});
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "PermuteNonContiguous ";
  write_permute_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// 特殊值：NaN / Inf
TEST_F(PermuteTest, PermuteSpecialValues) {
  at::Tensor special = at::zeros({2, 3}, at::kFloat);
  float* data = special.data_ptr<float>();
  data[0] = std::numeric_limits<float>::infinity();
  data[1] = -std::numeric_limits<float>::infinity();
  data[2] = std::nanf("");
  data[3] = 0.0f;
  data[4] = -0.0f;
  data[5] = 1e38f;

  at::Tensor result = at::permute(special, {1, 0});
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "PermuteSpecialValues ";
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
