#include <ATen/ATen.h>
#include <ATen/core/Tensor.h>
#include <ATen/ops/ones.h>
#include <ATen/ops/squeeze.h>
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

class SqueezeTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // {1, 3, 1, 4, 1}
    tensor = at::zeros({1, 3, 1, 4, 1}, at::kFloat);
    float* data = tensor.data_ptr<float>();
    for (int64_t i = 0; i < 12; ++i) {
      data[i] = static_cast<float>(i);
    }
  }

  at::Tensor tensor;
};

static void write_squeeze_result_to_file(FileManerger* file,
                                         const at::Tensor& result) {
  *file << std::to_string(result.dim()) << " ";
  *file << std::to_string(result.numel()) << " ";
  for (int64_t i = 0; i < result.dim(); ++i) {
    *file << std::to_string(result.sizes()[i]) << " ";
  }
}

// squeeze 所有大小为 1 的维度：{1,3,1,4,1} -> {3,4}
TEST_F(SqueezeTest, SqueezeAll) {
  at::Tensor result = at::squeeze(tensor);
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();
  write_squeeze_result_to_file(&file, result);
  at::Tensor cont = result.contiguous();
  float* data = cont.data_ptr<float>();
  for (int64_t i = 0; i < cont.numel(); ++i) {
    file << std::to_string(data[i]) << " ";
  }
  file.saveFile();
}

// squeeze 指定 dim=0（大小为 1）：{1,3,1,4,1} -> {3,1,4,1}
TEST_F(SqueezeTest, SqueezeSingleDim) {
  at::Tensor result = at::squeeze(tensor, 0);
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  write_squeeze_result_to_file(&file, result);
  file.saveFile();
}

// squeeze 指定 dim=1（大小不为 1，应 no-op）：{1,3,1,4,1} -> {1,3,1,4,1}
TEST_F(SqueezeTest, SqueezeNonOneDim) {
  at::Tensor result = at::squeeze(tensor, 1);
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  write_squeeze_result_to_file(&file, result);
  file.saveFile();
}

// squeeze 多个维度：IntArrayRef
TEST_F(SqueezeTest, SqueezeMultipleDims) {
  at::Tensor result = at::squeeze(tensor, {0, 2, 4});
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  write_squeeze_result_to_file(&file, result);
  file.saveFile();
}

// 成员函数：squeeze()
TEST_F(SqueezeTest, SqueezeMemberAll) {
  at::Tensor result = tensor.squeeze();
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  write_squeeze_result_to_file(&file, result);
  file.saveFile();
}

// 成员函数：squeeze(dim)
TEST_F(SqueezeTest, SqueezeMemberDim) {
  at::Tensor result = tensor.squeeze(2);
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  write_squeeze_result_to_file(&file, result);
  file.saveFile();
}

// inplace squeeze_()
TEST_F(SqueezeTest, SqueezeInplaceAll) {
  at::Tensor t = at::zeros({1, 3, 1, 4, 1}, at::kFloat);
  float* data = t.data_ptr<float>();
  for (int64_t i = 0; i < 12; ++i) {
    data[i] = static_cast<float>(i);
  }
  t.squeeze_();
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  write_squeeze_result_to_file(&file, t);
  file.saveFile();
}

// inplace squeeze_(dim)
TEST_F(SqueezeTest, SqueezeInplaceDim) {
  at::Tensor t = at::zeros({1, 3, 1, 4, 1}, at::kFloat);
  t.squeeze_(0);
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  write_squeeze_result_to_file(&file, t);
  file.saveFile();
}

// inplace squeeze_(IntArrayRef)
TEST_F(SqueezeTest, SqueezeInplaceMultipleDims) {
  at::Tensor t = at::zeros({1, 3, 1, 4, 1}, at::kFloat);
  t.squeeze_({0, 2});
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  write_squeeze_result_to_file(&file, t);
  file.saveFile();
}

// 标量 tensor（0-d）squeeze：应 no-op
TEST_F(SqueezeTest, SqueezeScalar) {
  at::Tensor scalar = at::zeros({}, at::kFloat);
  scalar.fill_(42.0f);
  at::Tensor result = at::squeeze(scalar);
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  write_squeeze_result_to_file(&file, result);
  float* data = result.data_ptr<float>();
  file << std::to_string(data[0]) << " ";
  file.saveFile();
}

// 全一维度 tensor：{1,1,1}
TEST_F(SqueezeTest, SqueezeAllOnes) {
  at::Tensor t = at::zeros({1, 1, 1}, at::kFloat);
  t.fill_(7.0f);
  at::Tensor result = at::squeeze(t);
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  write_squeeze_result_to_file(&file, result);
  float* data = result.data_ptr<float>();
  file << std::to_string(data[0]) << " ";
  file.saveFile();
}

// Double 类型
TEST_F(SqueezeTest, SqueezeDouble) {
  at::Tensor td = at::zeros({1, 4, 1}, at::kDouble);
  double* data = td.data_ptr<double>();
  for (int64_t i = 0; i < 4; ++i) {
    data[i] = static_cast<double>(i) * 1.5;
  }
  at::Tensor result = at::squeeze(td);
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << std::to_string(result.dim()) << " ";
  file << std::to_string(result.numel()) << " ";
  file << std::to_string(static_cast<int>(result.scalar_type())) << " ";
  double* rdata = result.data_ptr<double>();
  for (int64_t i = 0; i < result.numel(); ++i) {
    file << std::to_string(rdata[i]) << " ";
  }
  file.saveFile();
}

// Int 类型
TEST_F(SqueezeTest, SqueezeInt) {
  at::Tensor ti = at::zeros({1, 5, 1}, at::kInt);
  int* data = ti.data_ptr<int>();
  for (int64_t i = 0; i < 5; ++i) {
    data[i] = static_cast<int>(i) - 2;
  }
  at::Tensor result = at::squeeze(ti);
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << std::to_string(result.dim()) << " ";
  file << std::to_string(result.numel()) << " ";
  file << std::to_string(static_cast<int>(result.scalar_type())) << " ";
  int* rdata = result.data_ptr<int>();
  for (int64_t i = 0; i < result.numel(); ++i) {
    file << std::to_string(rdata[i]) << " ";
  }
  file.saveFile();
}

// Long 类型
TEST_F(SqueezeTest, SqueezeLong) {
  at::Tensor tl = at::zeros({1, 3, 1}, at::kLong);
  int64_t* data = tl.data_ptr<int64_t>();
  for (int64_t i = 0; i < 3; ++i) {
    data[i] = i * 10000;
  }
  at::Tensor result = at::squeeze(tl);
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << std::to_string(result.dim()) << " ";
  file << std::to_string(result.numel()) << " ";
  file << std::to_string(static_cast<int>(result.scalar_type())) << " ";
  int64_t* rdata = result.data_ptr<int64_t>();
  for (int64_t i = 0; i < result.numel(); ++i) {
    file << std::to_string(rdata[i]) << " ";
  }
  file.saveFile();
}

// 大 shape
TEST_F(SqueezeTest, SqueezeLargeShape) {
  at::Tensor large = at::zeros({1, 10000, 1}, at::kFloat);
  float* data = large.data_ptr<float>();
  for (int64_t i = 0; i < 10000; ++i) {
    data[i] = static_cast<float>(i);
  }
  at::Tensor result = at::squeeze(large);
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  write_squeeze_result_to_file(&file, result);
  float* rdata = result.data_ptr<float>();
  file << std::to_string(rdata[0]) << " ";
  file << std::to_string(rdata[9999]) << " ";
  file.saveFile();
}

}  // namespace test
}  // namespace at
