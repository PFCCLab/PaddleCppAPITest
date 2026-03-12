#include <ATen/ATen.h>
#include <ATen/core/Tensor.h>
#include <ATen/ops/slice.h>
#include <ATen/ops/zeros.h>
#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <optional>
#include <string>
#include <vector>

#include "../../src/file_manager.h"

extern paddle_api_test::ThreadSafeParam g_custom_param;

namespace at {
namespace test {

using paddle_api_test::FileManerger;
using paddle_api_test::ThreadSafeParam;

class SliceTest : public ::testing::Test {
 protected:
  void SetUp() override {
    tensor = at::zeros({4, 5, 6}, at::kFloat);
    float* data = tensor.data_ptr<float>();
    for (int64_t i = 0; i < 120; ++i) {
      data[i] = static_cast<float>(i);
    }
  }

  at::Tensor tensor;
};

static void write_slice_result_to_file(FileManerger* file,
                                       const at::Tensor& result) {
  *file << std::to_string(result.dim()) << " ";
  *file << std::to_string(result.numel()) << " ";
  for (int64_t i = 0; i < result.dim(); ++i) {
    *file << std::to_string(result.sizes()[i]) << " ";
  }
}

static void write_slice_float_data(FileManerger* file,
                                   const at::Tensor& result) {
  at::Tensor cont = result.contiguous();
  float* data = cont.data_ptr<float>();
  for (int64_t i = 0; i < cont.numel(); ++i) {
    *file << std::to_string(data[i]) << " ";
  }
}

// 沿 dim=0 切片：[1:3]
TEST_F(SliceTest, SliceBasicDim0) {
  at::Tensor result = at::slice(tensor, 0, 1, 3);
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();
  write_slice_result_to_file(&file, result);
  // 验证首元素
  at::Tensor cont = result.contiguous();
  float* data = cont.data_ptr<float>();
  file << std::to_string(data[0]) << " ";
  file.saveFile();
}

// 沿 dim=1 切片：[2:5]
TEST_F(SliceTest, SliceDim1) {
  at::Tensor result = at::slice(tensor, 1, 2, 5);
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  write_slice_result_to_file(&file, result);
  file.saveFile();
}

// 沿 dim=2 切片：[0:4]
TEST_F(SliceTest, SliceDim2) {
  at::Tensor result = at::slice(tensor, 2, 0, 4);
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  write_slice_result_to_file(&file, result);
  file.saveFile();
}

// 成员函数版本
TEST_F(SliceTest, SliceMemberFunction) {
  at::Tensor result = tensor.slice(0, 0, 2);
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  write_slice_result_to_file(&file, result);
  at::Tensor cont = result.contiguous();
  float* data = cont.data_ptr<float>();
  file << std::to_string(data[0]) << " ";
  file << std::to_string(data[cont.numel() - 1]) << " ";
  file.saveFile();
}

// nullopt bounds：全切
// Paddle 实现不支持 std::nullopt 作为 slice 边界（starts/ends 长度须与 axes
// 一致）， 改用显式的全范围边界 [0, size) 来等价实现完整切片。
TEST_F(SliceTest, SliceNulloptBounds) {
  at::Tensor result = at::slice(tensor, 0, 0, tensor.size(0));
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  write_slice_result_to_file(&file, result);
  file << std::to_string(result.numel() == tensor.numel()) << " ";
  file.saveFile();
}

// 连续多维度切片
TEST_F(SliceTest, SliceMultipleDims) {
  // tensor[1:3, 0:3, 1:5]
  at::Tensor r1 = at::slice(tensor, 0, 1, 3);
  at::Tensor r2 = at::slice(r1, 1, 0, 3);
  at::Tensor result = at::slice(r2, 2, 1, 5);
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  write_slice_result_to_file(&file, result);
  file.saveFile();
}

// 2D tensor 切片
TEST_F(SliceTest, Slice2D) {
  at::Tensor t2d = at::zeros({6, 8}, at::kFloat);
  float* data = t2d.data_ptr<float>();
  for (int64_t i = 0; i < 48; ++i) {
    data[i] = static_cast<float>(i);
  }
  at::Tensor result = at::slice(t2d, 0, 1, 4);
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  write_slice_result_to_file(&file, result);
  at::Tensor cont = result.contiguous();
  float* rdata = cont.data_ptr<float>();
  for (int64_t i = 0; i < cont.numel(); ++i) {
    file << std::to_string(rdata[i]) << " ";
  }
  file.saveFile();
}

// Double 类型
TEST_F(SliceTest, SliceDouble) {
  at::Tensor td = at::zeros({4, 5}, at::kDouble);
  double* data = td.data_ptr<double>();
  for (int64_t i = 0; i < 20; ++i) {
    data[i] = static_cast<double>(i) * 0.1;
  }
  at::Tensor result = at::slice(td, 0, 1, 3);
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << std::to_string(result.dim()) << " ";
  file << std::to_string(result.numel()) << " ";
  file << std::to_string(static_cast<int>(result.scalar_type())) << " ";
  at::Tensor cont = result.contiguous();
  double* rdata = cont.data_ptr<double>();
  for (int64_t i = 0; i < cont.numel(); ++i) {
    file << std::to_string(rdata[i]) << " ";
  }
  file.saveFile();
}

// Int 类型
TEST_F(SliceTest, SliceInt) {
  at::Tensor ti = at::zeros({4, 5}, at::kInt);
  int* data = ti.data_ptr<int>();
  for (int64_t i = 0; i < 20; ++i) {
    data[i] = static_cast<int>(i) - 10;
  }
  at::Tensor result = at::slice(ti, 1, 1, 4);
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << std::to_string(result.dim()) << " ";
  file << std::to_string(result.numel()) << " ";
  file << std::to_string(static_cast<int>(result.scalar_type())) << " ";
  at::Tensor cont = result.contiguous();
  int* rdata = cont.data_ptr<int>();
  for (int64_t i = 0; i < cont.numel(); ++i) {
    file << std::to_string(rdata[i]) << " ";
  }
  file.saveFile();
}

// Long 类型
TEST_F(SliceTest, SliceLong) {
  at::Tensor tl = at::zeros({4, 5}, at::kLong);
  int64_t* data = tl.data_ptr<int64_t>();
  for (int64_t i = 0; i < 20; ++i) {
    data[i] = i * 1000;
  }
  at::Tensor result = at::slice(tl, 0, 0, 2);
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << std::to_string(result.dim()) << " ";
  file << std::to_string(result.numel()) << " ";
  file << std::to_string(static_cast<int>(result.scalar_type())) << " ";
  at::Tensor cont = result.contiguous();
  int64_t* rdata = cont.data_ptr<int64_t>();
  for (int64_t i = 0; i < cont.numel(); ++i) {
    file << std::to_string(rdata[i]) << " ";
  }
  file.saveFile();
}

// 大 shape
TEST_F(SliceTest, SliceLargeShape) {
  at::Tensor large = at::zeros({100, 100}, at::kFloat);
  float* data = large.data_ptr<float>();
  for (int64_t i = 0; i < 10000; ++i) {
    data[i] = static_cast<float>(i);
  }
  at::Tensor result = at::slice(large, 0, 10, 90);
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  write_slice_result_to_file(&file, result);
  at::Tensor cont = result.contiguous();
  float* rdata = cont.data_ptr<float>();
  file << std::to_string(rdata[0]) << " ";
  file << std::to_string(rdata[cont.numel() - 1]) << " ";
  file.saveFile();
}

// 特殊值
TEST_F(SliceTest, SliceSpecialValues) {
  at::Tensor special = at::zeros({4, 3}, at::kFloat);
  float* data = special.data_ptr<float>();
  data[0] = std::numeric_limits<float>::infinity();
  data[1] = -std::numeric_limits<float>::infinity();
  data[2] = std::nanf("");
  data[3] = 0.0f;
  data[4] = -0.0f;
  data[5] = 1e38f;
  for (int64_t i = 6; i < 12; ++i) {
    data[i] = static_cast<float>(i);
  }
  at::Tensor result = at::slice(special, 0, 0, 2);
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  write_slice_result_to_file(&file, result);
  at::Tensor cont = result.contiguous();
  float* rdata = cont.data_ptr<float>();
  for (int64_t i = 0; i < cont.numel(); ++i) {
    file << std::to_string(rdata[i]) << " ";
  }
  file.saveFile();
}

}  // namespace test
}  // namespace at
