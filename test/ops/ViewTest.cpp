#include <ATen/ATen.h>
#include <ATen/core/Tensor.h>
#include <ATen/ops/view.h>
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

class ViewTest : public ::testing::Test {
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

static void write_view_result_to_file(FileManerger* file,
                                      const at::Tensor& result) {
  *file << std::to_string(result.dim()) << " ";
  *file << std::to_string(result.numel()) << " ";
  for (int64_t i = 0; i < result.dim(); ++i) {
    *file << std::to_string(result.sizes()[i]) << " ";
  }
}

// view {2,3,4} -> {24}
TEST_F(ViewTest, ViewFlatten) {
  at::Tensor result = tensor.view({24});
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();
  write_view_result_to_file(&file, result);
  float* data = result.data_ptr<float>();
  for (int64_t i = 0; i < 24; ++i) {
    file << std::to_string(data[i]) << " ";
  }
  file.saveFile();
}

// view {2,3,4} -> {6,4}
TEST_F(ViewTest, View3DTo2D) {
  at::Tensor result = tensor.view({6, 4});
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  write_view_result_to_file(&file, result);
  float* data = result.data_ptr<float>();
  for (int64_t i = 0; i < 24; ++i) {
    file << std::to_string(data[i]) << " ";
  }
  file.saveFile();
}

// view {2,3,4} -> {2,12}
TEST_F(ViewTest, ViewMergeLastDims) {
  at::Tensor result = tensor.view({2, 12});
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  write_view_result_to_file(&file, result);
  file.saveFile();
}

// view {2,3,4} -> {4,3,2}
TEST_F(ViewTest, ViewDifferentShape) {
  at::Tensor result = tensor.view({4, 3, 2});
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  write_view_result_to_file(&file, result);
  float* data = result.data_ptr<float>();
  for (int64_t i = 0; i < 24; ++i) {
    file << std::to_string(data[i]) << " ";
  }
  file.saveFile();
}

// view 使用 -1 推断维度
TEST_F(ViewTest, ViewAutoInfer) {
  at::Tensor result = tensor.view({-1});
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  write_view_result_to_file(&file, result);
  file.saveFile();
}

// view 使用 -1 推断部分维度
TEST_F(ViewTest, ViewAutoInferPartial) {
  at::Tensor result = tensor.view({2, -1});
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  write_view_result_to_file(&file, result);
  file.saveFile();
}

// 成员函数版本
TEST_F(ViewTest, ViewMemberFunction) {
  at::Tensor result = tensor.view({6, 4});
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  write_view_result_to_file(&file, result);
  file.saveFile();
}

// view 共享 storage
TEST_F(ViewTest, ViewSharesStorage) {
  at::Tensor result = tensor.view({6, 4});
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  // 验证 data_ptr 相同
  file << std::to_string(result.data_ptr() == tensor.data_ptr()) << " ";
  file << std::to_string(result.numel()) << " ";
  file.saveFile();
}

// view dtype：float -> 以 int 视角查看
TEST_F(ViewTest, ViewDtype) {
  at::Tensor result = tensor.view(at::kInt);
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << std::to_string(result.dim()) << " ";
  file << std::to_string(result.numel()) << " ";
  file << std::to_string(static_cast<int>(result.scalar_type())) << " ";
  for (int64_t i = 0; i < result.dim(); ++i) {
    file << std::to_string(result.sizes()[i]) << " ";
  }
  file.saveFile();
}

// 成员函数 view dtype
TEST_F(ViewTest, ViewDtypeMember) {
  at::Tensor result = tensor.view(at::kInt);
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << std::to_string(result.dim()) << " ";
  file << std::to_string(result.numel()) << " ";
  file << std::to_string(static_cast<int>(result.scalar_type())) << " ";
  file.saveFile();
}

// Double 类型
TEST_F(ViewTest, ViewDouble) {
  at::Tensor td = at::zeros({2, 3}, at::kDouble);
  double* data = td.data_ptr<double>();
  for (int64_t i = 0; i < 6; ++i) {
    data[i] = static_cast<double>(i) * 0.1;
  }
  at::Tensor result = td.view({6});
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << std::to_string(result.dim()) << " ";
  file << std::to_string(result.numel()) << " ";
  file << std::to_string(static_cast<int>(result.scalar_type())) << " ";
  double* rdata = result.data_ptr<double>();
  for (int64_t i = 0; i < 6; ++i) {
    file << std::to_string(rdata[i]) << " ";
  }
  file.saveFile();
}

// Int 类型
TEST_F(ViewTest, ViewInt) {
  at::Tensor ti = at::zeros({4, 3}, at::kInt);
  int* data = ti.data_ptr<int>();
  for (int64_t i = 0; i < 12; ++i) {
    data[i] = static_cast<int>(i) - 6;
  }
  at::Tensor result = ti.view({2, 6});
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << std::to_string(result.dim()) << " ";
  file << std::to_string(result.numel()) << " ";
  file << std::to_string(static_cast<int>(result.scalar_type())) << " ";
  int* rdata = result.data_ptr<int>();
  for (int64_t i = 0; i < 12; ++i) {
    file << std::to_string(rdata[i]) << " ";
  }
  file.saveFile();
}

// Long 类型
TEST_F(ViewTest, ViewLong) {
  at::Tensor tl = at::zeros({3, 4}, at::kLong);
  int64_t* data = tl.data_ptr<int64_t>();
  for (int64_t i = 0; i < 12; ++i) {
    data[i] = i * 1000;
  }
  at::Tensor result = tl.view({12});
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << std::to_string(result.dim()) << " ";
  file << std::to_string(result.numel()) << " ";
  file << std::to_string(static_cast<int>(result.scalar_type())) << " ";
  int64_t* rdata = result.data_ptr<int64_t>();
  for (int64_t i = 0; i < 12; ++i) {
    file << std::to_string(rdata[i]) << " ";
  }
  file.saveFile();
}

// 大 shape
TEST_F(ViewTest, ViewLargeShape) {
  at::Tensor large = at::zeros({100, 100}, at::kFloat);
  float* data = large.data_ptr<float>();
  for (int64_t i = 0; i < 10000; ++i) {
    data[i] = static_cast<float>(i);
  }
  at::Tensor result = large.view({10000});
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  write_view_result_to_file(&file, result);
  float* rdata = result.data_ptr<float>();
  file << std::to_string(rdata[0]) << " ";
  file << std::to_string(rdata[9999]) << " ";
  file.saveFile();
}

// view 到高维
TEST_F(ViewTest, ViewToHighDim) {
  at::Tensor result = tensor.view({1, 2, 3, 4, 1});
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  write_view_result_to_file(&file, result);
  file.saveFile();
}

}  // namespace test
}  // namespace at
