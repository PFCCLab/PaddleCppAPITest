#include <ATen/ATen.h>
#include <ATen/core/Tensor.h>
#include <ATen/ops/unsqueeze.h>
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

class UnsqueezeTest : public ::testing::Test {
 protected:
  void SetUp() override {
    tensor = at::zeros({3, 4}, at::kFloat);
    float* data = tensor.data_ptr<float>();
    for (int64_t i = 0; i < 12; ++i) {
      data[i] = static_cast<float>(i);
    }
  }

  at::Tensor tensor;
};

static void write_unsqueeze_result_to_file(FileManerger* file,
                                           const at::Tensor& result) {
  *file << std::to_string(result.dim()) << " ";
  *file << std::to_string(result.numel()) << " ";
  for (int64_t i = 0; i < result.dim(); ++i) {
    *file << std::to_string(result.sizes()[i]) << " ";
  }
}

// unsqueeze dim=0：{3,4} -> {1,3,4}
TEST_F(UnsqueezeTest, UnsqueezeBasic) {
  at::Tensor result = at::unsqueeze(tensor, 0);
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();
  write_unsqueeze_result_to_file(&file, result);
  float* data = result.data_ptr<float>();
  for (int64_t i = 0; i < 12; ++i) {
    file << std::to_string(data[i]) << " ";
  }
  file.saveFile();
}

// unsqueeze dim=1：{3,4} -> {3,1,4}
TEST_F(UnsqueezeTest, UnsqueezeMiddle) {
  at::Tensor result = at::unsqueeze(tensor, 1);
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  write_unsqueeze_result_to_file(&file, result);
  file.saveFile();
}

// unsqueeze dim=2（最后）：{3,4} -> {3,4,1}
TEST_F(UnsqueezeTest, UnsqueezeLastDim) {
  at::Tensor result = at::unsqueeze(tensor, 2);
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  write_unsqueeze_result_to_file(&file, result);
  file.saveFile();
}

// unsqueeze dim=-1：{3,4} -> {3,4,1}
TEST_F(UnsqueezeTest, UnsqueezeNegativeDim) {
  at::Tensor result = at::unsqueeze(tensor, -1);
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  write_unsqueeze_result_to_file(&file, result);
  file.saveFile();
}

// 成员函数版本
TEST_F(UnsqueezeTest, UnsqueezeMemberFunction) {
  at::Tensor result = tensor.unsqueeze(0);
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  write_unsqueeze_result_to_file(&file, result);
  file.saveFile();
}

// inplace unsqueeze_
TEST_F(UnsqueezeTest, UnsqueezeInplace) {
  at::Tensor t = at::zeros({3, 4}, at::kFloat);
  float* data = t.data_ptr<float>();
  for (int64_t i = 0; i < 12; ++i) {
    data[i] = static_cast<float>(i);
  }
  t.unsqueeze_(0);
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  write_unsqueeze_result_to_file(&file, t);
  file.saveFile();
}

// 标量 tensor unsqueeze：{} -> {1}
TEST_F(UnsqueezeTest, UnsqueezeScalar) {
  at::Tensor scalar = at::zeros({}, at::kFloat);
  scalar.fill_(42.0f);
  at::Tensor result = at::unsqueeze(scalar, 0);
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  write_unsqueeze_result_to_file(&file, result);
  float* data = result.data_ptr<float>();
  file << std::to_string(data[0]) << " ";
  file.saveFile();
}

// 连续 unsqueeze：{4} -> {1,4} -> {1,1,4} -> {1,1,1,4}
TEST_F(UnsqueezeTest, UnsqueezeMultiple) {
  at::Tensor t = at::zeros({4}, at::kFloat);
  float* data = t.data_ptr<float>();
  for (int64_t i = 0; i < 4; ++i) {
    data[i] = static_cast<float>(i + 1);
  }
  at::Tensor r1 = at::unsqueeze(t, 0);
  at::Tensor r2 = at::unsqueeze(r1, 0);
  at::Tensor result = at::unsqueeze(r2, 0);
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  write_unsqueeze_result_to_file(&file, result);
  float* rdata = result.data_ptr<float>();
  for (int64_t i = 0; i < 4; ++i) {
    file << std::to_string(rdata[i]) << " ";
  }
  file.saveFile();
}

// Double 类型
TEST_F(UnsqueezeTest, UnsqueezeDouble) {
  at::Tensor td = at::zeros({3, 4}, at::kDouble);
  double* data = td.data_ptr<double>();
  for (int64_t i = 0; i < 12; ++i) {
    data[i] = static_cast<double>(i) * 0.5;
  }
  at::Tensor result = at::unsqueeze(td, 1);
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

// Int 类型
TEST_F(UnsqueezeTest, UnsqueezeInt) {
  at::Tensor ti = at::zeros({3, 4}, at::kInt);
  int* data = ti.data_ptr<int>();
  for (int64_t i = 0; i < 12; ++i) {
    data[i] = static_cast<int>(i) - 6;
  }
  at::Tensor result = at::unsqueeze(ti, 0);
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << std::to_string(result.dim()) << " ";
  file << std::to_string(result.numel()) << " ";
  file << std::to_string(static_cast<int>(result.scalar_type())) << " ";
  for (int64_t i = 0; i < result.dim(); ++i) {
    file << std::to_string(result.sizes()[i]) << " ";
  }
  int* rdata = result.data_ptr<int>();
  for (int64_t i = 0; i < result.numel(); ++i) {
    file << std::to_string(rdata[i]) << " ";
  }
  file.saveFile();
}

// Long 类型
TEST_F(UnsqueezeTest, UnsqueezeLong) {
  at::Tensor tl = at::zeros({5}, at::kLong);
  int64_t* data = tl.data_ptr<int64_t>();
  for (int64_t i = 0; i < 5; ++i) {
    data[i] = i * 1000;
  }
  at::Tensor result = at::unsqueeze(tl, 0);
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << std::to_string(result.dim()) << " ";
  file << std::to_string(result.numel()) << " ";
  file << std::to_string(static_cast<int>(result.scalar_type())) << " ";
  for (int64_t i = 0; i < result.dim(); ++i) {
    file << std::to_string(result.sizes()[i]) << " ";
  }
  int64_t* rdata = result.data_ptr<int64_t>();
  for (int64_t i = 0; i < result.numel(); ++i) {
    file << std::to_string(rdata[i]) << " ";
  }
  file.saveFile();
}

// 大 shape
TEST_F(UnsqueezeTest, UnsqueezeLargeShape) {
  at::Tensor large = at::zeros({10000}, at::kFloat);
  float* data = large.data_ptr<float>();
  for (int64_t i = 0; i < 10000; ++i) {
    data[i] = static_cast<float>(i);
  }
  at::Tensor result = at::unsqueeze(large, 0);
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  write_unsqueeze_result_to_file(&file, result);
  float* rdata = result.data_ptr<float>();
  file << std::to_string(rdata[0]) << " ";
  file << std::to_string(rdata[9999]) << " ";
  file.saveFile();
}

// squeeze 后 unsqueeze 恢复
TEST_F(UnsqueezeTest, UnsqueezeAfterSqueeze) {
  at::Tensor t = at::zeros({1, 3, 1, 4}, at::kFloat);
  float* data = t.data_ptr<float>();
  for (int64_t i = 0; i < 12; ++i) {
    data[i] = static_cast<float>(i);
  }
  at::Tensor squeezed = at::squeeze(t, 0);         // {3, 1, 4}
  at::Tensor result = at::unsqueeze(squeezed, 0);  // {1, 3, 1, 4}
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  write_unsqueeze_result_to_file(&file, result);
  file << std::to_string(result.numel() == t.numel()) << " ";
  file.saveFile();
}

}  // namespace test
}  // namespace at
