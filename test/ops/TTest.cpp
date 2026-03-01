#include <ATen/ATen.h>
#include <ATen/core/Tensor.h>
#include <ATen/ops/ones.h>
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

class TTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // 2x3 tensor，values 0..5
    tensor2d = at::zeros({2, 3}, at::kFloat);
    float* d = tensor2d.data_ptr<float>();
    for (int64_t i = 0; i < 6; ++i) d[i] = static_cast<float>(i);

    // 1-dim tensor
    tensor1d = at::zeros({5}, at::kFloat);
    float* d1 = tensor1d.data_ptr<float>();
    for (int64_t i = 0; i < 5; ++i) d1[i] = static_cast<float>(i);
  }

  at::Tensor tensor2d;
  at::Tensor tensor1d;
};

// 测试 t()：2D tensor shape {2,3} -> {3,2}
TEST_F(TTest, T2DShape) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();

  at::Tensor result = tensor2d.t();
  file << std::to_string(result.dim()) << " ";
  file << std::to_string(result.sizes()[0]) << " ";
  file << std::to_string(result.sizes()[1]) << " ";
  file.saveFile();
}

// 测试 t() 数据正确性：result[i][j] == original[j][i]
TEST_F(TTest, T2DData) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();

  at::Tensor result = tensor2d.t();
  // tensor2d[0][0]=0, [0][1]=1, [0][2]=2
  // [1][0]=3, [1][1]=4, [1][2]=5
  // result[i][j] = tensor2d[j][i]
  file << std::to_string(result[0][0].item<float>()) << " ";  // 0
  file << std::to_string(result[0][1].item<float>()) << " ";  // 3
  file << std::to_string(result[1][0].item<float>()) << " ";  // 1
  file << std::to_string(result[2][0].item<float>()) << " ";  // 2
  file.saveFile();
}

// 测试 t() 对 1D tensor：返回原 tensor（shape 不变）
TEST_F(TTest, T1DNoChange) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();

  at::Tensor result = tensor1d.t();
  file << std::to_string(result.dim()) << " ";
  file << std::to_string(result.sizes()[0]) << " ";
  file << std::to_string(result.numel()) << " ";
  file.saveFile();
}

// 测试 t_()：原地转置，返回自身引用
TEST_F(TTest, TInplace) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();

  at::Tensor t = tensor2d.clone();
  at::Tensor& result = t.t_();
  // 返回的是同一对象
  file << std::to_string(&result == &t) << " ";
  file << std::to_string(t.sizes()[0]) << " ";
  file << std::to_string(t.sizes()[1]) << " ";
  file.saveFile();
}

// 测试 t() 与 transpose(0,1) 等价
TEST_F(TTest, TEquivTranspose) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();

  at::Tensor t_result = tensor2d.t();
  at::Tensor tr_result = tensor2d.transpose(0, 1);
  // 逐元素比较
  bool equal = at::equal(t_result.contiguous(), tr_result.contiguous());
  file << std::to_string(equal) << " ";
  file.saveFile();
}

}  // namespace test
}  // namespace at
