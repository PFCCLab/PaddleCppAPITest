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

class TransposeTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // 2x3 tensor, values 0..5
    tensor2d = at::zeros({2, 3}, at::kFloat);
    float* d = tensor2d.data_ptr<float>();
    for (int64_t i = 0; i < 6; ++i) d[i] = static_cast<float>(i);

    // 2x3x4 tensor, values 0..23
    tensor3d = at::zeros({2, 3, 4}, at::kFloat);
    float* d3 = tensor3d.data_ptr<float>();
    for (int64_t i = 0; i < 24; ++i) d3[i] = static_cast<float>(i);
  }

  at::Tensor tensor2d;
  at::Tensor tensor3d;
};

// 测试二维 transpose(0, 1)：shape {2,3} -> {3,2}
TEST_F(TransposeTest, Transpose2DBasic) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();

  at::Tensor result = tensor2d.transpose(0, 1);
  file << std::to_string(result.dim()) << " ";
  file << std::to_string(result.sizes()[0]) << " ";
  file << std::to_string(result.sizes()[1]) << " ";
  // 转置后 [0][0]=0, [1][0]=1, [2][0]=2
  file << std::to_string(result[0][0].item<float>()) << " ";
  file << std::to_string(result[1][0].item<float>()) << " ";
  file << std::to_string(result[2][0].item<float>()) << " ";
  file.saveFile();
}

// 测试 transpose 使用负数维度索引
TEST_F(TransposeTest, Transpose2DNegativeDims) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();

  at::Tensor result = tensor2d.transpose(-2, -1);
  file << std::to_string(result.sizes()[0]) << " ";
  file << std::to_string(result.sizes()[1]) << " ";
  file << std::to_string(result.numel()) << " ";
  file.saveFile();
}

// 测试 transpose 同一维度（恒等变换）
TEST_F(TransposeTest, TransposeSameDim) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();

  at::Tensor result = tensor2d.transpose(0, 0);
  file << std::to_string(result.sizes()[0]) << " ";
  file << std::to_string(result.sizes()[1]) << " ";
  file << std::to_string(result[0][0].item<float>()) << " ";
  file.saveFile();
}

// 测试三维 transpose(0, 2)：shape {2,3,4} -> {4,3,2}
TEST_F(TransposeTest, Transpose3D) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();

  at::Tensor result = tensor3d.transpose(0, 2);
  file << std::to_string(result.dim()) << " ";
  file << std::to_string(result.sizes()[0]) << " ";
  file << std::to_string(result.sizes()[1]) << " ";
  file << std::to_string(result.sizes()[2]) << " ";
  file.saveFile();
}

// 测试 transpose_ 原地转置（二维）
TEST_F(TransposeTest, TransposeInplace) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();

  at::Tensor& result = tensor2d.transpose_(0, 1);
  // 原地操作应返回同一对象
  file << std::to_string(&result == &tensor2d) << " ";
  file << std::to_string(tensor2d.sizes()[0]) << " ";
  file << std::to_string(tensor2d.sizes()[1]) << " ";
  file.saveFile();
}

// 测试 contiguous 之后 transpose_ 数据是否正确
TEST_F(TransposeTest, TransposeInplaceData) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();

  at::Tensor t = tensor2d.clone();
  t.transpose_(0, 1);
  // t[i][j] 应等于原 tensor2d[j][i]
  file << std::to_string(t[0][0].item<float>()) << " ";  // 原 [0][0]=0
  file << std::to_string(t[0][1].item<float>()) << " ";  // 原 [1][0]=3
  file << std::to_string(t[1][0].item<float>()) << " ";  // 原 [0][1]=1
  file.saveFile();
}

}  // namespace test
}  // namespace at
