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

class FlattenTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // 创建一个 2x3x4 的 tensor
    tensor = at::ones({2, 3, 4}, at::kFloat);
    float* data = tensor.data_ptr<float>();
    for (int64_t i = 0; i < 24; ++i) {
      data[i] = static_cast<float>(i);
    }
  }

  at::Tensor tensor;
};

// 测试 flatten 默认参数 (start_dim=0, end_dim=-1)
TEST_F(FlattenTest, FlattenDefault) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();

  at::Tensor result = tensor.flatten(0, -1);
  file << std::to_string(result.dim()) << " ";
  file << std::to_string(result.numel()) << " ";
  file << std::to_string(result.sizes()[0]) << " ";

  // 验证数据
  float* data = result.data_ptr<float>();
  file << std::to_string(data[0]) << " ";
  file << std::to_string(data[23]) << " ";
  file.saveFile();
}

// 测试 flatten 指定 start_dim 和 end_dim
TEST_F(FlattenTest, FlattenWithDims) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();

  // flatten dim 1 and 2: shape {2, 3, 4} -> {2, 12}
  at::Tensor result = tensor.flatten(1, 2);
  file << std::to_string(result.dim()) << " ";
  file << std::to_string(result.sizes()[0]) << " ";
  file << std::to_string(result.sizes()[1]) << " ";
  file.saveFile();
}

// 测试 flatten 使用负数索引
TEST_F(FlattenTest, FlattenNegativeDims) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();

  // flatten from dim -2 to -1: shape {2, 3, 4} -> {2, 12}
  at::Tensor result = tensor.flatten(-2, -1);
  file << std::to_string(result.dim()) << " ";
  file << std::to_string(result.sizes()[0]) << " ";
  file << std::to_string(result.sizes()[1]) << " ";
  file.saveFile();
}

// 测试 flatten 从 dim 0 开始
TEST_F(FlattenTest, FlattenFromStart) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();

  // flatten dim 0 and 1: shape {2, 3, 4} -> {6, 4}
  at::Tensor result = tensor.flatten(0, 1);
  file << std::to_string(result.dim()) << " ";
  file << std::to_string(result.sizes()[0]) << " ";
  file << std::to_string(result.sizes()[1]) << " ";
  file.saveFile();
}

// 测试 unflatten
TEST_F(FlattenTest, Unflatten) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();

  // 先 flatten 成 {2, 12}，然后 unflatten 回 {2, 3, 4}
  at::Tensor flattened = tensor.flatten(1, 2);
  at::Tensor result = flattened.unflatten(1, {3, 4});

  file << std::to_string(result.dim()) << " ";
  file << std::to_string(result.sizes()[0]) << " ";
  file << std::to_string(result.sizes()[1]) << " ";
  file << std::to_string(result.sizes()[2]) << " ";
  file << std::to_string(result.numel()) << " ";
  file.saveFile();
}

// 测试 unflatten 使用负数维度
TEST_F(FlattenTest, UnflattenNegativeDim) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();

  // 创建一个 {6, 4} 的 tensor
  at::Tensor flat_tensor = at::ones({6, 4}, at::kFloat);
  // unflatten dim -2 (即 dim 0) 成 {2, 3}
  at::Tensor result = flat_tensor.unflatten(-2, {2, 3});

  file << std::to_string(result.dim()) << " ";
  file << std::to_string(result.sizes()[0]) << " ";
  file << std::to_string(result.sizes()[1]) << " ";
  file << std::to_string(result.sizes()[2]) << " ";
  file.saveFile();
}

// 测试 unflatten_symint
TEST_F(FlattenTest, UnflattenSymint) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();

  at::Tensor flattened = tensor.flatten(1, 2);
  c10::SymIntArrayRef sizes({3, 4});
  at::Tensor result = flattened.unflatten_symint(1, sizes);

  file << std::to_string(result.dim()) << " ";
  file << std::to_string(result.sizes()[0]) << " ";
  file << std::to_string(result.sizes()[1]) << " ";
  file << std::to_string(result.sizes()[2]) << " ";
  file.saveFile();
}

// 测试 flatten 后数据保持正确
TEST_F(FlattenTest, FlattenDataIntegrity) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();

  at::Tensor result = tensor.flatten(0, -1);
  float* src_data = tensor.data_ptr<float>();
  float* dst_data = result.data_ptr<float>();

  // 检查数据是否一致
  bool data_equal = true;
  for (int64_t i = 0; i < 24; ++i) {
    if (src_data[i] != dst_data[i]) {
      data_equal = false;
      break;
    }
  }
  file << std::to_string(data_equal) << " ";
  file.saveFile();
}

}  // namespace test
}  // namespace at
