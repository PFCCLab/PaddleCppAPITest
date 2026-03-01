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

class ViewAsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // 源 tensor：shape {2, 3}，numel=6
    src = at::zeros({2, 3}, at::kFloat);
    float* d = src.data_ptr<float>();
    for (int64_t i = 0; i < 6; ++i) d[i] = static_cast<float>(i + 1);
  }

  at::Tensor src;
};

// 测试 view_as 相同 shape
TEST_F(ViewAsTest, ViewAsSameShape) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();

  at::Tensor other = at::zeros({2, 3}, at::kFloat);
  at::Tensor result = src.view_as(other);
  file << std::to_string(result.dim()) << " ";
  file << std::to_string(result.sizes()[0]) << " ";
  file << std::to_string(result.sizes()[1]) << " ";
  file << std::to_string(result.numel()) << " ";
  file.saveFile();
}

// 测试 view_as：{2,3} -> {6}（降维）
TEST_F(ViewAsTest, ViewAs2DTo1D) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();

  at::Tensor other = at::zeros({6}, at::kFloat);
  at::Tensor result = src.view_as(other);
  file << std::to_string(result.dim()) << " ";
  file << std::to_string(result.sizes()[0]) << " ";
  // 数据保持不变
  float* data = result.data_ptr<float>();
  file << std::to_string(data[0]) << " ";
  file << std::to_string(data[5]) << " ";
  file.saveFile();
}

// 测试 view_as：{2,3} -> {1,2,3}（升维）
TEST_F(ViewAsTest, ViewAs2DTo3D) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();

  at::Tensor other = at::zeros({1, 2, 3}, at::kFloat);
  at::Tensor result = src.view_as(other);
  file << std::to_string(result.dim()) << " ";
  file << std::to_string(result.sizes()[0]) << " ";
  file << std::to_string(result.sizes()[1]) << " ";
  file << std::to_string(result.sizes()[2]) << " ";
  file.saveFile();
}

// 测试 view_as：{2,3} -> {3,2}（转置形状，相同 numel）
TEST_F(ViewAsTest, ViewAs2DReshape) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();

  at::Tensor other = at::zeros({3, 2}, at::kFloat);
  at::Tensor result = src.view_as(other);
  file << std::to_string(result.sizes()[0]) << " ";
  file << std::to_string(result.sizes()[1]) << " ";
  // view 共享存储，修改 result 会影响 src
  result.fill_(0.f);
  file << std::to_string(src[0][0].item<float>()) << " ";
  file.saveFile();
}

// 测试 view_as 共享底层存储（非 copy）
TEST_F(ViewAsTest, ViewAsSharesStorage) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();

  at::Tensor other = at::zeros({6}, at::kFloat);
  at::Tensor result = src.view_as(other);
  // 修改 result 的第一个元素
  result[0] = 99.f;
  // src 的对应位置应同步更新
  file << std::to_string(src[0][0].item<float>()) << " ";
  file.saveFile();
}

}  // namespace test
}  // namespace at
