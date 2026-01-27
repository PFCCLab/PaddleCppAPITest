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

class NarrowTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // 创建一个 4x5x6 的 tensor
    tensor = at::zeros({4, 5, 6}, at::kFloat);
    float* data = tensor.data_ptr<float>();
    for (int64_t i = 0; i < 120; ++i) {
      data[i] = static_cast<float>(i);
    }
  }

  at::Tensor tensor;
};

// 测试 narrow 在 dim 0
TEST_F(NarrowTest, NarrowDim0) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();

  // narrow(dim=0, start=1, length=2): shape {4, 5, 6} -> {2, 5, 6}
  at::Tensor result = tensor.narrow(0, 1, 2);
  file << std::to_string(result.dim()) << " ";
  file << std::to_string(result.sizes()[0]) << " ";
  file << std::to_string(result.sizes()[1]) << " ";
  file << std::to_string(result.sizes()[2]) << " ";
  file << std::to_string(result.numel()) << " ";

  // 验证第一个元素值 (应该是原 tensor 的 index [1, 0, 0])
  float* data = result.data_ptr<float>();
  file << std::to_string(data[0]) << " ";
  file.saveFile();
}

// 测试 narrow 在 dim 1
TEST_F(NarrowTest, NarrowDim1) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();

  // narrow(dim=1, start=2, length=3): shape {4, 5, 6} -> {4, 3, 6}
  at::Tensor result = tensor.narrow(1, 2, 3);
  file << std::to_string(result.dim()) << " ";
  file << std::to_string(result.sizes()[0]) << " ";
  file << std::to_string(result.sizes()[1]) << " ";
  file << std::to_string(result.sizes()[2]) << " ";
  file << std::to_string(result.numel()) << " ";
  file.saveFile();
}

// 测试 narrow 在 dim 2
TEST_F(NarrowTest, NarrowDim2) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();

  // narrow(dim=2, start=0, length=4): shape {4, 5, 6} -> {4, 5, 4}
  at::Tensor result = tensor.narrow(2, 0, 4);
  file << std::to_string(result.dim()) << " ";
  file << std::to_string(result.sizes()[0]) << " ";
  file << std::to_string(result.sizes()[1]) << " ";
  file << std::to_string(result.sizes()[2]) << " ";
  file << std::to_string(result.numel()) << " ";
  file.saveFile();
}

// 测试 narrow_symint
TEST_F(NarrowTest, NarrowSymint) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();

  c10::SymInt start(1);
  c10::SymInt length(2);
  at::Tensor result = tensor.narrow_symint(0, start, length);

  file << std::to_string(result.dim()) << " ";
  file << std::to_string(result.sizes()[0]) << " ";
  file << std::to_string(result.sizes()[1]) << " ";
  file << std::to_string(result.sizes()[2]) << " ";
  file.saveFile();
}

// 测试 narrow_copy
TEST_F(NarrowTest, NarrowCopy) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();

  at::Tensor result = tensor.narrow_copy(0, 1, 2);
  file << std::to_string(result.dim()) << " ";
  file << std::to_string(result.sizes()[0]) << " ";
  file << std::to_string(result.sizes()[1]) << " ";
  file << std::to_string(result.sizes()[2]) << " ";

  // narrow_copy 返回的是拷贝，验证数据独立性
  float* result_data = result.data_ptr<float>();
  result_data[0] = -999.0f;

  // 原 tensor 不应该被修改
  float* src_data = tensor.data_ptr<float>();
  file << std::to_string(src_data[30] != -999.0f) << " ";  // index [1,0,0] = 30
  file.saveFile();
}

// 测试 narrow_copy_symint
TEST_F(NarrowTest, NarrowCopySymint) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();

  c10::SymInt start(0);
  c10::SymInt length(3);
  at::Tensor result = tensor.narrow_copy_symint(0, start, length);

  file << std::to_string(result.dim()) << " ";
  file << std::to_string(result.sizes()[0]) << " ";
  file << std::to_string(result.sizes()[1]) << " ";
  file << std::to_string(result.sizes()[2]) << " ";
  file.saveFile();
}

// 测试 narrow 使用 Tensor 作为 start
TEST_F(NarrowTest, NarrowWithTensorStart) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();

  // 创建一个标量 tensor 作为 start (0-dim tensor)
  at::Tensor start_tensor = at::zeros({}, at::kLong);
  int64_t* start_data = start_tensor.data_ptr<int64_t>();
  start_data[0] = 2;

  at::Tensor result = tensor.narrow(0, start_tensor, 2);
  file << std::to_string(result.dim()) << " ";
  file << std::to_string(result.sizes()[0]) << " ";
  file << std::to_string(result.sizes()[1]) << " ";
  file << std::to_string(result.sizes()[2]) << " ";

  // 验证第一个元素值 (应该是原 tensor 的 index [2, 0, 0])
  float* data = result.data_ptr<float>();
  file << std::to_string(data[0]) << " ";
  file.saveFile();
}

// 测试 narrow_symint 使用 Tensor 作为 start
TEST_F(NarrowTest, NarrowSymintWithTensorStart) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();

  at::Tensor start_tensor = at::zeros({}, at::kLong);
  int64_t* start_data = start_tensor.data_ptr<int64_t>();
  start_data[0] = 1;

  c10::SymInt length(2);
  at::Tensor result = tensor.narrow_symint(1, start_tensor, length);

  file << std::to_string(result.dim()) << " ";
  file << std::to_string(result.sizes()[0]) << " ";
  file << std::to_string(result.sizes()[1]) << " ";
  file << std::to_string(result.sizes()[2]) << " ";
  file.saveFile();
}

// 测试多次 narrow 操作
TEST_F(NarrowTest, MultipleNarrow) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();

  // 连续 narrow: {4, 5, 6} -> {2, 5, 6} -> {2, 3, 6} -> {2, 3, 4}
  at::Tensor result = tensor.narrow(0, 1, 2).narrow(1, 1, 3).narrow(2, 1, 4);
  file << std::to_string(result.dim()) << " ";
  file << std::to_string(result.sizes()[0]) << " ";
  file << std::to_string(result.sizes()[1]) << " ";
  file << std::to_string(result.sizes()[2]) << " ";
  file << std::to_string(result.numel()) << " ";
  file.saveFile();
}

}  // namespace test
}  // namespace at
