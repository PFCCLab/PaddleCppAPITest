#include <ATen/ATen.h>
#include <ATen/core/Tensor.h>
#include <ATen/ops/resize.h>
#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "src/file_manager.h"

extern paddle_api_test::ThreadSafeParam g_custom_param;

namespace at {
namespace test {

using paddle_api_test::FileManerger;

class ResizeTest : public ::testing::Test {
 protected:
  void SetUp() override {}
};

// 测试 resize_ - 缩小元素数时应成功并保留前缀数据
TEST_F(ResizeTest, Resize) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();
  file << "Resize ";
  at::Tensor tensor = at::ones({2, 3, 4}, at::kFloat);
  tensor.resize_({4, 5});
  file << std::to_string(tensor.sizes()[0]) << " ";
  file << std::to_string(tensor.sizes()[1]) << " ";
  file << std::to_string(tensor.numel()) << " ";
  file << std::to_string(tensor.data_ptr<float>()[0]) << " ";
  file << std::to_string(tensor.data_ptr<float>()[19]) << " ";
  file << "\n";
  file.saveFile();
}

TEST_F(ResizeTest, ResizeGrowStorage) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "ResizeGrowStorage ";

  at::Tensor grow = at::ones({2, 2}, at::kFloat);
  grow.data_ptr<float>()[0] = 11.0f;
  grow.data_ptr<float>()[3] = 44.0f;
  grow.resize_({4, 4});

  file << std::to_string(grow.sizes()[0]) << " ";
  file << std::to_string(grow.sizes()[1]) << " ";
  file << std::to_string(grow.numel()) << " ";
  file << std::to_string(grow.data_ptr<float>()[0]) << " ";
  file << "\n";
  file.saveFile();
}

TEST_F(ResizeTest, ResizeGrowStorageNbytes) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "ResizeGrowStorageNbytes ";

  at::Tensor grow = at::ones({2}, at::kInt);
  const size_t before_bytes = grow.storage().nbytes();

  grow.resize_({4});

  const size_t after_bytes = grow.storage().nbytes();
  const size_t expected_bytes = 4 * sizeof(int32_t);
  const bool shape_ok = grow.numel() == 4;
  const bool storage_ok = after_bytes >= expected_bytes;

  file << std::to_string(before_bytes >= 2 * sizeof(int32_t) ? 1 : 0) << " ";
  file << std::to_string(shape_ok ? 1 : 0) << " ";
  file << std::to_string(storage_ok ? 1 : 0) << " ";
  file << "\n";
  file.saveFile();

  EXPECT_EQ(grow.numel(), 4);
  EXPECT_GE(after_bytes, expected_bytes);
}

}  // namespace test
}  // namespace at
