#include <ATen/ATen.h>
#if USE_PADDLE_API
#include <ATen/indexing.h>
#else
#include <ATen/TensorIndexing.h>
#endif
#include <gtest/gtest.h>

#include <string>

#include "../src/file_manager.h"

extern paddle_api_test::ThreadSafeParam g_custom_param;

namespace at {
namespace test {

using paddle_api_test::FileManerger;
using paddle_api_test::ThreadSafeParam;

class IndexingTest : public ::testing::Test {
 protected:
  void SetUp() override {}
};

// EllipsisIndexType test
TEST_F(IndexingTest, EllipsisIndexType) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();

  // Construct EllipsisIndexType
  at::indexing::EllipsisIndexType ellipsis;
  (void)ellipsis;  // suppress unused variable warning
  file << "EllipsisIndexType ";
  file.saveFile();
}

// EllipsisIndexType with batch size - EllipsisIndexType only has default
// constructor
TEST_F(IndexingTest, EllipsisIndexTypeWithBatchSize) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();

  at::indexing::EllipsisIndexType ellipsis;
  (void)ellipsis;  // suppress unused variable warning
  file << "EllipsisIndexType_batch ";
  file.saveFile();
}

// Slice test - default constructor
TEST_F(IndexingTest, SliceDefault) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();

  at::indexing::Slice slice;
  (void)slice;  // suppress unused variable warning
  file << "Slice_default ";
  file.saveFile();
}

// Slice with values
TEST_F(IndexingTest, SliceWithValues) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();

  // Slice with start, end, step
  at::indexing::Slice slice(1, 10, 2);
  (void)slice;  // suppress unused variable warning
  file << "Slice_values ";
  file.saveFile();
}

// Test using indexing with tensors
TEST_F(IndexingTest, TensorIndexing) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();

  // Create a test tensor
  at::Tensor t = at::arange(12, at::kInt).view({3, 4});

  // 【API 差异】Paddle compat 的 Tensor::operator[] 仅重载 int64_t，不支持传入
  // Slice； 须改用 index(std::vector<at::indexing::Slice>) 接口。 PyTorch 支持
  // operator[](Slice) 及 index({Slice, ...}) 两种写法。
#if USE_PADDLE_API
  at::Tensor result =
      t.index(std::vector<at::indexing::Slice>{at::indexing::Slice()});
#else
  at::Tensor result = t.index({at::indexing::Slice()});
#endif
  file << std::to_string(result.dim()) << " ";
  file << std::to_string(result.numel()) << " ";
  file.saveFile();
}

// Test Slice indexing
TEST_F(IndexingTest, SliceIndexing) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();

  at::Tensor t = at::arange(12, at::kInt).view({3, 4});

  // 【API 差异】同上：Paddle 不支持链式 operator[](Slice)，
  // 多维 Slice 须放入同一个 std::vector<Slice> 传给 index()；
  // PyTorch 可用 index({Slice(0,2), Slice(1,3)}) 的 initializer_list 写法。
#if USE_PADDLE_API
  at::Tensor result = t.index(std::vector<at::indexing::Slice>{
      at::indexing::Slice(0, 2), at::indexing::Slice(1, 3)});
#else
  at::Tensor result =
      t.index({at::indexing::Slice(0, 2), at::indexing::Slice(1, 3)});
#endif
  file << std::to_string(result.dim()) << " ";
  file << std::to_string(result.size(0)) << " ";
  file << std::to_string(result.size(1)) << " ";
  file.saveFile();
}

}  // namespace test
}  // namespace at
