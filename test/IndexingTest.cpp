#include <ATen/ATen.h>
#include <ATen/TensorIndexing.h>
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
  at::Tensor result = t.index({at::indexing::Slice()});
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
  at::Tensor result =
      t.index({at::indexing::Slice(0, 2), at::indexing::Slice(1, 3)});
  file << std::to_string(result.dim()) << " ";
  file << std::to_string(result.size(0)) << " ";
  file << std::to_string(result.size(1)) << " ";
  file.saveFile();
}

}  // namespace test
}  // namespace at
