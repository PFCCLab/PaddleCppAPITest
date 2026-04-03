#include <ATen/ATen.h>
#include <ATen/TensorIndexing.h>
#include <gtest/gtest.h>

#include <string>

#include "src/file_manager.h"

extern paddle_api_test::ThreadSafeParam g_custom_param;

namespace at {
namespace test {

using paddle_api_test::FileManerger;
using paddle_api_test::ThreadSafeParam;

class IndexingTest : public ::testing::Test {
 protected:
  void SetUp() override {}
};

// 当前基线下，at::indexing 的 Slice / TensorIndex 路径已与 PyTorch 对齐；
// 本文件保留行为回归测试，避免历史差异重新引入。

// EllipsisIndexType test
TEST_F(IndexingTest, EllipsisIndexType) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();

  // Construct EllipsisIndexType
  at::indexing::EllipsisIndexType ellipsis;
  (void)ellipsis;  // suppress unused variable warning
  file << "EllipsisIndexType ";
  file << "\n";
  file.saveFile();
}

// EllipsisIndexType with batch size - EllipsisIndexType only has default
// constructor
TEST_F(IndexingTest, EllipsisIndexTypeWithBatchSize) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "EllipsisIndexTypeWithBatchSize ";

  at::indexing::EllipsisIndexType ellipsis;
  (void)ellipsis;  // suppress unused variable warning
  file << "EllipsisIndexType_batch ";
  file << "\n";
  file.saveFile();
}

// Slice test - default constructor
TEST_F(IndexingTest, SliceDefault) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "SliceDefault ";

  at::indexing::Slice slice;
  (void)slice;  // suppress unused variable warning
  file << "Slice_default ";
  file << "\n";
  file.saveFile();
}

// Slice with values
TEST_F(IndexingTest, SliceWithValues) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "SliceWithValues ";

  // Slice with start, end, step
  at::indexing::Slice slice(1, 10, 2);
  (void)slice;  // suppress unused variable warning
  file << "Slice_values ";
  file << "\n";
  file.saveFile();
}

// Test using indexing with tensors
TEST_F(IndexingTest, TensorIndexing) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "TensorIndexing ";

  // Create a test tensor
  at::Tensor t = at::arange(12, at::kInt).view({3, 4});

  at::Tensor result = t.index({at::indexing::Slice()});
  file << std::to_string(result.dim()) << " ";
  file << std::to_string(result.numel()) << " ";
  file << "\n";
  file.saveFile();
}

// Test Slice indexing
TEST_F(IndexingTest, SliceIndexing) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "SliceIndexing ";

  at::Tensor t = at::arange(12, at::kInt).view({3, 4});

  at::Tensor result =
      t.index({at::indexing::Slice(0, 2), at::indexing::Slice(1, 3)});
  file << std::to_string(result.dim()) << " ";
  file << std::to_string(result.size(0)) << " ";
  file << std::to_string(result.size(1)) << " ";
  file << "\n";
  file.saveFile();
}

// TensorIndexNone - Test None type index
TEST_F(IndexingTest, TensorIndexNone) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "TensorIndexNone ";

  at::indexing::TensorIndex idx(at::indexing::None);
  file << std::to_string(idx.is_none()) << " ";
  file << std::to_string(idx.is_ellipsis()) << " ";
  file << std::to_string(idx.is_integer()) << " ";
  file << std::to_string(idx.is_boolean()) << " ";
  file << std::to_string(idx.is_slice()) << " ";
  file << std::to_string(idx.is_tensor()) << " ";
  file << "\n";
  file.saveFile();
}

// TensorIndexEllipsis - Test Ellipsis type index
TEST_F(IndexingTest, TensorIndexEllipsis) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "TensorIndexEllipsis ";

  at::indexing::TensorIndex idx(at::indexing::Ellipsis);
  file << std::to_string(idx.is_none()) << " ";
  file << std::to_string(idx.is_ellipsis()) << " ";
  file << std::to_string(idx.is_integer()) << " ";
  file << std::to_string(idx.is_boolean()) << " ";
  file << std::to_string(idx.is_slice()) << " ";
  file << std::to_string(idx.is_tensor()) << " ";
  file << "\n";
  file.saveFile();
}

// TensorIndexEllipsisString - Test "..." string construction
TEST_F(IndexingTest, TensorIndexEllipsisString) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "TensorIndexEllipsisString ";

  at::indexing::TensorIndex idx("...");
  file << std::to_string(idx.is_none()) << " ";
  file << std::to_string(idx.is_ellipsis()) << " ";
  file << std::to_string(idx.is_integer()) << " ";
  file << std::to_string(idx.is_boolean()) << " ";
  file << std::to_string(idx.is_slice()) << " ";
  file << std::to_string(idx.is_tensor()) << " ";
  file << "\n";
  file.saveFile();
}

// Helper function to convert integer to string.
// 当前两端都走 SymInt，但保留对历史 int64_t 形态的兼容，便于回归比对。
template <typename T,
          typename std::enable_if_t<std::is_same_v<T, int64_t>, int> = 0>
std::string integer_to_string(T val) {
  return std::to_string(val);
}

template <typename T,
          typename std::enable_if_t<!std::is_same_v<T, int64_t>, int> = 0>
std::string integer_to_string(T val) {
  auto maybe_int = val.maybe_as_int();
  if (maybe_int.has_value()) {
    return std::to_string(maybe_int.value());
  }
  return "0";
}

// TensorIndexIntegerPos - Test positive integer index
TEST_F(IndexingTest, TensorIndexIntegerPos) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "TensorIndexIntegerPos ";

  at::indexing::TensorIndex idx(static_cast<int64_t>(5));
  auto int_val = idx.integer();
  file << std::to_string(idx.is_none()) << " ";
  file << std::to_string(idx.is_ellipsis()) << " ";
  file << std::to_string(idx.is_integer()) << " ";
  file << integer_to_string(int_val) << " ";
  file << std::to_string(idx.is_boolean()) << " ";
  file << std::to_string(idx.is_slice()) << " ";
  file << std::to_string(idx.is_tensor()) << " ";
  file << "\n";
  file.saveFile();
}

// TensorIndexIntegerNeg - Test negative integer index
TEST_F(IndexingTest, TensorIndexIntegerNeg) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "TensorIndexIntegerNeg ";

  at::indexing::TensorIndex idx(static_cast<int64_t>(-3));
  auto int_val = idx.integer();
  file << std::to_string(idx.is_none()) << " ";
  file << std::to_string(idx.is_ellipsis()) << " ";
  file << std::to_string(idx.is_integer()) << " ";
  file << integer_to_string(int_val) << " ";
  file << std::to_string(idx.is_boolean()) << " ";
  file << std::to_string(idx.is_slice()) << " ";
  file << std::to_string(idx.is_tensor()) << " ";
  file << "\n";
  file.saveFile();
}

// TensorIndexIntegerZero - Test zero integer index
TEST_F(IndexingTest, TensorIndexIntegerZero) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "TensorIndexIntegerZero ";

  at::indexing::TensorIndex idx(static_cast<int64_t>(0));
  auto int_val = idx.integer();
  file << std::to_string(idx.is_none()) << " ";
  file << std::to_string(idx.is_ellipsis()) << " ";
  file << std::to_string(idx.is_integer()) << " ";
  file << integer_to_string(int_val) << " ";
  file << std::to_string(idx.is_boolean()) << " ";
  file << std::to_string(idx.is_slice()) << " ";
  file << std::to_string(idx.is_tensor()) << " ";
  file << "\n";
  file.saveFile();
}

// TensorIndexBooleanTrue - Test true boolean index
TEST_F(IndexingTest, TensorIndexBooleanTrue) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "TensorIndexBooleanTrue ";

  at::indexing::TensorIndex idx(true);
  file << std::to_string(idx.is_none()) << " ";
  file << std::to_string(idx.is_ellipsis()) << " ";
  file << std::to_string(idx.is_integer()) << " ";
  file << std::to_string(idx.is_boolean()) << " ";
  file << std::to_string(idx.boolean()) << " ";
  file << std::to_string(idx.is_slice()) << " ";
  file << std::to_string(idx.is_tensor()) << " ";
  file << "\n";
  file.saveFile();
}

// TensorIndexBooleanFalse - Test false boolean index
TEST_F(IndexingTest, TensorIndexBooleanFalse) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "TensorIndexBooleanFalse ";

  at::indexing::TensorIndex idx(false);
  file << std::to_string(idx.is_none()) << " ";
  file << std::to_string(idx.is_ellipsis()) << " ";
  file << std::to_string(idx.is_integer()) << " ";
  file << std::to_string(idx.is_boolean()) << " ";
  file << std::to_string(idx.boolean()) << " ";
  file << std::to_string(idx.is_slice()) << " ";
  file << std::to_string(idx.is_tensor()) << " ";
  file << "\n";
  file.saveFile();
}

// TensorIndexSliceDefault - Test default Slice
TEST_F(IndexingTest, TensorIndexSliceDefault) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "TensorIndexSliceDefault ";

  at::indexing::Slice slice;
  at::indexing::TensorIndex idx(slice);
  file << std::to_string(idx.is_none()) << " ";
  file << std::to_string(idx.is_ellipsis()) << " ";
  file << std::to_string(idx.is_integer()) << " ";
  file << std::to_string(idx.is_boolean()) << " ";
  file << std::to_string(idx.is_slice()) << " ";
  file << std::to_string(idx.is_tensor()) << " ";
  file << "\n";
  file.saveFile();
}

// TensorIndexSliceWithValues - Test Slice with values
TEST_F(IndexingTest, TensorIndexSliceWithValues) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "TensorIndexSliceWithValues ";

  at::indexing::TensorIndex idx(at::indexing::Slice(1, 10, 2));
  file << std::to_string(idx.is_none()) << " ";
  file << std::to_string(idx.is_ellipsis()) << " ";
  file << std::to_string(idx.is_integer()) << " ";
  file << std::to_string(idx.is_boolean()) << " ";
  file << std::to_string(idx.is_slice()) << " ";
  file << std::to_string(idx.is_tensor()) << " ";
  file << "\n";
  file.saveFile();
}

// TensorIndexSliceNegative - Test negative value Slice
TEST_F(IndexingTest, TensorIndexSliceNegative) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "TensorIndexSliceNegative ";

  at::indexing::TensorIndex idx(at::indexing::Slice(-5, -1, 2));
  file << std::to_string(idx.is_none()) << " ";
  file << std::to_string(idx.is_ellipsis()) << " ";
  file << std::to_string(idx.is_integer()) << " ";
  file << std::to_string(idx.is_boolean()) << " ";
  file << std::to_string(idx.is_slice()) << " ";
  file << std::to_string(idx.is_tensor()) << " ";
  file << "\n";
  file.saveFile();
}

// TensorIndexTensor - Test Tensor type index
TEST_F(IndexingTest, TensorIndexTensor) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "TensorIndexTensor ";

  at::Tensor tensor_idx = at::arange(3, at::kInt);
  at::indexing::TensorIndex idx(tensor_idx);
  file << std::to_string(idx.is_none()) << " ";
  file << std::to_string(idx.is_ellipsis()) << " ";
  file << std::to_string(idx.is_integer()) << " ";
  file << std::to_string(idx.is_boolean()) << " ";
  file << std::to_string(idx.is_slice()) << " ";
  file << std::to_string(idx.is_tensor()) << " ";
  file << "\n";
  file.saveFile();
}

}  // namespace test
}  // namespace at
