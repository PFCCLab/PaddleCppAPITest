#include <ATen/ATen.h>
#include <ATen/core/Tensor.h>
#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "src/file_manager.h"

extern paddle_api_test::ThreadSafeParam g_custom_param;

namespace at {
namespace test {

using paddle_api_test::FileManerger;

class ExpandTest : public ::testing::Test {
 protected:
  void SetUp() override {}
};

// Compute element offset from flat index using strides (strides-aware access)
static inline int64_t compute_offset_from_flat_index(int64_t flat_idx,
                                                     const at::Tensor& tensor) {
  int64_t offset = 0;
  int64_t remainder = flat_idx;
  for (int64_t d = tensor.dim() - 1; d >= 0; --d) {
    int64_t coord = remainder % tensor.sizes()[d];
    remainder /= tensor.sizes()[d];
    offset += coord * tensor.strides()[d];
  }
  return offset;
}

// Write tensor metadata (dim, numel, sizes, strides) and all element values.
// Uses strides-aware access to faithfully reflect the underlying layout.
// If Paddle and PyTorch produce different strides, result_cmp will DIFFER,
// and the difference should be recorded as a known mismatch.
static void write_expand_result_to_file(FileManerger* file,
                                        const at::Tensor& result) {
  *file << std::to_string(result.dim()) << " ";
  *file << std::to_string(result.numel()) << " ";
  for (int64_t i = 0; i < result.dim(); ++i) {
    *file << std::to_string(result.sizes()[i]) << " ";
  }
  // Record strides so layout differences are detected by result_cmp
  for (int64_t i = 0; i < result.dim(); ++i) {
    *file << std::to_string(result.strides()[i]) << " ";
  }
  if (result.numel() == 0) {
    *file << "empty ";
    return;
  }
  float* data = result.data_ptr<float>();
  for (int64_t i = 0; i < result.numel(); ++i) {
    int64_t offset = compute_offset_from_flat_index(i, result);
    *file << std::to_string(data[offset]) << " ";
  }
}

TEST_F(ExpandTest, Expand) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();
  file << "Expand ";
  at::Tensor small = at::ones({1, 3}, at::kFloat);
  at::Tensor expanded = small.expand({4, 3});
  file << std::to_string(expanded.sizes()[0]) << " ";
  file << std::to_string(expanded.sizes()[1]) << " ";
  file << "\n";
  file.saveFile();
}

TEST_F(ExpandTest, ExpandAs) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "ExpandAs ";
  at::Tensor small = at::ones({1, 3}, at::kFloat);
  at::Tensor target = at::ones({4, 3}, at::kFloat);
  at::Tensor expanded = small.expand_as(target);
  file << std::to_string(expanded.sizes()[0]) << " ";
  file << std::to_string(expanded.sizes()[1]) << " ";
  file << "\n";
  file.saveFile();
}

TEST_F(ExpandTest, ExpandRankLessCanUseExpand) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "ExpandRankLessCanUseExpand ";

  at::Tensor small = at::ones({1}, at::kFloat);
  at::Tensor expanded = small.expand({2, 3});
  write_expand_result_to_file(&file, expanded);

  file << "\n";
  file.saveFile();
}

TEST_F(ExpandTest, ExpandRankLessFallbackGrowTarget) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "ExpandRankLessFallbackGrowTarget ";

  at::Tensor small = at::ones({1, 2}, at::kFloat);
  at::Tensor expanded = small.expand({2, 3, 2});
  write_expand_result_to_file(&file, expanded);

  file << "\n";
  file.saveFile();
}

TEST_F(ExpandTest, ExpandRankLessFallbackShrinkTarget) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "ExpandRankLessFallbackShrinkTarget ";

  at::Tensor small = at::ones({1, 2}, at::kFloat);
  at::Tensor expanded = small.expand({2, 1, 2});
  write_expand_result_to_file(&file, expanded);

  file << "\n";
  file.saveFile();
}

TEST_F(ExpandTest, ExpandRankLessFallbackZeroSize) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "ExpandRankLessFallbackZeroSize ";

  at::Tensor small = at::ones({1, 0}, at::kFloat);
  at::Tensor expanded = small.expand({2, 3, 0});
  file << std::to_string(expanded.dim()) << " ";
  file << std::to_string(expanded.numel()) << " ";
  file << "empty ";

  file << "\n";
  file.saveFile();
}

TEST_F(ExpandTest, ExpandSameRankFallbackShrink) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "ExpandSameRankFallbackShrink ";

  try {
    at::Tensor small = at::ones({2, 3}, at::kFloat);
    at::Tensor expanded = small.expand({2, 2});
    write_expand_result_to_file(&file, expanded);
  } catch (const std::exception&) {
    file << "exception ";
  }

  file << "\n";
  file.saveFile();
}

TEST_F(ExpandTest, ExpandInputRankGreaterThanTargetRank) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "ExpandInputRankGreaterThanTargetRank ";

  try {
    at::Tensor small = at::ones({1, 2, 3}, at::kFloat);
    at::Tensor expanded = small.expand({2, 3});
    write_expand_result_to_file(&file, expanded);
  } catch (const std::exception&) {
    file << "exception ";
  }

  file << "\n";
  file.saveFile();
}

// Scalar expand (0-d tensor)
TEST_F(ExpandTest, ExpandScalar) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "ExpandScalar ";

  at::Tensor t = at::full({}, 5.0f, at::kFloat);
  at::Tensor result = t.expand({2, 3});
  write_expand_result_to_file(&file, result);

  file << "\n";
  file.saveFile();
}

// expand supports -1 (keep original size)
TEST_F(ExpandTest, ExpandNegativeOne) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "ExpandNegativeOne ";

  at::Tensor t = at::ones({3}, at::kFloat);
  at::Tensor result = t.expand({-1});
  write_expand_result_to_file(&file, result);

  file << "\n";
  file.saveFile();
}

// -1 in leading non-existing dimension throws
TEST_F(ExpandTest, ExpandNegativeOneLeadingError) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "ExpandNegativeOneLeadingError ";

  try {
    at::Tensor t = at::ones({3}, at::kFloat);
    at::Tensor result = t.expand({-1, 4});
    write_expand_result_to_file(&file, result);
  } catch (const std::exception&) {
    file << "exception ";
  }

  file << "\n";
  file.saveFile();
}

}  // namespace test
}  // namespace at
