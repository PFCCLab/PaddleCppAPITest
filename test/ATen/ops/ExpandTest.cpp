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

static void write_expand_result_to_file(FileManerger* file,
                                        const at::Tensor& result) {
  *file << std::to_string(result.dim()) << " ";
  *file << std::to_string(result.numel()) << " ";
  for (int64_t i = 0; i < result.dim(); ++i) {
    *file << std::to_string(result.sizes()[i]) << " ";
  }
  if (result.numel() == 0) {
    *file << "empty ";
    return;
  }
  at::Tensor cont = result.contiguous();
  float* data = cont.data_ptr<float>();
  *file << std::to_string(data[0]) << " ";
  *file << std::to_string(data[cont.numel() - 1]) << " ";
  *file << std::to_string(cont.sum().item<float>()) << " ";
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

}  // namespace test
}  // namespace at
