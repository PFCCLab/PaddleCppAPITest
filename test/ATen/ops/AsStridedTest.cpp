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

class AsStridedTest : public ::testing::Test {
 protected:
  void SetUp() override {}
};

// 返回当前用例的结果文件名
std::string GetTestCaseResultFileName() {
  std::string base = g_custom_param.get();
  std::string test_name =
      ::testing::UnitTest::GetInstance()->current_test_info()->name();
  if (base.size() >= 4 && base.substr(base.size() - 4) == ".txt") {
    base.resize(base.size() - 4);
  }
  return base + "_" + test_name + ".txt";
}

TEST_F(AsStridedTest, AsStrided) {
  FileManerger file(GetTestCaseResultFileName());
  file.createFile();
  file << "AsStrided ";
  at::Tensor tensor = at::ones({2, 3, 4}, at::kFloat);
  at::Tensor strided = tensor.as_strided({3, 4, 2}, {2, 1, 6});
  file << std::to_string(strided.sizes()[0]) << " ";
  file << std::to_string(strided.sizes()[1]) << " ";
  file << std::to_string(strided.sizes()[2]) << " ";
  file << "\n";
  file.saveFile();
}

TEST_F(AsStridedTest, AsStridedInplace) {
  FileManerger file(GetTestCaseResultFileName());
  file.openAppend();
  file << "AsStridedInplace ";
  at::Tensor tensor = at::ones({2, 3, 4}, at::kFloat);
  tensor.as_strided_({3, 4, 2}, {2, 1, 6});
  file << std::to_string(tensor.sizes()[0]) << " ";
  file << std::to_string(tensor.sizes()[1]) << " ";
  file << std::to_string(tensor.sizes()[2]) << " ";
  file << "\n";
  file.saveFile();
}

TEST_F(AsStridedTest, AsStridedScatter) {
  FileManerger file(GetTestCaseResultFileName());
  file.openAppend();
  file << "AsStridedScatter ";
  at::Tensor tensor = at::ones({2, 3, 4}, at::kFloat);
  at::Tensor src = at::ones({3, 4, 2}, at::kFloat).fill_(2.0f);
  at::Tensor result = tensor.as_strided_scatter(src, {3, 4, 2}, {2, 1, 6});
  file << std::to_string(result.sizes()[0]) << " ";
  file << std::to_string(result.dim()) << " ";
  file << "\n";
  file.saveFile();
}

}  // namespace test
}  // namespace at
