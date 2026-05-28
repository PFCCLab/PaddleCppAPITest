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

class BitwiseTest : public ::testing::Test {
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

TEST_F(BitwiseTest, BitwiseRightShift) {
  FileManerger file(GetTestCaseResultFileName());
  file.createFile();
  file << "BitwiseRightShift ";
  at::Tensor input = at::ones({2, 3}, at::kInt).fill_(8);
  at::Tensor result = input.bitwise_right_shift(2);
  int* data = result.data_ptr<int>();
  file << std::to_string(data[0]) << " ";
  file << "\n";
  file.saveFile();
}

}  // namespace test
}  // namespace at
