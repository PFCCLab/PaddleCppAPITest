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

class ClampTest : public ::testing::Test {
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

TEST_F(ClampTest, ClampScalarMinMax) {
  FileManerger file(GetTestCaseResultFileName());
  file.createFile();
  file << "ClampScalarMinMax ";
  at::Tensor input = at::ones({2, 3}, at::kFloat).fill_(5.0f);
  at::Tensor clamped = input.clamp(1.0, 3.0);
  file << std::to_string(clamped.dim()) << " ";
  float* data = clamped.data_ptr<float>();
  file << std::to_string(data[0]) << " ";
  file << "\n";
  file.saveFile();
}

TEST_F(ClampTest, ClampTensorMinMax) {
  FileManerger file(GetTestCaseResultFileName());
  file.openAppend();
  file << "ClampTensorMinMax ";
  at::Tensor input = at::ones({2, 3}, at::kFloat).fill_(5.0f);
  at::Tensor min_tensor = at::ones({1}, at::kFloat).fill_(1.0f);
  at::Tensor max_tensor = at::ones({1}, at::kFloat).fill_(3.0f);
  at::Tensor clamped = input.clamp(min_tensor, max_tensor);
  file << std::to_string(clamped.dim()) << " ";
  file << "\n";
  file.saveFile();
}

TEST_F(ClampTest, ClampInplaceScalar) {
  FileManerger file(GetTestCaseResultFileName());
  file.openAppend();
  file << "ClampInplaceScalar ";
  at::Tensor input = at::ones({2, 3}, at::kFloat).fill_(5.0f);
  input.clamp_(1.0, 3.0);
  float* data = input.data_ptr<float>();
  file << std::to_string(data[0]) << " ";
  file << "\n";
  file.saveFile();
}

TEST_F(ClampTest, ClampInplaceTensor) {
  FileManerger file(GetTestCaseResultFileName());
  file.openAppend();
  file << "ClampInplaceTensor ";
  at::Tensor input = at::ones({2, 3}, at::kFloat).fill_(5.0f);
  at::Tensor min_tensor = at::ones({1}, at::kFloat).fill_(1.0f);
  at::Tensor max_tensor = at::ones({1}, at::kFloat).fill_(3.0f);
  input.clamp_(min_tensor, max_tensor);
  file << std::to_string(input.dim()) << " ";
  file << "\n";
  file.saveFile();
}

TEST_F(ClampTest, ClampMaxScalar) {
  FileManerger file(GetTestCaseResultFileName());
  file.openAppend();
  file << "ClampMaxScalar ";
  at::Tensor input = at::ones({2, 3}, at::kFloat).fill_(5.0f);
  at::Tensor clamped = input.clamp_max(3.0);
  float* data = clamped.data_ptr<float>();
  file << std::to_string(data[0]) << " ";
  file << "\n";
  file.saveFile();
}

TEST_F(ClampTest, ClampMaxTensor) {
  FileManerger file(GetTestCaseResultFileName());
  file.openAppend();
  file << "ClampMaxTensor ";
  at::Tensor input = at::ones({2, 3}, at::kFloat).fill_(5.0f);
  at::Tensor max_tensor = at::ones({1}, at::kFloat).fill_(3.0f);
  at::Tensor clamped = input.clamp_max(max_tensor);
  file << std::to_string(clamped.dim()) << " ";
  file << "\n";
  file.saveFile();
}

TEST_F(ClampTest, ClampMaxInplace) {
  FileManerger file(GetTestCaseResultFileName());
  file.openAppend();
  file << "ClampMaxInplace ";
  at::Tensor input = at::ones({2, 3}, at::kFloat).fill_(5.0f);
  input.clamp_max_(3.0);
  float* data = input.data_ptr<float>();
  file << std::to_string(data[0]) << " ";
  file << "\n";
  file.saveFile();
}

TEST_F(ClampTest, ClampMaxInplaceTensor) {
  FileManerger file(GetTestCaseResultFileName());
  file.openAppend();
  file << "ClampMaxInplaceTensor ";
  at::Tensor input = at::ones({2, 3}, at::kFloat).fill_(5.0f);
  at::Tensor max_tensor = at::ones({1}, at::kFloat).fill_(3.0f);
  input.clamp_max_(max_tensor);
  file << std::to_string(input.dim()) << " ";
  file << "\n";
  file.saveFile();
}

TEST_F(ClampTest, ClampMinScalar) {
  FileManerger file(GetTestCaseResultFileName());
  file.openAppend();
  file << "ClampMinScalar ";
  at::Tensor input = at::ones({2, 3}, at::kFloat);
  at::Tensor clamped = input.clamp_min(2.0);
  float* data = clamped.data_ptr<float>();
  file << std::to_string(data[0]) << " ";
  file << "\n";
  file.saveFile();
}

TEST_F(ClampTest, ClampMinTensor) {
  FileManerger file(GetTestCaseResultFileName());
  file.openAppend();
  file << "ClampMinTensor ";
  at::Tensor input = at::ones({2, 3}, at::kFloat);
  at::Tensor min_tensor = at::ones({1}, at::kFloat).fill_(2.0f);
  at::Tensor clamped = input.clamp_min(min_tensor);
  file << std::to_string(clamped.dim()) << " ";
  file << "\n";
  file.saveFile();
}

TEST_F(ClampTest, ClampMinInplace) {
  FileManerger file(GetTestCaseResultFileName());
  file.openAppend();
  file << "ClampMinInplace ";
  at::Tensor input = at::ones({2, 3}, at::kFloat);
  input.clamp_min_(2.0);
  float* data = input.data_ptr<float>();
  file << std::to_string(data[0]) << " ";
  file << "\n";
  file.saveFile();
}

TEST_F(ClampTest, ClampMinInplaceTensor) {
  FileManerger file(GetTestCaseResultFileName());
  file.openAppend();
  file << "ClampMinInplaceTensor ";
  at::Tensor input = at::ones({2, 3}, at::kFloat);
  at::Tensor min_tensor = at::ones({1}, at::kFloat).fill_(2.0f);
  input.clamp_min_(min_tensor);
  file << std::to_string(input.dim()) << " ";
  file << "\n";
  file.saveFile();
}

}  // namespace test
}  // namespace at
