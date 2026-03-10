#include <ATen/ATen.h>
#include <ATen/core/Tensor.h>
#include <ATen/ops/sum.h>
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

class SumTest : public ::testing::Test {
 protected:
  void SetUp() override {
    std::vector<int64_t> shape = {2, 3};
    test_tensor = at::zeros(shape, at::kFloat);
    float* data = test_tensor.data_ptr<float>();
    for (int64_t i = 0; i < 6; ++i) {
      data[i] = static_cast<float>(i + 1);
    }
  }
  at::Tensor test_tensor;
};

TEST_F(SumTest, SumAllElements) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();
  at::Tensor result = at::sum(test_tensor);
  file << std::to_string(result.dim()) << " ";
  file << std::to_string(result.numel()) << " ";
  float result_value = *result.data_ptr<float>();
  file << std::to_string(result_value) << " ";
  file.saveFile();
}

TEST_F(SumTest, SumWithDtype) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();
  at::Tensor result = at::sum(test_tensor, at::kDouble);
  file << std::to_string(result.dim()) << " ";
  file << std::to_string(static_cast<int>(result.scalar_type())) << " ";
  double result_value = *result.data_ptr<double>();
  file << std::to_string(result_value) << " ";
  file.saveFile();
}

TEST_F(SumTest, SumAlongDim0) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();
  at::Tensor result = at::sum(test_tensor, {0}, false);
  file << std::to_string(result.dim()) << " ";
  file << std::to_string(result.numel()) << " ";
  float* data = result.data_ptr<float>();
  file << std::to_string(data[0]) << " ";
  file << std::to_string(data[1]) << " ";
  file << std::to_string(data[2]) << " ";
  file.saveFile();
}

TEST_F(SumTest, SumAlongDim1) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();
  at::Tensor result = at::sum(test_tensor, {1}, false);
  file << std::to_string(result.dim()) << " ";
  file << std::to_string(result.numel()) << " ";
  float* data = result.data_ptr<float>();
  file << std::to_string(data[0]) << " ";
  file << std::to_string(data[1]) << " ";
  file.saveFile();
}

TEST_F(SumTest, SumWithKeepdim) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();
  at::Tensor result = at::sum(test_tensor, {0}, true);
  file << std::to_string(result.dim()) << " ";
  file << std::to_string(result.numel()) << " ";
  file << std::to_string(result.sizes()[0]) << " ";
  file << std::to_string(result.sizes()[1]) << " ";
  float* data = result.data_ptr<float>();
  file << std::to_string(data[0]) << " ";
  file << std::to_string(data[1]) << " ";
  file << std::to_string(data[2]) << " ";
  file.saveFile();
}

TEST_F(SumTest, SumOutFunction) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();
  at::Tensor output = at::zeros({}, at::kFloat);
  at::Tensor& result = at::sum_out(output, test_tensor);
  file << std::to_string(&result == &output) << " ";
  float result_value = *output.data_ptr<float>();
  file << std::to_string(result_value) << " ";
  file.saveFile();
}

}  // namespace test
}  // namespace at
