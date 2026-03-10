#include <ATen/ATen.h>
#include <ATen/core/Tensor.h>
#include <ATen/ops/abs.h>
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

class AbsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    std::vector<int64_t> shape = {4};
    test_tensor = at::zeros(shape, at::kFloat);
    float* data = test_tensor.data_ptr<float>();
    data[0] = 1.0f;
    data[1] = -2.0f;
    data[2] = 0.0f;
    data[3] = -3.5f;
  }
  at::Tensor test_tensor;
};

static void write_abs_result_to_file(FileManerger* file,
                                     const at::Tensor& result) {
  *file << std::to_string(result.dim()) << " ";
  *file << std::to_string(result.numel()) << " ";
  float* result_data = result.data_ptr<float>();
  for (int64_t i = 0; i < result.numel(); ++i) {
    *file << std::to_string(result_data[i]) << " ";
  }
}

TEST_F(AbsTest, BasicAbs) {
  at::Tensor result = at::abs(test_tensor);
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();
  write_abs_result_to_file(&file, result);
  file.saveFile();
}

TEST_F(AbsTest, PositiveTensor) {
  at::Tensor positive_tensor = at::zeros({3}, at::kFloat);
  float* data = positive_tensor.data_ptr<float>();
  data[0] = 1.5f;
  data[1] = 3.0f;
  data[2] = 7.2f;

  at::Tensor result = at::abs(positive_tensor);
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  write_abs_result_to_file(&file, result);
  file.saveFile();
}

TEST_F(AbsTest, NegativeTensor) {
  at::Tensor negative_tensor = at::zeros({3}, at::kFloat);
  float* data = negative_tensor.data_ptr<float>();
  data[0] = -1.5f;
  data[1] = -3.0f;
  data[2] = -7.2f;

  at::Tensor result = at::abs(negative_tensor);
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  write_abs_result_to_file(&file, result);
  file.saveFile();
}

}  // namespace test
}  // namespace at
