#include <ATen/ATen.h>
#include <ATen/core/Tensor.h>
#include <ATen/ops/cat.h>
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

class ConnectionOpsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    tensor1 = at::zeros({2, 3}, at::kFloat);
    tensor2 = at::zeros({2, 3}, at::kFloat);

    float* data1 = tensor1.data_ptr<float>();
    float* data2 = tensor2.data_ptr<float>();
    for (int64_t i = 0; i < 6; ++i) {
      data1[i] = static_cast<float>(i);
      data2[i] = static_cast<float>(i + 6);
    }
  }

  at::Tensor tensor1;
  at::Tensor tensor2;
};

TEST_F(ConnectionOpsTest, CatDim0) {
  std::vector<at::Tensor> tensors = {tensor1, tensor2};
  at::Tensor result = at::cat(tensors, 0);
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();
  file << std::to_string(result.dim()) << " ";
  file << std::to_string(result.sizes()[0]) << " ";
  file << std::to_string(result.sizes()[1]) << " ";
  file << std::to_string(result.numel()) << " ";
  float* data = result.data_ptr<float>();
  for (int64_t i = 0; i < 12; ++i) {
    file << std::to_string(data[i]) << " ";
  }
  file.saveFile();
}

TEST_F(ConnectionOpsTest, CatDim1) {
  std::vector<at::Tensor> tensors = {tensor1, tensor2};
  at::Tensor result = at::cat(tensors, 1);
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();
  file << std::to_string(result.dim()) << " ";
  file << std::to_string(result.sizes()[0]) << " ";
  file << std::to_string(result.sizes()[1]) << " ";
  file << std::to_string(result.numel()) << " ";
  float* data = result.data_ptr<float>();
  for (int64_t i = 0; i < 12; ++i) {
    file << std::to_string(data[i]) << " ";
  }
  file.saveFile();
}

TEST_F(ConnectionOpsTest, CatThreeTensors) {
  at::Tensor tensor3 = at::zeros({2, 3}, at::kFloat);
  float* data3 = tensor3.data_ptr<float>();
  for (int64_t i = 0; i < 6; ++i) {
    data3[i] = static_cast<float>(i + 12);
  }

  std::vector<at::Tensor> tensors = {tensor1, tensor2, tensor3};
  at::Tensor result = at::cat(tensors, 0);
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();
  file << std::to_string(result.sizes()[0]) << " ";
  file << std::to_string(result.sizes()[1]) << " ";
  file << std::to_string(result.numel()) << " ";
  file.saveFile();
}

TEST_F(ConnectionOpsTest, CatWithDifferentTypes) {
  at::Tensor int_tensor = at::zeros({1, 2}, at::kInt);
  at::Tensor float_tensor = at::zeros({1, 2}, at::kInt);

  std::vector<at::Tensor> tensors = {int_tensor, float_tensor};
  at::Tensor result = at::cat(tensors, 0);
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();
  file << std::to_string(static_cast<int>(result.scalar_type())) << " ";
  file << std::to_string(result.sizes()[0]) << " ";
  file << std::to_string(result.sizes()[1]) << " ";
  file.saveFile();
}

}  // namespace test
}  // namespace at
