#include <ATen/ATen.h>
#include <ATen/core/Tensor.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/full.h>
#include <ATen/ops/ones.h>
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

class CreationOpsTest : public ::testing::Test {
 protected:
  void SetUp() override {}
};

TEST_F(CreationOpsTest, ZerosBasic) {
  std::vector<int64_t> shape = {2, 3};
  at::Tensor result = at::zeros(shape);
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();
  file << std::to_string(result.dim()) << " ";
  file << std::to_string(result.numel()) << " ";
  file << std::to_string(result.sizes()[0]) << " ";
  file << std::to_string(result.sizes()[1]) << " ";
  float* data = result.data_ptr<float>();
  for (int64_t i = 0; i < 6; ++i) {
    file << std::to_string(data[i]) << " ";
  }
  file.saveFile();
}

TEST_F(CreationOpsTest, ZerosWithOptions) {
  at::Tensor result = at::zeros({3, 4}, at::TensorOptions().dtype(at::kDouble));
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();
  file << std::to_string(result.dim()) << " ";
  file << std::to_string(result.numel()) << " ";
  file << std::to_string(static_cast<int>(result.scalar_type())) << " ";
  double* data = result.data_ptr<double>();
  for (int64_t i = 0; i < 12; ++i) {
    file << std::to_string(data[i]) << " ";
  }
  file.saveFile();
}

TEST_F(CreationOpsTest, OnesBasic) {
  at::Tensor result = at::ones({2, 2});
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();
  file << std::to_string(result.dim()) << " ";
  file << std::to_string(result.numel()) << " ";
  float* data = result.data_ptr<float>();
  for (int64_t i = 0; i < 4; ++i) {
    file << std::to_string(data[i]) << " ";
  }
  file.saveFile();
}

TEST_F(CreationOpsTest, OnesWithOptions) {
  at::Tensor result = at::ones({3}, at::TensorOptions().dtype(at::kInt));
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();
  file << std::to_string(result.dim()) << " ";
  file << std::to_string(result.numel()) << " ";
  file << std::to_string(static_cast<int>(result.scalar_type())) << " ";
  int* data = result.data_ptr<int>();
  for (int64_t i = 0; i < 3; ++i) {
    file << std::to_string(data[i]) << " ";
  }
  file.saveFile();
}

TEST_F(CreationOpsTest, EmptyBasic) {
  at::Tensor result = at::empty({2, 3});
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();
  file << std::to_string(result.dim()) << " ";
  file << std::to_string(result.numel()) << " ";
  file << std::to_string(result.sizes()[0]) << " ";
  file << std::to_string(result.sizes()[1]) << " ";
  file << std::to_string(result.data_ptr() != nullptr) << " ";
  file.saveFile();
}

TEST_F(CreationOpsTest, EmptyWithOptions) {
  at::Tensor result = at::empty({4}, at::TensorOptions().dtype(at::kFloat));
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();
  file << std::to_string(result.dim()) << " ";
  file << std::to_string(result.numel()) << " ";
  file << std::to_string(static_cast<int>(result.scalar_type())) << " ";
  file << std::to_string(result.data_ptr<float>() != nullptr) << " ";
  file.saveFile();
}

TEST_F(CreationOpsTest, FullBasic) {
  at::Tensor result = at::full({2, 2}, 5.0f);
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();
  file << std::to_string(result.dim()) << " ";
  file << std::to_string(result.numel()) << " ";
  float* data = result.data_ptr<float>();
  for (int64_t i = 0; i < 4; ++i) {
    file << std::to_string(data[i]) << " ";
  }
  file.saveFile();
}

TEST_F(CreationOpsTest, FullWithOptions) {
  at::Tensor result = at::full({3}, 10, at::TensorOptions().dtype(at::kLong));
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();
  file << std::to_string(result.dim()) << " ";
  file << std::to_string(result.numel()) << " ";
  file << std::to_string(static_cast<int>(result.scalar_type())) << " ";
  int64_t* data = result.data_ptr<int64_t>();
  for (int64_t i = 0; i < 3; ++i) {
    file << std::to_string(data[i]) << " ";
  }
  file.saveFile();
}

}  // namespace test
}  // namespace at
