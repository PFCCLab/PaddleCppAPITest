#include <ATen/ATen.h>
#include <ATen/core/Tensor.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/reshape.h>
#include <ATen/ops/zeros.h>
#include <ATen/ops/zeros_like.h>
#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "../../src/file_manager.h"

extern paddle_api_test::ThreadSafeParam g_custom_param;

namespace at {
namespace test {

using paddle_api_test::FileManerger;
using paddle_api_test::ThreadSafeParam;

class ReshapeTest : public ::testing::Test {
 protected:
  void SetUp() override {
    original_tensor = at::zeros({2, 3}, at::kFloat);
    float* data = original_tensor.data_ptr<float>();
    for (int64_t i = 0; i < 6; ++i) {
      data[i] = static_cast<float>(i);
    }
  }
  at::Tensor original_tensor;
};

TEST_F(ReshapeTest, Reshape2DTo1D) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();
  at::Tensor result = at::reshape(original_tensor, {6});
  file << std::to_string(result.dim()) << " ";
  file << std::to_string(result.numel()) << " ";
  file << std::to_string(result.sizes()[0]) << " ";
  float* data = result.data_ptr<float>();
  for (int64_t i = 0; i < 6; ++i) {
    file << std::to_string(data[i]) << " ";
  }
  file.saveFile();
}

TEST_F(ReshapeTest, Reshape2DTo3D) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();
  at::Tensor result = at::reshape(original_tensor, {1, 2, 3});
  file << std::to_string(result.dim()) << " ";
  file << std::to_string(result.numel()) << " ";
  file << std::to_string(result.sizes()[0]) << " ";
  file << std::to_string(result.sizes()[1]) << " ";
  file << std::to_string(result.sizes()[2]) << " ";
  float* data = result.data_ptr<float>();
  for (int64_t i = 0; i < 6; ++i) {
    file << std::to_string(data[i]) << " ";
  }
  file.saveFile();
}

TEST_F(ReshapeTest, ReshapeAutoInferDim) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();
  at::Tensor result = at::reshape(original_tensor, {-1});
  file << std::to_string(result.dim()) << " ";
  file << std::to_string(result.numel()) << " ";
  file << std::to_string(result.sizes()[0]) << " ";
  file.saveFile();
}

TEST_F(ReshapeTest, ReshapeInferOneDim) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();
  at::Tensor result = at::reshape(original_tensor, {3, -1});
  file << std::to_string(result.dim()) << " ";
  file << std::to_string(result.sizes()[0]) << " ";
  file << std::to_string(result.sizes()[1]) << " ";
  file.saveFile();
}

TEST_F(ReshapeTest, EmptyLike) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();
  at::Tensor result = at::empty_like(original_tensor);
  file << std::to_string(result.dim()) << " ";
  file << std::to_string(result.sizes()[0]) << " ";
  file << std::to_string(result.sizes()[1]) << " ";
  file << std::to_string(static_cast<int>(result.scalar_type())) << " ";
  file << std::to_string(result.data_ptr() != nullptr) << " ";
  file.saveFile();
}

TEST_F(ReshapeTest, ZerosLike) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();
  at::Tensor result = at::zeros_like(original_tensor);
  file << std::to_string(result.dim()) << " ";
  file << std::to_string(result.sizes()[0]) << " ";
  file << std::to_string(result.sizes()[1]) << " ";
  file << std::to_string(static_cast<int>(result.scalar_type())) << " ";
  float* data = result.data_ptr<float>();
  for (int64_t i = 0; i < 6; ++i) {
    file << std::to_string(data[i]) << " ";
  }
  file.saveFile();
}

TEST_F(ReshapeTest, EmptyLikeWithOptions) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();
  at::Tensor result =
      at::empty_like(original_tensor, at::TensorOptions().dtype(at::kDouble));
  file << std::to_string(result.dim()) << " ";
  file << std::to_string(result.sizes()[0]) << " ";
  file << std::to_string(result.sizes()[1]) << " ";
  file << std::to_string(static_cast<int>(result.scalar_type())) << " ";
  file.saveFile();
}

TEST_F(ReshapeTest, ZerosLikeWithOptions) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();
  at::Tensor result =
      at::zeros_like(original_tensor, at::TensorOptions().dtype(at::kInt));
  file << std::to_string(result.dim()) << " ";
  file << std::to_string(result.sizes()[0]) << " ";
  file << std::to_string(result.sizes()[1]) << " ";
  file << std::to_string(static_cast<int>(result.scalar_type())) << " ";
  int* data = result.data_ptr<int>();
  for (int64_t i = 0; i < 6; ++i) {
    file << std::to_string(data[i]) << " ";
  }
  file.saveFile();
}

}  // namespace test
}  // namespace at
