#include <ATen/ATen.h>
#include <ATen/core/Tensor.h>
#include <ATen/ops/arange.h>
#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "../../src/file_manager.h"

extern paddle_api_test::ThreadSafeParam g_custom_param;

namespace at {
namespace test {

using paddle_api_test::FileManerger;
using paddle_api_test::ThreadSafeParam;

class ArangeTest : public ::testing::Test {
 protected:
  void SetUp() override {}
};

TEST_F(ArangeTest, BasicArangeWithEnd) {
  at::Tensor result = at::arange(5, at::TensorOptions().dtype(at::kLong));
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();
  file << std::to_string(result.dim()) << " ";
  file << std::to_string(result.numel()) << " ";
  int64_t* data = result.data_ptr<int64_t>();
  for (int64_t i = 0; i < 5; ++i) {
    file << std::to_string(data[i]) << " ";
  }
  file.saveFile();
}

TEST_F(ArangeTest, ArangeWithStartEnd) {
  at::Tensor result = at::arange(2, 7, at::TensorOptions().dtype(at::kLong));
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();
  file << std::to_string(result.dim()) << " ";
  file << std::to_string(result.numel()) << " ";
  int64_t* data = result.data_ptr<int64_t>();
  for (int64_t i = 0; i < 5; ++i) {
    file << std::to_string(data[i]) << " ";
  }
  file.saveFile();
}

TEST_F(ArangeTest, ArangeWithStartEndStep) {
  at::Tensor result =
      at::arange(1, 10, 2, at::TensorOptions().dtype(at::kLong));
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();
  file << std::to_string(result.dim()) << " ";
  file << std::to_string(result.numel()) << " ";
  int64_t* data = result.data_ptr<int64_t>();
  for (int64_t i = 0; i < 5; ++i) {
    file << std::to_string(data[i]) << " ";
  }
  file.saveFile();
}

TEST_F(ArangeTest, ArangeWithOptions) {
  at::Tensor result = at::arange(4, at::TensorOptions().dtype(at::kFloat));
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();
  file << std::to_string(result.dim()) << " ";
  file << std::to_string(result.numel()) << " ";
  file << std::to_string(static_cast<int>(result.scalar_type())) << " ";
  float* data = result.data_ptr<float>();
  for (int64_t i = 0; i < 4; ++i) {
    file << std::to_string(data[i]) << " ";
  }
  file.saveFile();
}

TEST_F(ArangeTest, NegativeValues) {
  at::Tensor result = at::arange(-3, 3, at::TensorOptions().dtype(at::kLong));
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();
  file << std::to_string(result.dim()) << " ";
  file << std::to_string(result.numel()) << " ";
  int64_t* data = result.data_ptr<int64_t>();
  for (int64_t i = 0; i < 6; ++i) {
    file << std::to_string(data[i]) << " ";
  }
  file.saveFile();
}

}  // namespace test
}  // namespace at
