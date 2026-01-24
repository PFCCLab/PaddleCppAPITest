#include <ATen/ATen.h>
#include <ATen/core/Tensor.h>
#include <ATen/ops/from_blob.h>
#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "../../src/file_manager.h"

extern paddle_api_test::ThreadSafeParam g_custom_param;

namespace at {
namespace test {

using paddle_api_test::FileManerger;
using paddle_api_test::ThreadSafeParam;

class FromBlobTest : public ::testing::Test {
 protected:
  void SetUp() override {
    data_buffer = new float[6];
    for (int i = 0; i < 6; ++i) {
      data_buffer[i] = static_cast<float>(i);
    }
  }

  void TearDown() override { delete[] data_buffer; }

  float* data_buffer;
};

TEST_F(FromBlobTest, FromBlobBasic) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();
  std::vector<int64_t> sizes = {2, 3};
  at::Tensor result = at::from_blob(data_buffer, sizes);
  file << std::to_string(result.dim()) << " ";
  file << std::to_string(result.sizes()[0]) << " ";
  file << std::to_string(result.sizes()[1]) << " ";
  file << std::to_string(result.data_ptr<float>() == data_buffer) << " ";
  float* data = result.data_ptr<float>();
  for (int i = 0; i < 6; ++i) {
    file << std::to_string(data[i]) << " ";
  }
  file.saveFile();
}

TEST_F(FromBlobTest, FromBlobWithOptions) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();
  std::vector<int64_t> sizes = {3, 2};
  at::Tensor result =
      at::from_blob(data_buffer, sizes, at::TensorOptions().dtype(at::kDouble));
  file << std::to_string(result.dim()) << " ";
  file << std::to_string(result.sizes()[0]) << " ";
  file << std::to_string(result.sizes()[1]) << " ";
  file << std::to_string(static_cast<int>(result.scalar_type())) << " ";
  file.saveFile();
}

TEST_F(FromBlobTest, FromBlobWithStrides) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();
  std::vector<int64_t> sizes = {2, 3};
  std::vector<int64_t> strides = {3, 1};  // Row-major
  at::Tensor result = at::from_blob(data_buffer, sizes, strides);
  file << std::to_string(result.dim()) << " ";
  file << std::to_string(result.strides()[0]) << " ";
  file << std::to_string(result.strides()[1]) << " ";
  file.saveFile();
}

TEST_F(FromBlobTest, FromBlob1D) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();
  std::vector<int64_t> sizes = {6};
  at::Tensor result = at::from_blob(data_buffer, sizes);
  file << std::to_string(result.dim()) << " ";
  file << std::to_string(result.numel()) << " ";
  file << std::to_string(result.data_ptr<float>() == data_buffer) << " ";
  file.saveFile();
}

TEST_F(FromBlobTest, FromBlobDifferentDataTypes) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();
  int* int_data = new int[4]{1, 2, 3, 4};
  std::vector<int64_t> sizes = {2, 2};
  at::Tensor result =
      at::from_blob(int_data, sizes, at::TensorOptions().dtype(at::kInt));
  file << std::to_string(static_cast<int>(result.scalar_type())) << " ";
  file << std::to_string(result.numel()) << " ";
  int* data = result.data_ptr<int>();
  for (int i = 0; i < 4; ++i) {
    file << std::to_string(data[i]) << " ";
  }
  file.saveFile();
  delete[] int_data;
}

}  // namespace test
}  // namespace at
