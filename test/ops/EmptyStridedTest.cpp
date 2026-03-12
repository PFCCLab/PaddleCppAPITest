#include <ATen/ATen.h>
#include <ATen/core/Tensor.h>
#include <ATen/ops/empty_strided.h>
#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "../../src/file_manager.h"

extern paddle_api_test::ThreadSafeParam g_custom_param;

namespace at {
namespace test {

using paddle_api_test::FileManerger;
using paddle_api_test::ThreadSafeParam;

class EmptyStridedTest : public ::testing::Test {
 protected:
  void SetUp() override {}
};

static void write_empty_strided_result_to_file(FileManerger* file,
                                               const at::Tensor& result) {
  *file << std::to_string(result.dim()) << " ";
  *file << std::to_string(result.numel()) << " ";
  for (int64_t i = 0; i < result.dim(); ++i) {
    *file << std::to_string(result.sizes()[i]) << " ";
  }
  for (int64_t i = 0; i < result.dim(); ++i) {
    *file << std::to_string(result.strides()[i]) << " ";
  }
  *file << std::to_string(result.data_ptr() != nullptr) << " ";
}

// 基本 row-major stride：{2,3} stride {3,1}
TEST_F(EmptyStridedTest, BasicRowMajor) {
  at::Tensor result = at::empty_strided({2, 3}, {3, 1});
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();
  write_empty_strided_result_to_file(&file, result);
  file.saveFile();
}

// column-major stride：{2,3} stride {1,2}
TEST_F(EmptyStridedTest, ColumnMajor) {
  at::Tensor result = at::empty_strided({2, 3}, {1, 2});
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  write_empty_strided_result_to_file(&file, result);
  file.saveFile();
}

// 指定 kFloat
TEST_F(EmptyStridedTest, WithDtypeFloat) {
  at::Tensor result =
      at::empty_strided({3, 4}, {4, 1}, at::TensorOptions().dtype(at::kFloat));
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  write_empty_strided_result_to_file(&file, result);
  file << std::to_string(static_cast<int>(result.scalar_type())) << " ";
  file.saveFile();
}

// 指定 kDouble
TEST_F(EmptyStridedTest, WithDtypeDouble) {
  at::Tensor result =
      at::empty_strided({3, 4}, {4, 1}, at::TensorOptions().dtype(at::kDouble));
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  write_empty_strided_result_to_file(&file, result);
  file << std::to_string(static_cast<int>(result.scalar_type())) << " ";
  file.saveFile();
}

// 指定 kInt
TEST_F(EmptyStridedTest, WithDtypeInt) {
  at::Tensor result =
      at::empty_strided({3, 4}, {4, 1}, at::TensorOptions().dtype(at::kInt));
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  write_empty_strided_result_to_file(&file, result);
  file << std::to_string(static_cast<int>(result.scalar_type())) << " ";
  file.saveFile();
}

// 指定 kLong
TEST_F(EmptyStridedTest, WithDtypeLong) {
  at::Tensor result =
      at::empty_strided({3, 4}, {4, 1}, at::TensorOptions().dtype(at::kLong));
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  write_empty_strided_result_to_file(&file, result);
  file << std::to_string(static_cast<int>(result.scalar_type())) << " ";
  file.saveFile();
}

// 1D tensor
TEST_F(EmptyStridedTest, OneDimensional) {
  at::Tensor result = at::empty_strided({10}, {1});
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  write_empty_strided_result_to_file(&file, result);
  file.saveFile();
}

// 4D tensor
TEST_F(EmptyStridedTest, HighDim) {
  // C-contiguous {2,3,4,5} stride {60,20,5,1}
  at::Tensor result = at::empty_strided({2, 3, 4, 5}, {60, 20, 5, 1});
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  write_empty_strided_result_to_file(&file, result);
  file.saveFile();
}

// 验证 stride 正确性
TEST_F(EmptyStridedTest, VerifyStrides) {
  std::vector<int64_t> sizes = {3, 4, 5};
  std::vector<int64_t> strides = {20, 5, 1};
  at::Tensor result = at::empty_strided(sizes, strides);
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  // 验证 strides 与请求一致
  bool strides_match = true;
  for (int64_t i = 0; i < result.dim(); ++i) {
    if (result.strides()[i] != strides[i]) {
      strides_match = false;
      break;
    }
  }
  file << std::to_string(strides_match) << " ";
  write_empty_strided_result_to_file(&file, result);
  file.saveFile();
}

// 大 shape
TEST_F(EmptyStridedTest, LargeShape) {
  at::Tensor result = at::empty_strided({100, 100}, {100, 1});
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  write_empty_strided_result_to_file(&file, result);
  file.saveFile();
}

}  // namespace test
}  // namespace at
