#include <ATen/ATen.h>
#include <ATen/core/Tensor.h>
#include <ATen/ops/eye.h>
#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "../../src/file_manager.h"

extern paddle_api_test::ThreadSafeParam g_custom_param;

namespace at {
namespace test {

using paddle_api_test::FileManerger;
using paddle_api_test::ThreadSafeParam;

class EyeTest : public ::testing::Test {
 protected:
  void SetUp() override {}
};

static void write_eye_result_to_file(FileManerger* file,
                                     const at::Tensor& result) {
  *file << std::to_string(result.dim()) << " ";
  *file << std::to_string(result.numel()) << " ";
  for (int64_t i = 0; i < result.dim(); ++i) {
    *file << std::to_string(result.sizes()[i]) << " ";
  }
  *file << std::to_string(static_cast<int>(result.scalar_type())) << " ";
  float* data = result.data_ptr<float>();
  for (int64_t i = 0; i < result.numel(); ++i) {
    *file << std::to_string(data[i]) << " ";
  }
}

// 基本 3×3 单位矩阵（默认 float dtype）
TEST_F(EyeTest, BasicEyeSquare) {
  at::Tensor result = at::eye(3);
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();
  write_eye_result_to_file(&file, result);
  file.saveFile();
}

// 1×1 单位矩阵
TEST_F(EyeTest, EyeSingleElement) {
  at::Tensor result = at::eye(1);
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  write_eye_result_to_file(&file, result);
  file.saveFile();
}

// 指定 double dtype 的 4×4 单位矩阵
TEST_F(EyeTest, EyeWithDoubleDtype) {
  at::Tensor result = at::eye(4, at::TensorOptions().dtype(at::kDouble));
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << std::to_string(result.dim()) << " ";
  file << std::to_string(result.numel()) << " ";
  for (int64_t i = 0; i < result.dim(); ++i) {
    file << std::to_string(result.sizes()[i]) << " ";
  }
  file << std::to_string(static_cast<int>(result.scalar_type())) << " ";
  double* data = result.data_ptr<double>();
  for (int64_t i = 0; i < result.numel(); ++i) {
    file << std::to_string(data[i]) << " ";
  }
  file.saveFile();
}

// 行数 < 列数的矩形单位矩阵（3×5）
TEST_F(EyeTest, EyeRectangularMoreCols) {
  at::Tensor result = at::eye(3, 5);
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  write_eye_result_to_file(&file, result);
  file.saveFile();
}

// 行数 > 列数的矩形单位矩阵（5×3）
TEST_F(EyeTest, EyeRectangularMoreRows) {
  at::Tensor result = at::eye(5, 3);
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  write_eye_result_to_file(&file, result);
  file.saveFile();
}

// 指定 int dtype 的矩形单位矩阵（2×4）
TEST_F(EyeTest, EyeRectangularWithIntDtype) {
  at::Tensor result = at::eye(2, 4, at::TensorOptions().dtype(at::kInt));
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << std::to_string(result.dim()) << " ";
  file << std::to_string(result.numel()) << " ";
  for (int64_t i = 0; i < result.dim(); ++i) {
    file << std::to_string(result.sizes()[i]) << " ";
  }
  file << std::to_string(static_cast<int>(result.scalar_type())) << " ";
  int* data = result.data_ptr<int>();
  for (int64_t i = 0; i < result.numel(); ++i) {
    file << std::to_string(data[i]) << " ";
  }
  file.saveFile();
}

}  // namespace test
}  // namespace at
