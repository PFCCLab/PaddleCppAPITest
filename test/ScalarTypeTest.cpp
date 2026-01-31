#include <ATen/ATen.h>
#include <ATen/core/Tensor.h>
#include <ATen/ops/ones.h>
#include <gtest/gtest.h>
#include <torch/all.h>

#include <string>
#include <vector>

#include "../src/file_manager.h"

extern paddle_api_test::ThreadSafeParam g_custom_param;

namespace at {
namespace test {

using paddle_api_test::FileManerger;
using paddle_api_test::ThreadSafeParam;

class ScalarTypeTest : public ::testing::Test {
 protected:
  void SetUp() override {
    std::vector<int64_t> shape = {2, 3, 4};
    tensor = at::ones(shape, at::kFloat);
  }

  at::Tensor tensor;
};

// 测试 is_complex
TEST_F(ScalarTypeTest, IsComplex) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();

  // Float tensor should not be complex
  file << std::to_string(tensor.is_complex()) << " ";

  // Test with actual complex tensor
  at::Tensor complex_tensor =
      at::ones({2, 3}, at::TensorOptions().dtype(at::kComplexFloat));
  file << std::to_string(complex_tensor.is_complex()) << " ";

  at::Tensor complex_double_tensor =
      at::ones({2, 3}, at::TensorOptions().dtype(at::kComplexDouble));
  file << std::to_string(complex_double_tensor.is_complex()) << " ";
  file.saveFile();
}

// 测试 is_floating_point
TEST_F(ScalarTypeTest, IsFloatingPoint) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();

  // Float tensor should be floating point
  file << std::to_string(tensor.is_floating_point()) << " ";

  // Test with double tensor
  at::Tensor double_tensor =
      at::ones({2, 3}, at::TensorOptions().dtype(at::kDouble));
  file << std::to_string(double_tensor.is_floating_point()) << " ";

  // Test with integer tensor
  at::Tensor int_tensor = at::ones({2, 3}, at::TensorOptions().dtype(at::kInt));
  file << std::to_string(int_tensor.is_floating_point()) << " ";

  // Test with long tensor
  at::Tensor long_tensor =
      at::ones({2, 3}, at::TensorOptions().dtype(at::kLong));
  file << std::to_string(long_tensor.is_floating_point()) << " ";
  file.saveFile();
}

// 测试 is_signed
TEST_F(ScalarTypeTest, IsSigned) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();

  // Float tensor should be signed
  file << std::to_string(tensor.is_signed()) << " ";

  // Test with int tensor (signed)
  at::Tensor int_tensor = at::ones({2, 3}, at::TensorOptions().dtype(at::kInt));
  file << std::to_string(int_tensor.is_signed()) << " ";

  // Test with long tensor (signed)
  at::Tensor long_tensor =
      at::ones({2, 3}, at::TensorOptions().dtype(at::kLong));
  file << std::to_string(long_tensor.is_signed()) << " ";

  // Test with byte tensor (unsigned)
  at::Tensor byte_tensor =
      at::ones({2, 3}, at::TensorOptions().dtype(at::kByte));
  file << std::to_string(byte_tensor.is_signed()) << " ";

  // Test with bool tensor (unsigned)
  at::Tensor bool_tensor =
      at::ones({2, 3}, at::TensorOptions().dtype(at::kBool));
  file << std::to_string(bool_tensor.is_signed()) << " ";
  file.saveFile();
}

}  // namespace test
}  // namespace at
