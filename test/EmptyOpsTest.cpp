#include <ATen/ATen.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/full.h>
#include <ATen/ops/ones.h>
#include <ATen/ops/reshape.h>
#include <ATen/ops/zeros.h>
#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "../src/file_manager.h"

extern paddle_api_test::ThreadSafeParam g_custom_param;

namespace at {
namespace test {

using paddle_api_test::FileManerger;
using paddle_api_test::ThreadSafeParam;

class EmptyOpsTest : public ::testing::Test {
 protected:
  void SetUp() override {}
};

// empty
TEST_F(EmptyOpsTest, Empty) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();

  // Test empty with IntArrayRef size
  at::Tensor t = at::empty({2, 3, 4}, at::kFloat);
  file << std::to_string(t.dim()) << " ";
  file << std::to_string(t.numel()) << " ";
  file << std::to_string(t.size(0)) << " ";
  file << std::to_string(t.size(1)) << " ";
  file << std::to_string(t.size(2)) << " ";
  file << std::to_string(static_cast<int>(t.scalar_type())) << " ";
  file.saveFile();
}

// empty with different dtype
TEST_F(EmptyOpsTest, EmptyDifferentDtype) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();

  at::Tensor t = at::empty({3, 4}, at::kInt);
  file << std::to_string(t.dim()) << " ";
  file << std::to_string(t.numel()) << " ";
  file << std::to_string(static_cast<int>(t.scalar_type())) << " ";
  file.saveFile();
}

// empty with ScalarType
TEST_F(EmptyOpsTest, EmptyWithScalarType) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();

  // Using at::kFloat equivalent to c10::ScalarType::Float
  at::Tensor t = at::empty({2, 3}, at::TensorOptions().dtype(at::kFloat));
  file << std::to_string(t.dim()) << " ";
  file << std::to_string(static_cast<int>(t.scalar_type())) << " ";
  file.saveFile();
}

// empty_cuda (if CUDA available)
TEST_F(EmptyOpsTest, EmptyCUDA) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();

  // Try to create empty CUDA tensor
  try {
    at::Tensor t = at::empty({2, 3}, at::TensorOptions().device(at::kCUDA));
    file << "cuda_tensor ";
  } catch (...) {
    file << "cuda_not_available ";
  }
  file.saveFile();
}

// full_symint
TEST_F(EmptyOpsTest, FullSymint) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();

  // full_symint with SymIntArrayRef
  at::Tensor t = at::full_symint({2, 3}, 5.0f, at::kFloat);
  file << std::to_string(t.dim()) << " ";
  file << std::to_string(t.numel()) << " ";
  file << std::to_string(t.data_ptr<float>()[0]) << " ";
  file << std::to_string(t.data_ptr<float>()[5]) << " ";
  file.saveFile();
}

// full_symint with int
TEST_F(EmptyOpsTest, FullSymintInt) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();

  // full_symint with array size
  at::Tensor t = at::full_symint({10}, 3, at::kInt);
  file << std::to_string(t.dim()) << " ";
  file << std::to_string(t.numel()) << " ";
  file << std::to_string(t.data_ptr<int>()[0]) << " ";
  file << std::to_string(t.data_ptr<int>()[9]) << " ";
  file.saveFile();
}

// ones_symint
TEST_F(EmptyOpsTest, OnesSymint) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();

  // ones_symint with array
  at::Tensor t = at::ones_symint({5}, at::kFloat);
  file << std::to_string(t.dim()) << " ";
  file << std::to_string(t.numel()) << " ";
  file << std::to_string(t.data_ptr<float>()[0]) << " ";
  file << std::to_string(t.data_ptr<float>()[4]) << " ";
  file.saveFile();
}

// ones_symint with shape
TEST_F(EmptyOpsTest, OnesSymintShape) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();

  // ones_symint with c10::SymIntArrayRef
  at::Tensor t = at::ones_symint({2, 3, 4}, at::kFloat);
  file << std::to_string(t.dim()) << " ";
  file << std::to_string(t.numel()) << " ";
  file << std::to_string(t.data_ptr<float>()[0]) << " ";
  file << std::to_string(t.data_ptr<float>()[23]) << " ";
  file.saveFile();
}

// zeros_symint
TEST_F(EmptyOpsTest, ZerosSymint) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();

  // zeros_symint with array
  at::Tensor t = at::zeros_symint({5}, at::kFloat);
  file << std::to_string(t.dim()) << " ";
  file << std::to_string(t.numel()) << " ";
  file << std::to_string(t.data_ptr<float>()[0]) << " ";
  file << std::to_string(t.data_ptr<float>()[4]) << " ";
  file.saveFile();
}

// zeros_symint with shape
TEST_F(EmptyOpsTest, ZerosSymintShape) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();

  // zeros_symint with c10::SymIntArrayRef
  at::Tensor t = at::zeros_symint({2, 3}, at::kFloat);
  file << std::to_string(t.dim()) << " ";
  file << std::to_string(t.numel()) << " ";
  file << std::to_string(t.data_ptr<float>()[0]) << " ";
  file << std::to_string(t.data_ptr<float>()[5]) << " ";
  file.saveFile();
}

// reshape_symint
TEST_F(EmptyOpsTest, ReshapeSymint) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();

  // Create a tensor and reshape
  at::Tensor t = at::arange(12, at::kInt);
  at::Tensor reshaped = at::reshape_symint(t, {3, 4});
  file << std::to_string(reshaped.dim()) << " ";
  file << std::to_string(reshaped.size(0)) << " ";
  file << std::to_string(reshaped.size(1)) << " ";
  file.saveFile();
}

// reshape_symint with int
TEST_F(EmptyOpsTest, ReshapeSymintInt) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();

  at::Tensor t = at::arange(24, at::kInt);
  at::Tensor reshaped = at::reshape_symint(t, {24});
  file << std::to_string(reshaped.dim()) << " ";
  file << std::to_string(reshaped.numel()) << " ";
  file.saveFile();
}

// Test empty with different sizes
TEST_F(EmptyOpsTest, EmptySizes) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();

  // Scalar (0-d)
  at::Tensor t0 = at::empty({}, at::kFloat);
  file << "scalar " << std::to_string(t0.dim()) << " ";

  // 1D
  at::Tensor t1 = at::empty({10}, at::kFloat);
  file << "1d " << std::to_string(t1.size(0)) << " ";

  // 2D
  at::Tensor t2 = at::empty({5, 6}, at::kFloat);
  file << "2d " << std::to_string(t2.size(0)) << " "
       << std::to_string(t2.size(1)) << " ";

  // Large
  at::Tensor t3 = at::empty({100, 100}, at::kFloat);
  file << "large " << std::to_string(t3.numel()) << " ";
  file.saveFile();
}

}  // namespace test
}  // namespace at
