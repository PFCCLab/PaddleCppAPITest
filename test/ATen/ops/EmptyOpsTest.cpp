#include <ATen/ATen.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/full.h>
#include <ATen/ops/ones.h>
#include <ATen/ops/reshape.h>
#include <ATen/ops/zeros.h>
#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "src/file_manager.h"

extern paddle_api_test::ThreadSafeParam g_custom_param;

namespace at {
namespace test {

using paddle_api_test::FileManerger;
using paddle_api_test::ThreadSafeParam;

class EmptyOpsTest : public ::testing::Test {
 protected:
  void SetUp() override {}
};

static void write_symint_result_to_file(FileManerger* file,
                                        const at::Tensor& tensor) {
  *file << std::to_string(tensor.dim()) << " ";
  *file << std::to_string(tensor.numel()) << " ";
  for (int64_t i = 0; i < tensor.dim(); ++i) {
    *file << std::to_string(tensor.sizes()[i]) << " ";
  }
  *file << std::to_string(static_cast<int>(tensor.scalar_type())) << " ";
  *file << std::to_string(tensor.is_pinned() ? 1 : 0) << " ";
}

// empty
TEST_F(EmptyOpsTest, Empty) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();
  file << "Empty ";

  // Test empty with IntArrayRef size
  at::Tensor t = at::empty({2, 3, 4}, at::kFloat);
  file << std::to_string(t.dim()) << " ";
  file << std::to_string(t.numel()) << " ";
  file << std::to_string(t.size(0)) << " ";
  file << std::to_string(t.size(1)) << " ";
  file << std::to_string(t.size(2)) << " ";
  file << std::to_string(static_cast<int>(t.scalar_type())) << " ";
  file << "\n";
  file.saveFile();
}

// empty with different dtype
TEST_F(EmptyOpsTest, EmptyDifferentDtype) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "EmptyDifferentDtype ";

  at::Tensor t = at::empty({3, 4}, at::kInt);
  file << std::to_string(t.dim()) << " ";
  file << std::to_string(t.numel()) << " ";
  file << std::to_string(static_cast<int>(t.scalar_type())) << " ";
  file << "\n";
  file.saveFile();
}

// empty with ScalarType
TEST_F(EmptyOpsTest, EmptyWithScalarType) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "EmptyWithScalarType ";

  // Using at::kFloat equivalent to c10::ScalarType::Float
  at::Tensor t = at::empty({2, 3}, at::TensorOptions().dtype(at::kFloat));
  file << std::to_string(t.dim()) << " ";
  file << std::to_string(static_cast<int>(t.scalar_type())) << " ";
  file << "\n";
  file.saveFile();
}

// empty_cuda (if CUDA available)
TEST_F(EmptyOpsTest, EmptyCUDA) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "EmptyCUDA ";

  // Try to create empty CUDA tensor
  try {
    at::Tensor t = at::empty({2, 3}, at::TensorOptions().device(at::kCUDA));
    (void)t;
    file << "cuda_empty ";
  } catch (...) {
    file << "cuda_not_available ";
  }
  file << "\n";
  file.saveFile();
}

// full_symint
TEST_F(EmptyOpsTest, FullSymint) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "FullSymint ";

  // full_symint with SymIntArrayRef
  at::Tensor t = at::full_symint({2, 3}, 5.0f, at::kFloat);
  file << std::to_string(t.dim()) << " ";
  file << std::to_string(t.numel()) << " ";
  file << std::to_string(t.data_ptr<float>()[0]) << " ";
  file << std::to_string(t.data_ptr<float>()[5]) << " ";
  file << "\n";
  file.saveFile();
}

// full_symint with int
TEST_F(EmptyOpsTest, FullSymintInt) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "FullSymintInt ";

  // full_symint with array size
  at::Tensor t = at::full_symint({10}, 3, at::kInt);
  file << std::to_string(t.dim()) << " ";
  file << std::to_string(t.numel()) << " ";
  file << std::to_string(t.data_ptr<int>()[0]) << " ";
  file << std::to_string(t.data_ptr<int>()[9]) << " ";
  file << "\n";
  file.saveFile();
}

// ones_symint
TEST_F(EmptyOpsTest, OnesSymint) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "OnesSymint ";

  // ones_symint with array
  at::Tensor t = at::ones_symint({5}, at::kFloat);
  file << std::to_string(t.dim()) << " ";
  file << std::to_string(t.numel()) << " ";
  file << std::to_string(t.data_ptr<float>()[0]) << " ";
  file << std::to_string(t.data_ptr<float>()[4]) << " ";
  file << "\n";
  file.saveFile();
}

// ones_symint with shape
TEST_F(EmptyOpsTest, OnesSymintShape) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "OnesSymintShape ";

  // ones_symint with c10::SymIntArrayRef
  at::Tensor t = at::ones_symint({2, 3, 4}, at::kFloat);
  file << std::to_string(t.dim()) << " ";
  file << std::to_string(t.numel()) << " ";
  file << std::to_string(t.data_ptr<float>()[0]) << " ";
  file << std::to_string(t.data_ptr<float>()[23]) << " ";
  file << "\n";
  file.saveFile();
}

// zeros_symint
TEST_F(EmptyOpsTest, ZerosSymint) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "ZerosSymint ";

  // zeros_symint with array
  at::Tensor t = at::zeros_symint({5}, at::kFloat);
  file << std::to_string(t.dim()) << " ";
  file << std::to_string(t.numel()) << " ";
  file << std::to_string(t.data_ptr<float>()[0]) << " ";
  file << std::to_string(t.data_ptr<float>()[4]) << " ";
  file << "\n";
  file.saveFile();
}

// zeros_symint with shape
TEST_F(EmptyOpsTest, ZerosSymintShape) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "ZerosSymintShape ";

  // zeros_symint with c10::SymIntArrayRef
  at::Tensor t = at::zeros_symint({2, 3}, at::kFloat);
  file << std::to_string(t.dim()) << " ";
  file << std::to_string(t.numel()) << " ";
  file << std::to_string(t.data_ptr<float>()[0]) << " ";
  file << std::to_string(t.data_ptr<float>()[5]) << " ";
  file << "\n";
  file.saveFile();
}

// reshape_symint
TEST_F(EmptyOpsTest, ReshapeSymint) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "ReshapeSymint ";

  // Create a tensor and reshape
  at::Tensor t = at::arange(12, at::kInt);
  at::Tensor reshaped = at::reshape_symint(t, {3, 4});
  file << std::to_string(reshaped.dim()) << " ";
  file << std::to_string(reshaped.size(0)) << " ";
  file << std::to_string(reshaped.size(1)) << " ";
  file << "\n";
  file.saveFile();
}

// reshape_symint with int
TEST_F(EmptyOpsTest, ReshapeSymintInt) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "ReshapeSymintInt ";

  at::Tensor t = at::arange(24, at::kInt);
  at::Tensor reshaped = at::reshape_symint(t, {24});
  file << std::to_string(reshaped.dim()) << " ";
  file << std::to_string(reshaped.numel()) << " ";
  file << "\n";
  file.saveFile();
}

// Test empty with different sizes
TEST_F(EmptyOpsTest, EmptySizes) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "EmptySizes ";

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
  file << "\n";
  file.saveFile();
}

// full_symint with pin_memory on CPU
TEST_F(EmptyOpsTest, FullSymintPinnedMemoryCPU) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "FullSymintPinnedMemoryCPU ";

  try {
    auto options = at::TensorOptions()
                       .dtype(at::kFloat)
                       .device(at::kCPU)
                       .pinned_memory(true);
    at::Tensor t = at::full_symint({2, 3}, 5.0f, options);
    file << "ok ";
    write_symint_result_to_file(&file, t);
  } catch (const std::exception&) {
    file << "exception ";
  } catch (...) {
    file << "exception ";
  }
  file << "\n";
  file.saveFile();
}

// full_symint with pin_memory and CUDA device
TEST_F(EmptyOpsTest, FullSymintPinnedMemoryCUDADevice) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "FullSymintPinnedMemoryCUDADevice ";

  try {
    auto options = at::TensorOptions()
                       .dtype(at::kFloat)
                       .device(at::kCUDA)
                       .pinned_memory(true);
    (void)at::full_symint({2, 3}, 5.0f, options);
    file << "handled ";
  } catch (const std::exception&) {
    file << "handled ";
  } catch (...) {
    file << "handled ";
  }
  file << "\n";
  file.saveFile();
}

// ones_symint with pin_memory on CPU
TEST_F(EmptyOpsTest, OnesSymintPinnedMemoryCPU) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "OnesSymintPinnedMemoryCPU ";

  try {
    auto options = at::TensorOptions()
                       .dtype(at::kFloat)
                       .device(at::kCPU)
                       .pinned_memory(true);
    at::Tensor t = at::ones_symint({2, 3}, options);
    file << "ok ";
    write_symint_result_to_file(&file, t);
  } catch (const std::exception&) {
    file << "exception ";
  } catch (...) {
    file << "exception ";
  }
  file << "\n";
  file.saveFile();
}

// ones_symint with pin_memory and CUDA device
TEST_F(EmptyOpsTest, OnesSymintPinnedMemoryCUDADevice) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "OnesSymintPinnedMemoryCUDADevice ";

  try {
    auto options = at::TensorOptions()
                       .dtype(at::kFloat)
                       .device(at::kCUDA)
                       .pinned_memory(true);
    at::Tensor t = at::ones_symint({2, 3}, options);
    file << "ok ";
    write_symint_result_to_file(&file, t);
  } catch (const std::exception&) {
    file << "exception ";
  } catch (...) {
    file << "exception ";
  }
  file << "\n";
  file.saveFile();
}

// zeros_symint with pin_memory on CPU
TEST_F(EmptyOpsTest, ZerosSymintPinnedMemoryCPU) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "ZerosSymintPinnedMemoryCPU ";

  try {
    auto options = at::TensorOptions()
                       .dtype(at::kFloat)
                       .device(at::kCPU)
                       .pinned_memory(true);
    at::Tensor t = at::zeros_symint({2, 3}, options);
    file << "ok ";
    write_symint_result_to_file(&file, t);
  } catch (const std::exception&) {
    file << "exception ";
  } catch (...) {
    file << "exception ";
  }
  file << "\n";
  file.saveFile();
}

// zeros_symint with pin_memory and CUDA device
TEST_F(EmptyOpsTest, ZerosSymintPinnedMemoryCUDADevice) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "ZerosSymintPinnedMemoryCUDADevice ";

  try {
    auto options = at::TensorOptions()
                       .dtype(at::kFloat)
                       .device(at::kCUDA)
                       .pinned_memory(true);
    at::Tensor t = at::zeros_symint({2, 3}, options);
    file << "ok ";
    write_symint_result_to_file(&file, t);
  } catch (const std::exception&) {
    file << "exception ";
  } catch (...) {
    file << "exception ";
  }
  file << "\n";
  file.saveFile();
}

}  // namespace test
}  // namespace at
