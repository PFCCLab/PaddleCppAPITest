#include <ATen/ATen.h>
#include <ATen/core/Tensor.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/full.h>
#include <ATen/ops/ones.h>
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

static void write_creation_result_to_file(FileManerger* file,
                                          const at::Tensor& result) {
  *file << std::to_string(result.dim()) << " ";
  *file << std::to_string(result.numel()) << " ";
  for (int64_t i = 0; i < result.dim(); ++i) {
    *file << std::to_string(result.sizes()[i]) << " ";
  }
}

class CreationOpsTest : public ::testing::Test {
 protected:
  void SetUp() override {}
};

// ========== zeros 测试 ==========

TEST_F(CreationOpsTest, ZerosBasic) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();
  file << "ZerosBasic ";
  std::vector<int64_t> shape = {2, 3};
  at::Tensor result = at::zeros(shape);
  write_creation_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

TEST_F(CreationOpsTest, ZerosWithOptions) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "ZerosWithOptions ";
  at::Tensor result = at::zeros({3, 4}, at::TensorOptions().dtype(at::kDouble));
  file << std::to_string(static_cast<int>(result.scalar_type())) << " ";
  write_creation_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// 标量 zeros
TEST_F(CreationOpsTest, ZerosScalar) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "ZerosScalar ";
  at::Tensor result = at::zeros({});
  write_creation_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// 大 shape zeros
TEST_F(CreationOpsTest, ZerosLargeShape) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "ZerosLargeShape ";
  at::Tensor result = at::zeros({100, 100});
  file << std::to_string(result.numel()) << " ";
  write_creation_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// 零维度 zeros
TEST_F(CreationOpsTest, ZerosZeroDim) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "ZerosZeroDim ";
  at::Tensor result = at::zeros({2, 0, 3});
  write_creation_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// ========== ones 测试 ==========

TEST_F(CreationOpsTest, OnesBasic) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "OnesBasic ";
  at::Tensor result = at::ones({2, 2});
  write_creation_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

TEST_F(CreationOpsTest, OnesWithOptions) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "OnesWithOptions ";
  at::Tensor result = at::ones({3}, at::TensorOptions().dtype(at::kInt));
  file << std::to_string(static_cast<int>(result.scalar_type())) << " ";
  write_creation_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// 标量 ones
TEST_F(CreationOpsTest, OnesScalar) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "OnesScalar ";
  at::Tensor result = at::ones({});
  write_creation_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// 大 shape ones
TEST_F(CreationOpsTest, OnesLargeShape) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "OnesLargeShape ";
  at::Tensor result = at::ones({100, 100});
  file << std::to_string(result.numel()) << " ";
  write_creation_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// 零维度 ones
TEST_F(CreationOpsTest, OnesZeroDim) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "OnesZeroDim ";
  at::Tensor result = at::ones({2, 0, 3});
  write_creation_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// ========== empty 测试 ==========

TEST_F(CreationOpsTest, EmptyBasic) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "EmptyBasic ";
  at::Tensor result = at::empty({2, 3});
  file << std::to_string(result.data_ptr() != nullptr) << " ";
  write_creation_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

TEST_F(CreationOpsTest, EmptyWithOptions) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "EmptyWithOptions ";
  at::Tensor result = at::empty({4}, at::TensorOptions().dtype(at::kFloat));
  file << std::to_string(static_cast<int>(result.scalar_type())) << " ";
  file << std::to_string(result.data_ptr() != nullptr) << " ";
  write_creation_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// 标量 empty
TEST_F(CreationOpsTest, EmptyScalar) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "EmptyScalar ";
  at::Tensor result = at::empty({});
  write_creation_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// 大 shape empty
TEST_F(CreationOpsTest, EmptyLargeShape) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "EmptyLargeShape ";
  at::Tensor result = at::empty({100, 100});
  file << std::to_string(result.numel()) << " ";
  write_creation_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// 零维度 empty
TEST_F(CreationOpsTest, EmptyZeroDim) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "EmptyZeroDim ";
  at::Tensor result = at::empty({2, 0, 3});
  write_creation_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// ========== full 测试 ==========

TEST_F(CreationOpsTest, FullBasic) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "FullBasic ";
  at::Tensor result = at::full({2, 2}, 5.0f);
  write_creation_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

TEST_F(CreationOpsTest, FullWithOptions) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "FullWithOptions ";
  at::Tensor result = at::full({3}, 10, at::TensorOptions().dtype(at::kLong));
  file << std::to_string(static_cast<int>(result.scalar_type())) << " ";
  write_creation_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// 标量 full
TEST_F(CreationOpsTest, FullScalar) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "FullScalar ";
  at::Tensor result = at::full({}, 42.0f);
  write_creation_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// 大 shape full
TEST_F(CreationOpsTest, FullLargeShape) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "FullLargeShape ";
  at::Tensor result = at::full({100, 100}, 7.0f);
  file << std::to_string(result.numel()) << " ";
  write_creation_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// 零维度 full
TEST_F(CreationOpsTest, FullZeroDim) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "FullZeroDim ";
  at::Tensor result = at::full({2, 0, 3}, 5.0f);
  write_creation_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// ========== Dtype 覆盖 ==========

// zeros 不同 dtype
TEST_F(CreationOpsTest, ZerosInt32) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "ZerosInt32 ";
  at::Tensor result = at::zeros({2, 3}, at::TensorOptions().dtype(at::kInt));
  file << std::to_string(static_cast<int>(result.scalar_type())) << " ";
  write_creation_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

TEST_F(CreationOpsTest, ZerosInt64) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "ZerosInt64 ";
  at::Tensor result = at::zeros({2, 3}, at::TensorOptions().dtype(at::kLong));
  file << std::to_string(static_cast<int>(result.scalar_type())) << " ";
  write_creation_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// ones 不同 dtype
TEST_F(CreationOpsTest, OnesFloat64) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "OnesFloat64 ";
  at::Tensor result = at::ones({2, 3}, at::TensorOptions().dtype(at::kDouble));
  file << std::to_string(static_cast<int>(result.scalar_type())) << " ";
  write_creation_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

TEST_F(CreationOpsTest, OnesInt64) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "OnesInt64 ";
  at::Tensor result = at::ones({2, 3}, at::TensorOptions().dtype(at::kLong));
  file << std::to_string(static_cast<int>(result.scalar_type())) << " ";
  write_creation_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// ========== pin_memory 分支覆盖 ==========

TEST_F(CreationOpsTest, ZerosPinnedMemoryCPU) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "ZerosPinnedMemoryCPU ";
  try {
    auto options = at::TensorOptions()
                       .dtype(at::kFloat)
                       .device(at::kCPU)
                       .pinned_memory(true);
    at::Tensor result = at::zeros({2, 3}, options);
    file << "ok ";
    file << std::to_string(result.is_pinned() ? 1 : 0) << " ";
    write_creation_result_to_file(&file, result);
  } catch (const std::exception&) {
    file << "exception ";
  } catch (...) {
    file << "exception ";
  }
  file << "\n";
  file.saveFile();
}

TEST_F(CreationOpsTest, ZerosPinnedMemoryCUDADevice) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "ZerosPinnedMemoryCUDADevice ";
  try {
    auto options = at::TensorOptions()
                       .dtype(at::kFloat)
                       .device(at::kCUDA)
                       .pinned_memory(true);
    at::Tensor result = at::zeros({2, 3}, options);
    file << "ok ";
    write_creation_result_to_file(&file, result);
  } catch (const std::exception&) {
    file << "exception ";
  } catch (...) {
    file << "exception ";
  }
  file << "\n";
  file.saveFile();
}

TEST_F(CreationOpsTest, OnesPinnedMemoryCPU) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "OnesPinnedMemoryCPU ";
  try {
    auto options = at::TensorOptions()
                       .dtype(at::kFloat)
                       .device(at::kCPU)
                       .pinned_memory(true);
    at::Tensor result = at::ones({2, 3}, options);
    file << "ok ";
    file << std::to_string(result.is_pinned() ? 1 : 0) << " ";
    write_creation_result_to_file(&file, result);
  } catch (const std::exception&) {
    file << "exception ";
  } catch (...) {
    file << "exception ";
  }
  file << "\n";
  file.saveFile();
}

TEST_F(CreationOpsTest, OnesPinnedMemoryCUDADevice) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "OnesPinnedMemoryCUDADevice ";
  try {
    auto options = at::TensorOptions()
                       .dtype(at::kFloat)
                       .device(at::kCUDA)
                       .pinned_memory(true);
    at::Tensor result = at::ones({2, 3}, options);
    file << "ok ";
    write_creation_result_to_file(&file, result);
  } catch (const std::exception&) {
    file << "exception ";
  } catch (...) {
    file << "exception ";
  }
  file << "\n";
  file.saveFile();
}

TEST_F(CreationOpsTest, FullPinnedMemoryCPU) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "FullPinnedMemoryCPU ";
  try {
    auto options = at::TensorOptions()
                       .dtype(at::kFloat)
                       .device(at::kCPU)
                       .pinned_memory(true);
    at::Tensor result = at::full({2, 3}, 7.0f, options);
    file << "ok ";
    file << std::to_string(result.is_pinned() ? 1 : 0) << " ";
    write_creation_result_to_file(&file, result);
  } catch (const std::exception&) {
    file << "exception ";
  } catch (...) {
    file << "exception ";
  }
  file << "\n";
  file.saveFile();
}

TEST_F(CreationOpsTest, FullPinnedMemoryCUDADevice) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "FullPinnedMemoryCUDADevice ";
  try {
    auto options = at::TensorOptions()
                       .dtype(at::kFloat)
                       .device(at::kCUDA)
                       .pinned_memory(true);
    at::Tensor result = at::full({2, 3}, 7.0f, options);
    file << "ok ";
    write_creation_result_to_file(&file, result);
  } catch (const std::exception&) {
    file << "exception ";
  } catch (...) {
    file << "exception ";
  }
  file << "\n";
  file.saveFile();
}

TEST_F(CreationOpsTest, EmptyPinnedMemoryCPU) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "EmptyPinnedMemoryCPU ";
  try {
    auto options = at::TensorOptions()
                       .dtype(at::kFloat)
                       .device(at::kCPU)
                       .pinned_memory(true);
    at::Tensor result = at::empty({2, 3}, options);
    file << "ok ";
    file << std::to_string(result.is_pinned() ? 1 : 0) << " ";
    write_creation_result_to_file(&file, result);
  } catch (const std::exception&) {
    file << "exception ";
  } catch (...) {
    file << "exception ";
  }
  file << "\n";
  file.saveFile();
}

TEST_F(CreationOpsTest, EmptyPinnedMemoryCUDADevice) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "EmptyPinnedMemoryCUDADevice ";
  try {
    auto options = at::TensorOptions()
                       .dtype(at::kFloat)
                       .device(at::kCUDA)
                       .pinned_memory(true);
    at::Tensor result = at::empty({2, 3}, options);
    file << "ok ";
    write_creation_result_to_file(&file, result);
  } catch (const std::exception&) {
    file << "exception ";
  } catch (...) {
    file << "exception ";
  }
  file << "\n";
  file.saveFile();
}

// 间接触发 _PD_AtenScalarTypeToPhiDataType 的 case Undefined (L30-31)
TEST_F(CreationOpsTest, ZerosUndefinedDtype) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "ZerosUndefinedDtype ";
  try {
    at::Tensor result =
        at::zeros({1}, at::TensorOptions().dtype(c10::ScalarType::Undefined));
    file << "success ";
  } catch (...) {
    file << "exception ";
  }
  file << "\n";
  file.saveFile();
}

// 间接触发 _PD_AtenScalarTypeToPhiDataType 的 default (L32-34)
TEST_F(CreationOpsTest, ZerosNumOptionsDtype) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "ZerosNumOptionsDtype ";
  try {
    at::Tensor result =
        at::zeros({1}, at::TensorOptions().dtype(c10::ScalarType::QInt8));
    file << "success ";
  } catch (...) {
    file << "exception ";
  }
  file << "\n";
  file.saveFile();
}

}  // namespace test
}  // namespace at
