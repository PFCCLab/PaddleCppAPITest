#include <ATen/ATen.h>
#include <ATen/ops/zeros.h>
#include <c10/core/DefaultDtype.h>
#include <c10/core/ScalarType.h>
#include <c10/core/TensorOptions.h>
#include <gtest/gtest.h>

#include <string>
#include <type_traits>

#include "../src/file_manager.h"

extern paddle_api_test::ThreadSafeParam g_custom_param;

namespace at {
namespace test {

using paddle_api_test::FileManerger;
using paddle_api_test::ThreadSafeParam;

template <typename DType>
static int dtype_to_int(const DType& dtype) {
  if constexpr (std::is_same_v<std::decay_t<DType>, c10::ScalarType>) {
    return static_cast<int>(dtype);
  } else {
    return static_cast<int>(dtype.toScalarType());
  }
}

class DefaultDtypeTest : public ::testing::Test {
 protected:
  // Save and restore the global default dtype so tests are isolated.
  void SetUp() override {
    original_dtype_ = c10::get_default_dtype_as_scalartype();
  }

  void TearDown() override {
    // Use tensor to restore the original dtype
    at::Tensor t = at::zeros({1}, at::TensorOptions().dtype(original_dtype_));
    c10::set_default_dtype(t.dtype());
  }

  c10::ScalarType original_dtype_;
};

// 获取默认 dtype（应为 Float）
TEST_F(DefaultDtypeTest, GetDefaultDtype) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();
  file << "GetDefaultDtype ";
  auto dtype = c10::get_default_dtype();
  file << std::to_string(dtype_to_int(dtype)) << " ";
  file << "\n";
  file.saveFile();
}

// get_default_dtype_as_scalartype
TEST_F(DefaultDtypeTest, GetDefaultDtypeAsScalarType) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "GetDefaultDtypeAsScalarType ";
  auto dtype = c10::get_default_dtype_as_scalartype();
  file << std::to_string(static_cast<int>(dtype)) << " ";
  file << "\n";
  file.saveFile();
}

// set_default_dtype 到 Double
TEST_F(DefaultDtypeTest, SetDefaultDtypeDouble) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "SetDefaultDtypeDouble ";
  at::Tensor t =
      at::zeros({1}, at::TensorOptions().dtype(c10::ScalarType::Double));
  c10::set_default_dtype(t.dtype());
  auto dtype = c10::get_default_dtype_as_scalartype();
  file << std::to_string(static_cast<int>(dtype)) << " ";
  file << "\n";
  file.saveFile();
}

// set_default_dtype 到 Half
TEST_F(DefaultDtypeTest, SetDefaultDtypeHalf) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "SetDefaultDtypeHalf ";
  at::Tensor t =
      at::zeros({1}, at::TensorOptions().dtype(c10::ScalarType::Half));
  c10::set_default_dtype(t.dtype());
  auto dtype = c10::get_default_dtype_as_scalartype();
  file << std::to_string(static_cast<int>(dtype)) << " ";
  file << "\n";
  file.saveFile();
}

// set_default_dtype 到 BFloat16
// [DIFF] PyTorch输出: 11, PaddlePaddle输出: 15
TEST_F(DefaultDtypeTest, SetDefaultDtypeBFloat16) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "SetDefaultDtypeBFloat16 ";
  at::Tensor t =
      at::zeros({1}, at::TensorOptions().dtype(c10::ScalarType::BFloat16));
  c10::set_default_dtype(t.dtype());
  auto dtype = c10::get_default_dtype_as_scalartype();
  file << std::to_string(static_cast<int>(dtype)) << " ";
  file << "\n";
  file.saveFile();
}

// 设置后恢复
TEST_F(DefaultDtypeTest, SetAndRestore) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "SetAndRestore ";
  auto before = c10::get_default_dtype_as_scalartype();
  at::Tensor t1 =
      at::zeros({1}, at::TensorOptions().dtype(c10::ScalarType::Double));
  c10::set_default_dtype(t1.dtype());
  auto during = c10::get_default_dtype_as_scalartype();
  at::Tensor t2 = at::zeros({1}, at::TensorOptions().dtype(before));
  c10::set_default_dtype(t2.dtype());
  auto after = c10::get_default_dtype_as_scalartype();
  file << std::to_string(static_cast<int>(before)) << " ";
  file << std::to_string(static_cast<int>(during)) << " ";
  file << std::to_string(static_cast<int>(after)) << " ";
  // before 应等于 after
  file << std::to_string(before == after ? 1 : 0) << " ";
  file << "\n";
  file.saveFile();
}

// get_default_complex_dtype
TEST_F(DefaultDtypeTest, GetDefaultComplexDtype) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "GetDefaultComplexDtype ";
  auto dtype = c10::get_default_complex_dtype();
  file << std::to_string(dtype_to_int(dtype)) << " ";
  file << "\n";
  file.saveFile();
}

}  // namespace test
}  // namespace at
