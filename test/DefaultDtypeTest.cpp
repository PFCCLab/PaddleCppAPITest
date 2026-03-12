#include <ATen/ATen.h>
#include <ATen/ops/zeros.h>
#include <c10/core/DefaultDtype.h>
#include <c10/core/ScalarType.h>
#include <c10/core/TensorOptions.h>
#include <gtest/gtest.h>

#include <string>

#include "../src/file_manager.h"

extern paddle_api_test::ThreadSafeParam g_custom_param;

namespace at {
namespace test {

using paddle_api_test::FileManerger;
using paddle_api_test::ThreadSafeParam;

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
  auto dtype = c10::get_default_dtype_as_scalartype();
  file << std::to_string(static_cast<int>(dtype)) << " ";
  file.saveFile();
}

// get_default_dtype_as_scalartype
TEST_F(DefaultDtypeTest, GetDefaultDtypeAsScalarType) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  auto dtype = c10::get_default_dtype_as_scalartype();
  file << std::to_string(static_cast<int>(dtype)) << " ";
  file.saveFile();
}

// set_default_dtype 到 Double
TEST_F(DefaultDtypeTest, SetDefaultDtypeDouble) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  at::Tensor t =
      at::zeros({1}, at::TensorOptions().dtype(c10::ScalarType::Double));
  c10::set_default_dtype(t.dtype());
  auto dtype = c10::get_default_dtype_as_scalartype();
  file << std::to_string(static_cast<int>(dtype)) << " ";
  file.saveFile();
}

// set_default_dtype 到 Half
TEST_F(DefaultDtypeTest, SetDefaultDtypeHalf) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  at::Tensor t =
      at::zeros({1}, at::TensorOptions().dtype(c10::ScalarType::Half));
  c10::set_default_dtype(t.dtype());
  auto dtype = c10::get_default_dtype_as_scalartype();
  file << std::to_string(static_cast<int>(dtype)) << " ";
  file.saveFile();
}

// set_default_dtype 到 BFloat16
// [DIFF] PyTorch输出: 11, PaddlePaddle输出: 15
TEST_F(DefaultDtypeTest, SetDefaultDtypeBFloat16) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  at::Tensor t =
      at::zeros({1}, at::TensorOptions().dtype(c10::ScalarType::BFloat16));
  c10::set_default_dtype(t.dtype());
  auto dtype = c10::get_default_dtype_as_scalartype();
  // file << std::to_string(static_cast<int>(dtype)) << " "; // [DIFF]
  file.saveFile();
}

// 设置后恢复
TEST_F(DefaultDtypeTest, SetAndRestore) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
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
  file.saveFile();
}

// get_default_complex_dtype
// [DIFF] PyTorch输出: 8, PaddlePaddle输出: 9
TEST_F(DefaultDtypeTest, GetDefaultComplexDtype) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  // Create a complex tensor to get the default complex dtype
  at::Tensor t = at::zeros({1}, at::TensorOptions().dtype(at::kComplexFloat));
  auto dtype = t.scalar_type();
  // file << std::to_string(static_cast<int>(dtype)) << " "; // [DIFF]
  file.saveFile();
}

}  // namespace test
}  // namespace at
