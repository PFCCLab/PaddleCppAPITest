#include <ATen/ATen.h>
#include <ATen/ops/zeros.h>
#include <c10/core/DefaultDtype.h>
#include <c10/core/ScalarType.h>
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
  void SetUp() override {
    // 保存原始默认 dtype
    original_dtype_ = c10::get_default_dtype();
  }

  void TearDown() override {
    // 恢复原始默认 dtype
    c10::set_default_dtype(original_dtype_);
  }

  c10::ScalarType original_dtype_;
};

// 获取默认 dtype（应为 Float）
TEST_F(DefaultDtypeTest, GetDefaultDtype) {
  auto dtype = c10::get_default_dtype();
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();
  file << std::to_string(static_cast<int>(dtype)) << " ";
  file.saveFile();
}

// get_default_dtype_as_scalartype
TEST_F(DefaultDtypeTest, GetDefaultDtypeAsScalarType) {
  auto dtype = c10::get_default_dtype_as_scalartype();
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << std::to_string(static_cast<int>(dtype)) << " ";
  file.saveFile();
}

// set_default_dtype 到 Double
TEST_F(DefaultDtypeTest, SetDefaultDtypeDouble) {
  c10::set_default_dtype(c10::ScalarType::Double);
  auto dtype = c10::get_default_dtype();
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << std::to_string(static_cast<int>(dtype)) << " ";
  file.saveFile();
}

// set_default_dtype 到 Half
TEST_F(DefaultDtypeTest, SetDefaultDtypeHalf) {
  c10::set_default_dtype(c10::ScalarType::Half);
  auto dtype = c10::get_default_dtype();
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << std::to_string(static_cast<int>(dtype)) << " ";
  file.saveFile();
}

// set_default_dtype 到 BFloat16
TEST_F(DefaultDtypeTest, SetDefaultDtypeBFloat16) {
  c10::set_default_dtype(c10::ScalarType::BFloat16);
  auto dtype = c10::get_default_dtype();
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << std::to_string(static_cast<int>(dtype)) << " ";
  file.saveFile();
}

// 设置后恢复
TEST_F(DefaultDtypeTest, SetAndRestore) {
  auto before = c10::get_default_dtype();
  c10::set_default_dtype(c10::ScalarType::Double);
  auto during = c10::get_default_dtype();
  c10::set_default_dtype(before);
  auto after = c10::get_default_dtype();

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << std::to_string(static_cast<int>(before)) << " ";
  file << std::to_string(static_cast<int>(during)) << " ";
  file << std::to_string(static_cast<int>(after)) << " ";
  // before 应等于 after
  file << std::to_string(before == after ? 1 : 0) << " ";
  file.saveFile();
}

// get_default_complex_dtype
TEST_F(DefaultDtypeTest, GetDefaultComplexDtype) {
  auto dtype = c10::get_default_complex_dtype();
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << std::to_string(static_cast<int>(dtype)) << " ";
  file.saveFile();
}

}  // namespace test
}  // namespace at
