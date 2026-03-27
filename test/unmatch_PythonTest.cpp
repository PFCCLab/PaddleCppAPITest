#include <ATen/ATen.h>
#include <gtest/gtest.h>
#ifndef USE_PADDLE_API
#include <torch/python.h>
#endif

#include <string>

#include "../src/file_manager.h"

extern paddle_api_test::ThreadSafeParam g_custom_param;

namespace at {
namespace test {

using paddle_api_test::FileManerger;
using paddle_api_test::ThreadSafeParam;

class PythonTest : public ::testing::Test {
 protected:
  void SetUp() override {}
};

// [DIFF] 文件级说明：torch/python.h 与 torch::getTHPDtype 属于 Python C API
// 桥接面， Paddle compat 当前不提供等价能力，因此本文件保留在 unmatch。

#ifndef USE_PADDLE_API
// [DIFF] 问题行：整个用例块仅 Torch 路径启用，Paddle
// 路径没有可编译的等价头/符号。

// 测试 torch::getTHPDtype 函数存在性
// 该函数将 c10::ScalarType 转换为 Python dtype 对象
TEST_F(PythonTest, GetTHPDtype) {
  // [DIFF] 用例级差异：getTHPDtype 在 Paddle 侧不存在。
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();
  file << "GetTHPDtype ";

  // 测试各种 ScalarType 到 Python 对象的转换
  // 注意：实际转换需要 Python 运行时，这里只验证函数存在且可以调用

  // 基础类型测试 - 使用 libtorch 支持的类型
  c10::ScalarType types[] = {c10::ScalarType::Float,
                             c10::ScalarType::Double,
                             c10::ScalarType::Int,
                             c10::ScalarType::Long,
                             c10::ScalarType::Short,
                             c10::ScalarType::Char,
                             c10::ScalarType::Byte,
                             c10::ScalarType::Bool};

  file << std::to_string(sizeof(types) / sizeof(types[0])) << " ";

  for (const auto& dtype : types) {
    // 调用 getTHPDtype，并在未抛异常的情况下通过
    try {
      // torch::getTHPDtype returns THPDtype*, not PyObject*
      // [DIFF] 问题行：该调用依赖 Torch Python C API 桥接符号。
      torch::getTHPDtype(dtype);
      file << "1 ";
    } catch (...) {
      file << "exception ";
    }
  }

  file << "\n";
  file.saveFile();
}

// 测试 torch::getTHPDtype 函数存在性 (简化版)
TEST_F(PythonTest, PyObjectToDtype) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "PyObjectToDtype ";

  // 测试 getTHPDtype 函数可用
  try {
    torch::getTHPDtype(c10::ScalarType::Float);
    file << "1 ";
  } catch (...) {
    file << "exception ";
  }

  file << "\n";
  file.saveFile();
}

// 测试 torch 命名空间下的 getTHPDtype 存在性
TEST_F(PythonTest, NamespaceExists) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "NamespaceExists ";

  // 验证 torch::getTHPDtype 可用
  // NOLINTNEXTLINE
  (void)torch::getTHPDtype;

  // 如果编译通过，说明函数存在
  file << "1 ";
  file << "\n";
  file.saveFile();
}

// 测试 getTHPDtype 与各种 ScalarType 的兼容性
TEST_F(PythonTest, GetTHPDtypeAllTypes) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "GetTHPDtypeAllTypes ";

  // 测试主要 ScalarType - 使用 libtorch 支持的类型
  c10::ScalarType all_types[] = {
      c10::ScalarType::Float,
      c10::ScalarType::Double,
      c10::ScalarType::Half,
      c10::ScalarType::BFloat16,
      c10::ScalarType::Int,
      c10::ScalarType::Long,
      c10::ScalarType::Short,
      c10::ScalarType::Char,
      c10::ScalarType::Byte,
      c10::ScalarType::Bool,
  };

  file << std::to_string(sizeof(all_types) / sizeof(all_types[0])) << " ";

  for (const auto& st : all_types) {
    try {
      torch::getTHPDtype(st);
      file << "1 ";
    } catch (...) {
      file << "exception ";
    }
  }

  file << "\n";
  file.saveFile();
}

// 测试 torch 命名空间下的 getTHPDtype 别名
TEST_F(PythonTest, TorchNamespaceGetTHPDtype) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();

  // torch::getTHPDtype 是 torch::python::detail::getTHPDtype 的别名
  c10::ScalarType dtype = c10::ScalarType::Float;

  try {
    // torch::getTHPDtype returns THPDtype*, not PyObject*
    torch::getTHPDtype(dtype);
    file << "1 ";
  } catch (...) {
    file << "exception ";
  }
}

#endif  // USE_PADDLE_API

}  // namespace test
}  // namespace at
