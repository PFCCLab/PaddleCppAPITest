/*
 * =====================================================================================
 * @brief: 兼容性对齐审计报告
 *
 * [异常点 1]
 * - 测试用例：整个文件的公共头文件引入部分
 * - 当前状况：`#include <torch/python.h>` 被 `#ifndef USE_PADDLE_API` 宏包裹，
 *             在 Paddle 构建模式下该头文件完全不被包含。
 * - 根本原因：Paddle compat 的 `torch/` 目录下仅有 `extension.h`、`library.h`、
 *             `library.cpp`，**完全没有 `python.h` 文件**。尽管 Paddle compat
 * 的 `torch/extension.h` 中有 `#include <torch/python.h>`，但
 *             对应文件并不存在，若在 Paddle 构建模式下包含此头文件将直接引发
 *             编译错误（找不到头文件）。
 *             而 libtorch 的 `torch/python.h` 存在并通过
 * `torch/csrc/DynamicTypes.h` 声明 `getTHPDtype(at::ScalarType) ->
 * THPDtype*`，是 PyTorch Python C API 桥接层的核心头文件。
 * - 期望解决：Paddle compat 需要在 `torch/` 下提供一个 `python.h` 文件，模拟
 *             `torch::getTHPDtype` 接口的声明（即使是 stub
 * 实现也可），才能使测试 中的 `#include <torch/python.h>`
 * 无需使用宏保护即可统一包含。
 *
 * [异常点 2]
 * - 测试用例：GetTHPDtype / PyObjectToDtype / NamespaceExists /
 *             GetTHPDtypeAllTypes / TorchNamespaceGetTHPDtype
 *             （共 5 个 TEST_F，全部被禁用）
 * - 当前状况：所有测试用例被整块 `#ifndef USE_PADDLE_API … #endif` 包裹，
 *             Paddle 构建模式下该文件是空测试套件，无任何用例参与对比测试。
 * - 根本原因：被测函数 `torch::getTHPDtype(c10::ScalarType)` 是 PyTorch 的
 *             Python 解释器桥接 API，其返回值类型为 `THPDtype*`（Python C API
 *             数据类型指针）。Paddle 的 C++ 兼容层聚焦于算子执行与 C++
 * 侧接口对齐，
 *             **没有对应的 Python 解释器桥接层设计**，因此该函数族在 Paddle
 * compat 中既无头文件声明，也无运行时实现。
 * - 期望解决：`torch::getTHPDtype` 等 Python C API 桥接函数属于"Paddle 兼容层
 *             不打算对齐的范畴"，建议将本文件整体标记为
 * `libtorch_only`，并在测试 框架配置层（CMakeLists.txt）中将其排除出 Paddle
 * 构建的测试目标， 而非在代码中用 `#ifndef USE_PADDLE_API` 包裹所有用例。若
 * Paddle 未来 计划支持 Python C API 桥接，则需要： ① 创建 `torch/python.h`
 * 并声明 `getTHPDtype` 原型； ② 在 Paddle 的 Python
 * 扩展层提供该函数的分发实现。
 * =====================================================================================
 */
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

#ifndef USE_PADDLE_API

// 测试 torch::getTHPDtype 函数存在性
// 该函数将 c10::ScalarType 转换为 Python dtype 对象
TEST_F(PythonTest, GetTHPDtype) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();

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
      torch::getTHPDtype(dtype);
      file << "1 ";
    } catch (...) {
      file << "exception ";
    }
  }

  file.saveFile();
}

// 测试 torch::getTHPDtype 函数存在性 (简化版)
TEST_F(PythonTest, PyObjectToDtype) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();

  // 测试 getTHPDtype 函数可用
  try {
    torch::getTHPDtype(c10::ScalarType::Float);
    file << "1 ";
  } catch (...) {
    file << "exception ";
  }

  file.saveFile();
}

// 测试 torch 命名空间下的 getTHPDtype 存在性
TEST_F(PythonTest, NamespaceExists) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();

  // 验证 torch::getTHPDtype 可用
  // NOLINTNEXTLINE
  (void)torch::getTHPDtype;

  // 如果编译通过，说明函数存在
  file << "1 ";
  file.saveFile();
}

// 测试 getTHPDtype 与各种 ScalarType 的兼容性
TEST_F(PythonTest, GetTHPDtypeAllTypes) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();

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
