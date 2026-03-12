/*
 * =====================================================================================
 * @brief: 兼容性对齐审计报告
 *
 * [异常点 1]
 * - 测试用例：整个文件的公共头文件引入部分
 * - 当前状况：`#include <ATen/core/ivalue.h>` 被 `#ifndef USE_PADDLE_API`
 * 宏包裹， 在 Paddle 构建模式下该头文件完全不被包含。
 * - 根本原因：Paddle compat 中虽存在 `ATen/core/ivalue.h`，但其中将 `IValue` 类
 *             定义在 `torch` 命名空间（`torch::IValue`），而 libtorch
 * 的同路径头文件 将其定义在 `c10` 命名空间（`c10::IValue`）。若同时 `#include`
 * 该头文件 并使用 `c10::IValue`，在 Paddle
 * 构建模式下将发生编译错误（符号未定义）。
 * - 期望解决：在 Paddle compat 的 `ATen/core/ivalue.h` 中，将 `IValue`
 * 同时暴露在 `c10` 命名空间下（即添加 `namespace c10 { using ::torch::IValue;
 * }` 或将原生实现移入 `c10` 命名空间），使两库的 `c10::IValue` 符号保持同一
 *             命名空间来源。届时可移除该 `#ifndef USE_PADDLE_API` 宏。
 *
 * [异常点 2]
 * - 测试用例：None / Bool / Int / Double / String / StringFromCharPtr /
 *             Tensor / ListOfInts / ListOfDoubles / ToTemplate / Tuple /
 *             ScalarType / Identity（共 13 个 TEST_F，全部被禁用）
 * - 当前状况：所有测试用例被一整块 `#ifndef USE_PADDLE_API … #endif` 包裹，
 *             Paddle 构建模式下该文件实际上是一个空测试套件。
 * - 根本原因：① 测试代码统一使用 `c10::IValue`，而 Paddle compat 中仅有
 *             `torch::IValue`，两者命名空间不同，直接导致编译失败。
 *             ② `ListOfInts` 和 `ListOfDoubles` 测试中调用了
 * `c10::List<int64_t>` 和 `c10::List<double>`，Paddle compat 未提供
 * `c10::List<T>` 模板类 （其实现仅有 `std::vector` 形式的
 * `GenericList`，无公开的 `c10::List<T>` 类型别名或具化）。 ③ `None` 测试中以
 * `iv.to<std::string>()` 检测空值行为：libtorch 中 对 `None` IValue 调用
 * `to<std::string>()` 会抛出异常，而 Paddle compat 的 `generic_to` 对 `None`
 * 的行为未必相同，存在语义偏差风险。
 * - 期望解决：① 将 Paddle compat 的 `IValue` 同时暴露在 `c10`
 * 命名空间（见异常点 1）； ② 在 Paddle compat 中补充 `c10::List<T>`
 * 的类型别名（可 typedef 为 `std::vector<T>` 或提供同名模板包装），使
 * `ListOfInts` 和 `ListOfDoubles` 测试可以统一编译； ③ 统一 `None` IValue 的
 * `to<std::string>()` 异常行为语义，或将 `None` 测试改为对 `is_none()`
 * 的直接断言，规避边界语义差异。 完成上述修改后，可移除整个 `#ifndef
 * USE_PADDLE_API` 宏块。
 * =====================================================================================
 */
#include <ATen/ATen.h>
#include <ATen/core/Tensor.h>
#ifndef USE_PADDLE_API
#include <ATen/core/ivalue.h>
#endif
#include <ATen/ops/zeros.h>
#include <gtest/gtest.h>

#include <string>
#include <tuple>
#include <vector>

#include "../src/file_manager.h"

extern paddle_api_test::ThreadSafeParam g_custom_param;

namespace at {
namespace test {

using paddle_api_test::FileManerger;
using paddle_api_test::ThreadSafeParam;

class IValueTest : public ::testing::Test {
 protected:
  void SetUp() override {}
};

#ifndef USE_PADDLE_API

// None
TEST_F(IValueTest, None) {
  // Use default constructor for None
  auto iv = c10::IValue();
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();
  // Check if it's None using to<T>() - None to anything returns false
  file << std::to_string(iv.to<std::string>().empty() ? 1 : 0) << " ";
  file.saveFile();
}

// Bool
TEST_F(IValueTest, Bool) {
  auto iv_true = c10::IValue(true);
  auto iv_false = c10::IValue(false);
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  // Use to<bool>() to extract values
  file << std::to_string(iv_true.to<bool>() ? 1 : 0) << " ";
  file << std::to_string(iv_false.to<bool>() ? 1 : 0) << " ";
  file.saveFile();
}

// Int
TEST_F(IValueTest, Int) {
  auto iv = c10::IValue(42);
  auto iv64 = c10::IValue(static_cast<int64_t>(100000));
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  // Use to<int64_t>() to extract values
  file << std::to_string(iv.to<int64_t>()) << " ";
  file << std::to_string(iv64.to<int64_t>()) << " ";
  file.saveFile();
}

// Double
TEST_F(IValueTest, Double) {
  auto iv = c10::IValue(3.14);
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << std::to_string(iv.to<double>()) << " ";
  file.saveFile();
}

// String (from std::string)
TEST_F(IValueTest, String) {
  auto iv = c10::IValue(std::string("hello_world"));
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << iv.to<std::string>() << " ";
  file.saveFile();
}

// String (from const char*)
TEST_F(IValueTest, StringFromCharPtr) {
  auto iv = c10::IValue("test_string");
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << iv.to<std::string>() << " ";
  file.saveFile();
}

// Tensor
TEST_F(IValueTest, Tensor) {
  at::Tensor t = at::zeros({3, 4});
  auto iv = c10::IValue(t);
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  // Use to<at::Tensor>() to extract
  at::Tensor retrieved = iv.to<at::Tensor>();
  file << std::to_string(retrieved.numel()) << " ";
  file.saveFile();
}

// List of ints
TEST_F(IValueTest, ListOfInts) {
  std::vector<int64_t> vec = {1, 2, 3, 4, 5};
  auto iv = c10::IValue(vec);
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  auto list = iv.to<c10::List<int64_t>>();
  for (size_t i = 0; i < list.size() && i < 3; i++) {
    file << std::to_string(list[i]) << " ";
  }
  file.saveFile();
}

// List of doubles
TEST_F(IValueTest, ListOfDoubles) {
  std::vector<double> vec = {1.1, 2.2, 3.3};
  auto iv = c10::IValue(vec);
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  auto list = iv.to<c10::List<double>>();
  for (size_t i = 0; i < list.size() && i < 2; i++) {
    file << std::to_string(list[i]) << " ";
  }
  file.saveFile();
}

// to<T> template method
TEST_F(IValueTest, ToTemplate) {
  auto iv_int = c10::IValue(42);
  auto iv_double = c10::IValue(3.14);
  auto iv_string = c10::IValue(std::string("test"));
  auto iv_bool = c10::IValue(true);
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << std::to_string(iv_int.to<int64_t>()) << " ";
  file << std::to_string(iv_double.to<double>()) << " ";
  file << iv_string.to<std::string>() << " ";
  file << std::to_string(iv_bool.to<bool>() ? 1 : 0) << " ";
  file.saveFile();
}

// Tuple - use to<T> with std::tuple
TEST_F(IValueTest, Tuple) {
  std::tuple<int64_t, double, std::string> tup(1, 2.5, "hello");
  auto iv = c10::IValue(tup);
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  auto result = iv.to<std::tuple<int64_t, double, std::string>>();
  file << std::to_string(std::get<0>(result)) << " ";
  file << std::to_string(std::get<1>(result)) << " ";
  file << std::get<2>(result) << " ";
  file.saveFile();
}

// ScalarType - construct from ScalarType
TEST_F(IValueTest, ScalarType) {
  auto iv = c10::IValue(at::kFloat);
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  // Use to<at::ScalarType>() to extract
  auto st = iv.to<at::ScalarType>();
  file << std::to_string(static_cast<int>(st)) << " ";
  file.saveFile();
}

// IValue identity test
TEST_F(IValueTest, Identity) {
  auto iv_int = c10::IValue(42);
  auto iv_double = c10::IValue(3.14);
  auto iv_string = c10::IValue(std::string("test"));
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  // Just verify we can create and extract various types
  file << std::to_string(iv_int.to<int64_t>() == 42 ? 1 : 0) << " ";
  file << std::to_string(iv_double.to<double>() > 3.0 ? 1 : 0) << " ";
  file << std::to_string(iv_string.to<std::string>() == "test" ? 1 : 0) << " ";
  file.saveFile();
}

#endif  // USE_PADDLE_API

}  // namespace test
}  // namespace at
