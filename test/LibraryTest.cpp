#include <ATen/ATen.h>
#include <gtest/gtest.h>
#include <torch/library.h>

#include <functional>
#include <string>
#include <vector>

#include "../src/file_manager.h"

extern paddle_api_test::ThreadSafeParam g_custom_param;

namespace at {
namespace test {

using paddle_api_test::FileManerger;
using paddle_api_test::ThreadSafeParam;

class LibraryTest : public ::testing::Test {
 protected:
  void SetUp() override {}
};

// [DIFF] 文件级说明：torch::Library / IValue
// 在两端架构差异较大（命名空间、方法名、注册体系）。

// 测试 torch::Library::Kind 枚举
TEST_F(LibraryTest, LibraryKindEnum) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();
  file << "LibraryKindEnum ";

  file << std::to_string(static_cast<int>(torch::Library::Kind::DEF)) << " ";
  file << std::to_string(static_cast<int>(torch::Library::Kind::IMPL)) << " ";
  file << std::to_string(static_cast<int>(torch::Library::Kind::FRAGMENT))
       << " ";
  file << "\n";
  file.saveFile();
}

// 测试 IValue 基本构造
TEST_F(LibraryTest, IValueBasicConstruction) {
  // [DIFF] 用例级差异：Paddle 用 torch::IValue + snake_case；Torch 用
  // c10::IValue + camelCase。
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "IValueBasicConstruction ";

#if USE_PADDLE_API
  // [DIFF] 问题行：Paddle 路径类型与方法名与 Torch
  // 路径不兼容，无法共用同一实现。 Paddle 兼容层使用 torch::IValue，方法名是
  // snake_case
  torch::IValue ival_int(42);
  torch::IValue ival_double(3.14);
  torch::IValue ival_string(std::string("test"));

  file << std::to_string(ival_int.is_int() ? 1 : 0) << " ";
  file << std::to_string(ival_double.is_double() ? 1 : 0) << " ";
  file << std::to_string(ival_string.is_string() ? 1 : 0) << " ";
#else
  // libtorch 使用 c10::IValue，方法名是 camelCase
  c10::IValue ival_int(42);
  c10::IValue ival_double(3.14);
  c10::IValue ival_string(std::string("test"));

  file << std::to_string(ival_int.isInt() ? 1 : 0) << " ";
  file << std::to_string(ival_double.isDouble() ? 1 : 0) << " ";
  file << std::to_string(ival_string.isString() ? 1 : 0) << " ";
#endif
  file << "\n";
  file.saveFile();
}

// 测试 IValue 从 vector 构造
TEST_F(LibraryTest, IValueVectorConstruction) {
  // [DIFF] 用例级差异：vector 元素类型在两端为 torch::IValue vs c10::IValue。
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "IValueVectorConstruction ";

#if USE_PADDLE_API
  std::vector<torch::IValue> args_vec = {
      torch::IValue(1), torch::IValue(2.5), torch::IValue(std::string("test"))};
#else
  std::vector<c10::IValue> args_vec = {
      c10::IValue(1), c10::IValue(2.5), c10::IValue(std::string("test"))};
#endif

  file << std::to_string(args_vec.size()) << " ";
  file << std::to_string(args_vec.empty() ? 0 : 1) << " ";
  file << "\n";
  file.saveFile();
}

// 测试 IValue get 方法
TEST_F(LibraryTest, IValueGet) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "IValueGet ";

#if USE_PADDLE_API
  torch::IValue ival_int(42);
  torch::IValue ival_double(3.14);

  try {
    int64_t int_val = ival_int.to_int();
    file << std::to_string(int_val) << " ";
  } catch (...) {
    file << "-1 ";
  }

  try {
    double double_val = ival_double.to_double();
    file << std::to_string(double_val) << " ";
  } catch (...) {
    file << "-1 ";
  }
#else
  c10::IValue ival_int(42);
  c10::IValue ival_double(3.14);

  try {
    int64_t int_val = ival_int.toInt();
    file << std::to_string(int_val) << " ";
  } catch (...) {
    file << "-1 ";
  }

  try {
    double double_val = ival_double.toDouble();
    file << std::to_string(double_val) << " ";
  } catch (...) {
    file << "-1 ";
  }
#endif

  file << "\n";
  file.saveFile();
}

// 测试 IValue is_none
TEST_F(LibraryTest, IValueIsNone) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "IValueIsNone ";

#if USE_PADDLE_API
  torch::IValue ival_none;
  torch::IValue ival_int(42);

  file << std::to_string(ival_none.is_none() ? 1 : 0) << " ";
  file << std::to_string(ival_int.is_none() ? 0 : 1) << " ";
#else
  c10::IValue ival_none;
  c10::IValue ival_int(42);

  file << std::to_string(ival_none.isNone() ? 1 : 0) << " ";
  file << std::to_string(ival_int.isNone() ? 0 : 1) << " ";
#endif
  file << "\n";
  file.saveFile();
}

// 测试 IValue 显式转换为 int64_t
TEST_F(LibraryTest, IValueSizeToInt64) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "IValueSizeToInt64 ";

  size_t sz = 42;
#if USE_PADDLE_API
  torch::IValue ival(static_cast<int64_t>(sz));
  file << std::to_string(ival.to_int()) << " ";
#else
  c10::IValue ival(static_cast<int64_t>(sz));
  file << std::to_string(ival.toInt()) << " ";
#endif
  file << "\n";
  file.saveFile();
}

// 测试 IValue 作为 Tensor
TEST_F(LibraryTest, IValueTensor) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "IValueTensor ";

  at::Tensor tensor = at::ones({3, 3});
#if USE_PADDLE_API
  torch::IValue ival(tensor);
  file << std::to_string(ival.is_tensor() ? 1 : 0) << " ";
#else
  c10::IValue ival(tensor);
  file << std::to_string(ival.isTensor() ? 1 : 0) << " ";
#endif
  file << "\n";
  file.saveFile();
}

// 测试 at::Tensor 操作
TEST_F(LibraryTest, TensorOperations) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "TensorOperations ";

  at::Tensor t1 = at::ones({2, 3});
  at::Tensor t2 = at::ones({2, 3});
#ifndef USE_PADDLE_API
  at::Tensor t3 = t1.add(t2);
  file << std::to_string(t3.numel()) << " ";
#else
  // Paddle 兼容层不支持 + 运算符和 .add() 方法
  file << "tensor_add_skipped ";
#endif
  file << "\n";
  file.saveFile();
}

// 测试 at::Device - 只在 libtorch 下
#ifndef USE_PADDLE_API
TEST_F(LibraryTest, DeviceTest) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "DeviceTest ";

  at::Device device(c10::DeviceType::CPU);
  file << std::to_string(device.type() == c10::DeviceType::CPU ? 1 : 0) << " ";
  file << "\n";
  file.saveFile();
}

// 测试 c10::TensorOptions - 只在 libtorch 下
TEST_F(LibraryTest, TensorOptionsTest) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "TensorOptionsTest ";

  auto opts =
      c10::TensorOptions().dtype(at::kFloat).device(c10::DeviceType::CPU);
  file << std::to_string(opts.dtype() == at::kFloat ? 1 : 0) << " ";
  file << "\n";
  file.saveFile();
}
#endif

#ifndef USE_PADDLE_API
// 测试 torch::Library 构造（不实际注册）
TEST_F(LibraryTest, LibraryConstruction) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "LibraryConstruction ";

  // 测试 Library 的各种方法
  torch::Library lib(torch::Library::DEF, "test_ns");
  lib.def("test_op(Tensor t) -> Tensor");

  auto fn = [](const at::Tensor& t) -> at::Tensor { return t; };
  auto fn_wrapper = [fn](const torch::FunctionArgs& args) -> torch::IValue {
    auto t = args.get<at::Tensor>(0);
    return torch::IValue(fn(t));
  };
  lib.impl("test_op", fn_wrapper);

  file << "1 ";
  file << "\n";
  file.saveFile();
}

// 测试 torch::CppFunction - 函数指针构造
TEST_F(LibraryTest, CppFunctionFromFunctionPointer) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "CppFunctionFromFunctionPointer ";

  // 定义一个简单的函数
  auto fn = [](const at::Tensor& t) -> at::Tensor { return t; };
  auto cpp_wrapper = [fn](const torch::FunctionArgs& args) -> torch::IValue {
    auto t = args.get<at::Tensor>(0);
    return torch::IValue(fn(t));
  };
  torch::CppFunction cpp_fn(cpp_wrapper);

  file << "1 ";
  file << "\n";
  file.saveFile();
}

#endif

#ifndef USE_PADDLE_API
// 测试 torch::CppFunction - makeFromBoxedKernel
TEST_F(LibraryTest, CppFunctionMakeFromBoxedKernel) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "CppFunctionMakeFromBoxedKernel ";

  // 使用 makeFromBoxedKernel 创建 CppFunction
  auto cpp_fn = torch::CppFunction::makeFromBoxedKernel(
      c10::BoxedKernel::makeFallthrough());

  file << "1 ";
  file << "\n";
  file.saveFile();
}

// 测试 torch::CppFunction::makeFallthrough
TEST_F(LibraryTest, CppFunctionMakeFallthrough) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "CppFunctionMakeFallthrough ";

  auto cpp_fn = torch::CppFunction::makeFallthrough();

  file << "1 ";
  file << "\n";
  file.saveFile();
}

// 测试 torch::CppFunction::makeNamedNotSupported
TEST_F(LibraryTest, CppFunctionMakeNamedNotSupported) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "CppFunctionMakeNamedNotSupported ";

  auto cpp_fn = torch::CppFunction::makeNamedNotSupported();

  file << "1 ";
  file << "\n";
  file.saveFile();
}

// 测试 torch::CppFunction::debug
TEST_F(LibraryTest, CppFunctionDebug) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "CppFunctionDebug ";

  auto fn = [](const at::Tensor& t) -> at::Tensor { return t; };
  auto dbg_wrapper = [fn](const torch::FunctionArgs& args) -> torch::IValue {
    auto t = args.get<at::Tensor>(0);
    return torch::IValue(fn(t));
  };
  torch::CppFunction cpp_fn =
      torch::CppFunction(dbg_wrapper).debug("test_debug_info");

  file << "1 ";
  file << "\n";
  file.saveFile();
}

// 测试 torch::dispatch - DispatchKey 版本
TEST_F(LibraryTest, DispatchWithDispatchKey) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "DispatchWithDispatchKey ";

  auto fn = [](const at::Tensor& t) -> at::Tensor { return t; };
  auto dispatch_wrapper =
      [fn](const torch::FunctionArgs& args) -> torch::IValue {
    auto t = args.get<at::Tensor>(0);
    return torch::IValue(fn(t));
  };

  // 使用 DispatchKey::CPU 测试 dispatch
  auto cpp_fn = torch::dispatch(torch::DispatchKey::CPU, dispatch_wrapper);

  file << "1 ";
  file << "\n";
  file.saveFile();
}

// 测试 torch::dispatch - DeviceType 版本
TEST_F(LibraryTest, DispatchWithDeviceType) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "DispatchWithDeviceType ";

  auto fn = [](const at::Tensor& t) -> at::Tensor { return t; };
  auto dispatch_wrapper2 =
      [fn](const torch::FunctionArgs& args) -> torch::IValue {
    auto t = args.get<at::Tensor>(0);
    return torch::IValue(fn(t));
  };

  // 使用 DeviceType::CPU 测试 dispatch
  auto cpp_fn2 = torch::dispatch(c10::DeviceType::CPU, dispatch_wrapper2);

  file << "1 ";
  file << "\n";
  file.saveFile();
}

// 测试 torch::schema - 从字符串构造
TEST_F(LibraryTest, SchemaFromString) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "SchemaFromString ";

  // 使用 torch::schema 从字符串构造
  c10::FunctionSchema schema =
      torch::schema("add(Tensor self, Tensor other) -> Tensor");

  file << schema.name() << " ";
  file << std::to_string(schema.arguments().size()) << " ";
  file << std::to_string(schema.returns().size()) << " ";
  file << "\n";
  file.saveFile();
}

// 测试 torch::schema - 带 AliasAnalysisKind
TEST_F(LibraryTest, SchemaWithAliasAnalysis) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "SchemaWithAliasAnalysis ";

  // 使用 torch::schema 带 AliasAnalysisKind
  c10::FunctionSchema schema = torch::schema(
      "foo(Tensor self) -> Tensor", c10::AliasAnalysisKind::PURE_FUNCTION);

  file << schema.name() << " ";
  file << "\n";
  file.saveFile();
}

// 测试 torch::detail::SelectiveStr
TEST_F(LibraryTest, SelectiveStrEnabled) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "SelectiveStrEnabled ";

  // 测试 SelectiveStr<true>
  torch::detail::SelectiveStr<true> sel_str("test_operator");

  // 转换为 const char*
  const char* name = sel_str.operator const char*();

  file << name << " ";
  file << "\n";
  file.saveFile();
}

// 测试 torch::detail::ClassNotSelected
TEST_F(LibraryTest, ClassNotSelected) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "ClassNotSelected ";

  // ClassNotSelected 的方法返回 *this，所以可以链式调用
  torch::detail::ClassNotSelected not_selected;
  not_selected.def("test_method").def_pickle("test_pickle");

  file << "1 ";
  file << "\n";
  file.saveFile();
}

// 测试 c10::DispatchKey 枚举值
TEST_F(LibraryTest, DispatchKeyEnum) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "DispatchKeyEnum ";

  // 测试各种 DispatchKey
  file << std::to_string(static_cast<int>(torch::DispatchKey::CPU)) << " ";
  file << std::to_string(static_cast<int>(torch::DispatchKey::CUDA)) << " ";
  file << std::to_string(static_cast<int>(torch::DispatchKey::XLA)) << " ";
  file << std::to_string(static_cast<int>(torch::DispatchKey::Meta)) << " ";
  file << std::to_string(static_cast<int>(torch::DispatchKey::CPU)) << " ";
  file << "\n";
  file.saveFile();
}

// 测试 c10::FunctionSchema 基本操作
TEST_F(LibraryTest, FunctionSchemaBasic) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "FunctionSchemaBasic ";

  c10::FunctionSchema schema =
      torch::schema("mul(Tensor self, Tensor other) -> Tensor");

  // 测试 schema 属性
  file << schema.name() << " ";
  file << schema.arguments().size() << " ";
  file << schema.returns().size() << " ";

  // 测试 argument
  if (schema.arguments().size() > 0) {
    file << schema.arguments()[0].name() << " ";
  }
  if (schema.returns().size() > 0) {
    file << schema.returns()[0].name() << " ";
  }

  file << "\n";
  file.saveFile();
}

// 测试 c10::FunctionSchema - 带默认参数
TEST_F(LibraryTest, FunctionSchemaWithDefault) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "FunctionSchemaWithDefault ";

  c10::FunctionSchema schema = torch::schema(
      "zeros(int[] size, ScalarType dtype=None, Device device=None, bool "
      "requires_grad=False) -> Tensor");

  file << schema.name() << " ";
  file << std::to_string(schema.arguments().size()) << " ";

  file << "\n";
  file.saveFile();
}

#endif

#ifndef USE_PADDLE_API

// 测试 c10::AliasAnalysisKind 枚举
TEST_F(LibraryTest, AliasAnalysisKindEnum) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "AliasAnalysisKindEnum ";

  file << std::to_string(static_cast<int>(c10::AliasAnalysisKind::FROM_SCHEMA))
       << " ";
  file << std::to_string(
              static_cast<int>(c10::AliasAnalysisKind::PURE_FUNCTION))
       << " ";
  file << std::to_string(static_cast<int>(c10::AliasAnalysisKind::CONSERVATIVE))
       << " ";

  file << "\n";
  file.saveFile();
}

// 测试 MAKE_TORCH_LIBRARY 宏
TEST_F(LibraryTest, MakeTorchLibraryMacro) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "MakeTorchLibraryMacro ";

  // 使用 MAKE_TORCH_LIBRARY 宏创建 Library（不实际注册）
  torch::Library lib = MAKE_TORCH_LIBRARY(test_ns);

  // 测试 Library 的各种方法调用覆盖
  lib.def("test_op2(Tensor x) -> Tensor");

  // 测试 Library::Kind::DEF
  file << std::to_string(static_cast<int>(lib.kind_)) << " ";

  file << "\n";
  file.saveFile();
}

// 测试 MAKE_TORCH_LIBRARY_IMPL 宏
TEST_F(LibraryTest, MakeTorchLibraryImplMacro) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "MakeTorchLibraryImplMacro ";

  // 使用 MAKE_TORCH_LIBRARY_IMPL 宏创建 Library（不实际注册）
  torch::Library lib = MAKE_TORCH_LIBRARY_IMPL(test_ns, CPU);

  // 测试 Library::impl 的另一种宏调用情况
  auto fn = [](const at::Tensor& t) -> at::Tensor { return t; };
  auto impl_wrapper = [fn](const torch::FunctionArgs& args) -> torch::IValue {
    auto t = args.get<at::Tensor>(0);
    return torch::IValue(fn(t));
  };
  lib.impl("test_op2", impl_wrapper);

  // 测试 fallback
  lib.fallback(torch::CppFunction::makeFallthrough());

  // 测试 Library::Kind::IMPL
  file << std::to_string(static_cast<int>(lib.kind_)) << " ";

  file << "\n";
  file.saveFile();
}

// 测试 c10::OperatorName 构造
TEST_F(LibraryTest, OperatorNameConstruction) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "OperatorNameConstruction ";

  c10::OperatorName op_name;
  op_name.name = "test_op";
  op_name.overload_name = "test_overload";

  file << op_name.name << " ";
  file << op_name.overload_name << " ";

  file << "\n";
  file.saveFile();
}

// 测试 c10::FunctionSchema argument 类型
TEST_F(LibraryTest, FunctionSchemaArgumentTypes) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "FunctionSchemaArgumentTypes ";

  c10::FunctionSchema schema =
      torch::schema("test_op(Tensor t, int count, float rate) -> Tensor");

  // 检查参数类型
  for (size_t i = 0; i < schema.arguments().size(); ++i) {
    const auto& arg = schema.arguments()[i];
    file << arg.type->kind() << " ";
  }

  file << "\n";
  file.saveFile();
}

// 测试 c10::FunctionSchema 返回类型
TEST_F(LibraryTest, FunctionSchemaReturnType) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "FunctionSchemaReturnType ";

  c10::FunctionSchema schema1 = torch::schema("op1() -> Tensor");
  c10::FunctionSchema schema2 = torch::schema("op2() -> int");
  c10::FunctionSchema schema3 = torch::schema("op3() -> float");

  // 检查返回类型
  file << schema1.returns()[0].type->kind() << " ";
  file << schema2.returns()[0].type->kind() << " ";
  file << schema3.returns()[0].type->kind() << " ";

  file << "\n";
  file.saveFile();
}

// ----------------------------------------------------
// 新增的补全覆盖率测试
// ----------------------------------------------------

// 测试 torch::arg (函数参数定义)
TEST_F(LibraryTest, ArgTest) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "ArgTest ";

  torch::arg my_arg("x");
  my_arg = torch::arg::none();

  file << "1 ";
  file << "\n";
  file.saveFile();
}

// 测试 torch::FunctionArgs
TEST_F(LibraryTest, FunctionArgsTest) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "FunctionArgsTest ";

  torch::FunctionArgs args1;          // 默认构造
  torch::FunctionArgs args2(1, 2.0);  // 模板参数构造

  std::vector<torch::IValue> vec = {torch::IValue(3)};
  torch::FunctionArgs args3 = torch::FunctionArgs::from_vector(vec);

  args1.add_arg(4);
  const std::vector<torch::IValue>& vals = args1.get();

  file << vals.size() << " " << args2.get().size() << " " << args3.get().size()
       << " ";
  file << "\n";
  file.saveFile();
}

// 测试 torch::FunctionResult
TEST_F(LibraryTest, FunctionResultTest) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "FunctionResultTest ";

  torch::FunctionResult r1;      // 默认构造
  torch::FunctionResult r2(42);  // 值构造

  torch::IValue iv(3.14);
  torch::FunctionResult r3(iv);                   // const引用构造
  torch::FunctionResult r4(torch::IValue(2.71));  // 右值构造

  if (r2.has_value()) {
    file << r2.get_value().is_int()
         << " ";  // 注意Paddle用 is_int, 若是裸测这里为了避免编译错误用 auto
    file << r2.get<int>() << " ";
  }

  torch::FunctionResult rv = torch::FunctionResult::void_result();

  // to_string() 方法
  std::string s = r2.to_string();

  file << "1 ";
  file << "\n";
  file.saveFile();
}

// 新增测试: FunctionArgs::get<T>(index) - 模板方法
TEST_F(LibraryTest, FunctionArgsGetTemplate) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "FunctionArgsGetTemplate ";

#if USE_PADDLE_API
  torch::FunctionArgs args(1, 2.5, std::string("test"));

  // 测试模板方法 get<T>(index)
  int val0 = args.get<int>(0);
  double val1 = args.get<double>(1);
  std::string val2 = args.get<std::string>(2);

  file << val0 << " " << val1 << " " << val2.length() << " ";
#else
  // libtorch 没有 FunctionArgs，使用占位输出
  file << "-1 -1 -1 ";
#endif
  file << "\n";
  file.saveFile();
}

// 新增测试: FunctionArgs::empty
TEST_F(LibraryTest, FunctionArgsEmpty) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "FunctionArgsEmpty ";

#if USE_PADDLE_API
  torch::FunctionArgs empty_args;
  torch::FunctionArgs non_empty_args(1, 2);

  file << (empty_args.empty() ? 1 : 0) << " ";
  file << (non_empty_args.empty() ? 0 : 1) << " ";
#else
  file << "-1 -1 ";
#endif
  file << "\n";
  file.saveFile();
}

// 新增测试: FunctionArgs::begin/end 迭代器
TEST_F(LibraryTest, FunctionArgsIterator) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "FunctionArgsIterator ";

#if USE_PADDLE_API
  torch::FunctionArgs args(10, 20, 30);

  // 测试迭代器
  auto iter_begin = args.begin();
  auto iter_end = args.end();

  // 计算迭代器距离
  size_t distance = 0;
  for (auto it = args.begin(); it != args.end(); ++it) {
    distance++;
  }

  // 获取第一个和最后一个元素
  int first_val = args.get<int>(0);
  int last_val = args.get<int>(distance - 1);

  file << distance << " " << first_val << " " << last_val << " ";
#else
  file << "-1 -1 -1 ";
#endif
  file << "\n";
  file.saveFile();
}

// 新增测试: FunctionResult::get<T>() - 模板方法
TEST_F(LibraryTest, FunctionResultGetTemplate) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "FunctionResultGetTemplate ";

#if USE_PADDLE_API
  torch::FunctionResult r_int(42);
  torch::FunctionResult r_double(3.14);
  torch::FunctionResult r_string(std::string("hello"));

  // 测试模板方法 get<T>()
  int int_val = r_int.get<int>();
  double double_val = r_double.get<double>();
  std::string string_val = r_string.get<std::string>();

  file << int_val << " " << double_val << " " << string_val.length() << " ";
#else
  file << "-1 -1 -1 ";
#endif
  file << "\n";
  file.saveFile();
}

// 新增测试: FunctionResult::has_value
TEST_F(LibraryTest, FunctionResultHasValue) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "FunctionResultHasValue ";

#if USE_PADDLE_API
  torch::FunctionResult r_with_value(42);
  torch::FunctionResult r_void = torch::FunctionResult::void_result();

  file << (r_with_value.has_value() ? 1 : 0) << " ";
  file << (r_void.has_value() ? 0 : 1) << " ";
#else
  file << "-1 -1 ";
#endif
  file << "\n";
  file.saveFile();
}

// 测试 CppFunction Call 方法
TEST_F(LibraryTest, CppFunctionCallTest) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "CppFunctionCallTest ";

  auto fn = [](int x) -> int { return x + 1; };
  auto call_wrapper = [fn](const torch::FunctionArgs& args) -> torch::IValue {
    int x = args.get<int>(0);
    return torch::IValue(fn(x));
  };
  torch::CppFunction cpp_fn(call_wrapper);

  bool valid = cpp_fn.valid();

  torch::FunctionResult r1 = cpp_fn.call(41);

  torch::FunctionArgs args;
  args.add_arg(41);
  torch::FunctionResult r2 = cpp_fn.call_with_args(args);

  file << std::to_string(valid ? 1 : 0) << " " << r1.get<int>() << " "
       << r2.get<int>() << " ";
  file << "\n";
  file.saveFile();
}

class MyCustomHolder : public torch::CustomClassHolder {
 public:
  MyCustomHolder() {}
  int my_method() { return 42; }
  static int my_static_method() { return 43; }
};

// 测试 ClassRegistry / OperatorRegistry 等全局注册表相关的占位调用
TEST_F(LibraryTest, GlobalRegistryTest) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "GlobalRegistryTest ";

  // 测试 ClassRegistration 构造
  torch::ClassRegistration class_reg("my_ns", "MyClass");

  // ClassRegistry 测试
  torch::ClassRegistry& cls_reg = torch::ClassRegistry::instance();
  cls_reg.register_class("my_ns", "MyClass");

  auto dummy_fn = []() -> int { return 1; };
  auto dummy_wrapper =
      [](const torch::FunctionArgs& /*args*/) -> torch::IValue {
    return torch::IValue(1);
  };
  cls_reg.register_constructor("my_ns::MyClass",
                               torch::CppFunction(dummy_wrapper));
  cls_reg.register_method(
      "my_ns::MyClass", "my_m", torch::CppFunction(dummy_wrapper));
  cls_reg.register_static_method(
      "my_ns::MyClass", "my_s", torch::CppFunction(dummy_wrapper));

  cls_reg.has_class("my_ns::MyClass");
  cls_reg.has_method("my_ns::MyClass", "my_m");
  cls_reg.has_static_method("my_ns::MyClass", "my_s");

  torch::FunctionArgs args;
  cls_reg.call_constructor_with_args("my_ns::MyClass", args);
  cls_reg.call_method_with_args("my_ns::MyClass", "my_m", args);
  cls_reg.call_static_method_with_args("my_ns::MyClass", "my_s", args);
  cls_reg.print_all_classes();

  // 测试 OperatorRegistration 构造
  torch::OperatorRegistration op_reg("my_ns::my_op",
                                     "my_ns::my_op() -> Tensor");

  // OperatorRegistry 测试
  torch::OperatorRegistry& op_registry = torch::OperatorRegistry::instance();
  op_registry.register_schema("my_ns::my_op", "my_ns::my_op() -> Tensor");
  op_registry.register_implementation("my_ns::my_op",
                                      torch::DispatchKey::CPU,
                                      torch::CppFunction(dummy_wrapper));
  op_registry.has_operator("my_ns::my_op");
  op_registry.find_operator("my_ns::my_op");

  // class_<T> 和 init, def_static 测试
  torch::class_<MyCustomHolder> c("my_ns2", "MyCustomHolder");
  c.def(torch::init<>());
  c.def_static("my_static_method",
               []() { return MyCustomHolder::my_static_method(); });

  // invoke_function / invoke_member_function
  auto f = [](int x) { return x; };
  torch::FunctionArgs func_args;
  func_args.add_arg(10);
  torch::invoke_function(f, func_args);

  MyCustomHolder holder;
  torch::FunctionArgs empty_args;
  torch::invoke_member_function(
      &MyCustomHolder::my_method, &holder, empty_args);

  // Library 的一些扩展方法测试
  torch::Library lib(torch::Library::DEF, "test_ns3");
  lib.class_<MyCustomHolder>("MyCustomHolder2");
  lib.print_info();

  // TorchLibraryInit 测试
  auto init_fn = [](torch::Library& l) {};
  torch::detail::TorchLibraryInit tli(
      torch::Library::DEF, init_fn, "my_ns4", std::nullopt, "file.cpp", 1);

  // dispatch_key_to_string
  torch::dispatch_key_to_string(torch::DispatchKey::CPU);

  file << "1 ";
  file << "\n";
  file.saveFile();
}

#endif  // USE_PADDLE_API

}  // namespace test
}  // namespace at
