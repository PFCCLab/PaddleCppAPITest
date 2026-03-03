#include <ATen/ATen.h>
#include <ATen/core/Tensor.h>
#include <ATen/core/ivalue.h>
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

// None
TEST_F(IValueTest, None) {
  torch::IValue iv;
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();
  file << std::to_string(iv.is_none() ? 1 : 0) << " ";
  file << iv.type_string() << " ";
  file.saveFile();
}

// Bool
TEST_F(IValueTest, Bool) {
  torch::IValue iv_true(true);
  torch::IValue iv_false(false);
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << std::to_string(iv_true.is_bool() ? 1 : 0) << " ";
  file << std::to_string(iv_true.to_bool() ? 1 : 0) << " ";
  file << std::to_string(iv_false.to_bool() ? 1 : 0) << " ";
  file << iv_true.type_string() << " ";
  file.saveFile();
}

// Int
TEST_F(IValueTest, Int) {
  torch::IValue iv(42);
  torch::IValue iv64(static_cast<int64_t>(100000));
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << std::to_string(iv.is_int() ? 1 : 0) << " ";
  file << std::to_string(iv.to_int()) << " ";
  file << std::to_string(iv64.to_int()) << " ";
  file.saveFile();
}

// Double
TEST_F(IValueTest, Double) {
  torch::IValue iv(3.14);
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << std::to_string(iv.is_double() ? 1 : 0) << " ";
  file << std::to_string(iv.to_double()) << " ";
  file.saveFile();
}

// String
TEST_F(IValueTest, String) {
  torch::IValue iv(std::string("hello_world"));
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << std::to_string(iv.is_string() ? 1 : 0) << " ";
  file << iv.to_string() << " ";
  file.saveFile();
}

// String from const char*
TEST_F(IValueTest, StringFromCharPtr) {
  torch::IValue iv("test_string");
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << std::to_string(iv.is_string() ? 1 : 0) << " ";
  file << iv.to_string() << " ";
  file.saveFile();
}

// Tensor
TEST_F(IValueTest, Tensor) {
  at::Tensor t = at::zeros({2, 3}, at::kFloat);
  torch::IValue iv(t);
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << std::to_string(iv.is_tensor() ? 1 : 0) << " ";
  at::Tensor retrieved = iv.to_tensor();
  file << std::to_string(retrieved.dim()) << " ";
  file << std::to_string(retrieved.numel()) << " ";
  file.saveFile();
}

// List of ints
TEST_F(IValueTest, ListOfInts) {
  std::vector<int64_t> vec = {1, 2, 3, 4, 5};
  torch::IValue iv(vec);
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << std::to_string(iv.is_list() ? 1 : 0) << " ";
  const auto& list = iv.to_list();
  file << std::to_string(list.size()) << " ";
  for (const auto& item : list) {
    file << std::to_string(item.to_int()) << " ";
  }
  file.saveFile();
}

// List of doubles
TEST_F(IValueTest, ListOfDoubles) {
  std::vector<double> vec = {1.1, 2.2, 3.3};
  torch::IValue iv(vec);
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << std::to_string(iv.is_list() ? 1 : 0) << " ";
  const auto& list = iv.to_list();
  file << std::to_string(list.size()) << " ";
  file.saveFile();
}

// to<T>() 模板转换
TEST_F(IValueTest, ToTemplate) {
  torch::IValue iv_int(42);
  torch::IValue iv_double(3.14);
  torch::IValue iv_string(std::string("test"));
  torch::IValue iv_bool(true);

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << std::to_string(iv_int.to<int64_t>()) << " ";
  file << std::to_string(iv_double.to<double>()) << " ";
  file << iv_string.to<std::string>() << " ";
  file << std::to_string(iv_bool.to<bool>() ? 1 : 0) << " ";
  file.saveFile();
}

// Tuple
TEST_F(IValueTest, Tuple) {
  auto tup = std::make_tuple(42, 3.14, std::string("hello"));
  torch::IValue iv(tup);
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << std::to_string(iv.is_tuple() ? 1 : 0) << " ";
  const auto& t = iv.to_tuple();
  file << std::to_string(t.size()) << " ";
  file << std::to_string(t[0].to_int()) << " ";
  file.saveFile();
}

// Optional (with value)
TEST_F(IValueTest, OptionalWithValue) {
  std::optional<int64_t> opt = 99;
  torch::IValue iv(opt);
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << std::to_string(iv.is_int() ? 1 : 0) << " ";
  file << std::to_string(iv.to_int()) << " ";
  file.saveFile();
}

// Optional (nullopt)
TEST_F(IValueTest, OptionalNullopt) {
  std::optional<int64_t> opt = std::nullopt;
  torch::IValue iv(opt);
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << std::to_string(iv.is_none() ? 1 : 0) << " ";
  file.saveFile();
}

// ScalarType 转换
TEST_F(IValueTest, ScalarType) {
  torch::IValue iv(at::kFloat);
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << std::to_string(iv.is_int() ? 1 : 0) << " ";
  at::ScalarType st = iv.to_scalar_type();
  file << std::to_string(static_cast<int>(st)) << " ";
  file.saveFile();
}

// to_repr / type_string
TEST_F(IValueTest, ToRepr) {
  torch::IValue iv_none;
  torch::IValue iv_int(42);
  torch::IValue iv_str(std::string("hello"));

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << iv_none.to_repr() << " ";
  file << iv_int.to_repr() << " ";
  file << iv_str.to_repr() << " ";
  file.saveFile();
}

// try_to_xxx 方法
TEST_F(IValueTest, TryToMethods) {
  torch::IValue iv_int(42);
  torch::IValue iv_double(3.14);
  torch::IValue iv_str(std::string("test"));

  bool b_val;
  int i_val;
  double d_val;
  std::string s_val;

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  // int → bool (非零 = true)
  file << std::to_string(iv_int.try_to_bool(b_val) ? 1 : 0) << " ";
  file << std::to_string(b_val ? 1 : 0) << " ";
  // double → double
  file << std::to_string(iv_double.try_to_double(d_val) ? 1 : 0) << " ";
  file << std::to_string(d_val) << " ";
  // string → string
  file << std::to_string(iv_str.try_to_string(s_val) ? 1 : 0) << " ";
  file << s_val << " ";
  file.saveFile();
}

// intrusive_ptr / CustomClass
TEST_F(IValueTest, CustomClass) {
  class MyClass : public torch::CustomClassHolder {
   public:
    int value;
    explicit MyClass(int v) : value(v) {}
  };

  auto ptr = torch::make_intrusive<MyClass>(42);
  torch::IValue iv(ptr);

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << std::to_string(iv.is_custom_class() ? 1 : 0) << " ";
  auto retrieved = iv.to_custom_class<MyClass>();
  file << std::to_string(retrieved->value) << " ";
  file.saveFile();
}

}  // namespace test
}  // namespace at
