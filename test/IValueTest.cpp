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

}  // namespace test
}  // namespace at
