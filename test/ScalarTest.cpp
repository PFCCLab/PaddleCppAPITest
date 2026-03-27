#include <ATen/ATen.h>
#include <c10/core/Scalar.h>
#include <gtest/gtest.h>

#include <string>

#include "../src/file_manager.h"

extern paddle_api_test::ThreadSafeParam g_custom_param;

namespace at {
namespace test {

using paddle_api_test::FileManerger;
using paddle_api_test::ThreadSafeParam;

class ScalarTest : public ::testing::Test {
 protected:
  void SetUp() override {}
};

// 从 float 构造
TEST_F(ScalarTest, FromFloat) {
  c10::Scalar s(3.14f);
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();
  file << "FromFloat ";
  file << std::to_string(s.to<float>()) << " ";
  file << std::to_string(s.to<double>()) << " ";
  file << "\n";
  file.saveFile();
}

// 从 double 构造
TEST_F(ScalarTest, FromDouble) {
  c10::Scalar s(2.718281828);
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "FromDouble ";
  file << std::to_string(s.to<double>()) << " ";
  file << "\n";
  file.saveFile();
}

// 从 int 构造
TEST_F(ScalarTest, FromInt) {
  c10::Scalar s(42);
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "FromInt ";
  file << std::to_string(s.to<int>()) << " ";
  file << std::to_string(s.to<int64_t>()) << " ";
  file << std::to_string(s.to<float>()) << " ";
  file << "\n";
  file.saveFile();
}

// 从 int64_t 构造
TEST_F(ScalarTest, FromInt64) {
  c10::Scalar s(static_cast<int64_t>(100000));
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "FromInt64 ";
  file << std::to_string(s.to<int64_t>()) << " ";
  file << "\n";
  file.saveFile();
}

// 从 bool 构造
TEST_F(ScalarTest, FromBool) {
  c10::Scalar s_true(true);
  c10::Scalar s_false(false);
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "FromBool ";
  file << std::to_string(s_true.to<bool>() ? 1 : 0) << " ";
  file << std::to_string(s_false.to<bool>() ? 1 : 0) << " ";
  file << "\n";
  file.saveFile();
}

// 负数
TEST_F(ScalarTest, NegativeValues) {
  c10::Scalar s_neg_int(-42);
  c10::Scalar s_neg_float(-3.14f);
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "NegativeValues ";
  file << std::to_string(s_neg_int.to<int>()) << " ";
  file << std::to_string(s_neg_float.to<float>()) << " ";
  file << "\n";
  file.saveFile();
}

// 零
TEST_F(ScalarTest, ZeroValues) {
  c10::Scalar s_zero_int(0);
  c10::Scalar s_zero_float(0.0f);
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "ZeroValues ";
  file << std::to_string(s_zero_int.to<int>()) << " ";
  file << std::to_string(s_zero_float.to<float>()) << " ";
  file << "\n";
  file.saveFile();
}

// 极值
TEST_F(ScalarTest, ExtremeValues) {
  c10::Scalar s_max_int(std::numeric_limits<int>::max());
  c10::Scalar s_min_int(std::numeric_limits<int>::min());
  c10::Scalar s_max_float(std::numeric_limits<float>::max());

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "ExtremeValues ";
  file << std::to_string(s_max_int.to<int>()) << " ";
  file << std::to_string(s_min_int.to<int>()) << " ";
  file << std::to_string(s_max_float.to<float>()) << " ";
  file << "\n";
  file.saveFile();
}

// at / c10 命名空间别名
TEST_F(ScalarTest, NamespaceAliases) {
  c10::Scalar s1(10);
  c10::Scalar s2(20);
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "NamespaceAliases ";
  file << std::to_string(s1.to<int>()) << " ";
  file << std::to_string(s2.to<int>()) << " ";
  file << "\n";
  file.saveFile();
}

}  // namespace test
}  // namespace at
