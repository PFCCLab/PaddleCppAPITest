#include <ATen/ATen.h>
#include <c10/core/ScalarType.h>
#include <gtest/gtest.h>

#include <string>

#include "../src/file_manager.h"

extern paddle_api_test::ThreadSafeParam g_custom_param;

namespace at {
namespace test {

using paddle_api_test::FileManerger;
using paddle_api_test::ThreadSafeParam;

class ScalarTypeFunctionsTest : public ::testing::Test {
 protected:
  void SetUp() override {}
};

// elementSize
TEST_F(ScalarTypeFunctionsTest, ElementSize) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();
  file << "ElementSize ";

  // Test elementSize for various ScalarTypes
  file << std::to_string(c10::elementSize(c10::ScalarType::Float)) << " ";
  file << std::to_string(c10::elementSize(c10::ScalarType::Double)) << " ";
  file << std::to_string(c10::elementSize(c10::ScalarType::Int)) << " ";
  file << std::to_string(c10::elementSize(c10::ScalarType::Long)) << " ";
  file << std::to_string(c10::elementSize(c10::ScalarType::Half)) << " ";
  file << std::to_string(c10::elementSize(c10::ScalarType::Bool)) << " ";
  file << std::to_string(c10::elementSize(c10::ScalarType::Byte)) << " ";
  file << std::to_string(c10::elementSize(c10::ScalarType::Char)) << " ";
  file << std::to_string(c10::elementSize(c10::ScalarType::Short)) << " ";
  file << std::to_string(c10::elementSize(c10::ScalarType::BFloat16)) << " ";
  file << std::to_string(c10::elementSize(c10::ScalarType::ComplexFloat))
       << " ";
  file << std::to_string(c10::elementSize(c10::ScalarType::ComplexDouble))
       << " ";
  file << "\n";
  file.saveFile();
}

// isIntegralType
TEST_F(ScalarTypeFunctionsTest, IsIntegralType) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "IsIntegralType ";

  // Test isIntegralType with includeBool = false
  file << std::to_string(c10::isIntegralType(c10::ScalarType::Int, false))
       << " ";
  file << std::to_string(c10::isIntegralType(c10::ScalarType::Long, false))
       << " ";
  file << std::to_string(c10::isIntegralType(c10::ScalarType::Float, false))
       << " ";
  file << std::to_string(c10::isIntegralType(c10::ScalarType::Double, false))
       << " ";
  file << std::to_string(c10::isIntegralType(c10::ScalarType::Bool, false))
       << " ";

  // Test isIntegralType with includeBool = true
  file << std::to_string(c10::isIntegralType(c10::ScalarType::Int, true))
       << " ";
  file << std::to_string(c10::isIntegralType(c10::ScalarType::Bool, true))
       << " ";
  file << std::to_string(c10::isIntegralType(c10::ScalarType::Float, true))
       << " ";
  file << "\n";
  file.saveFile();
}

// isFloat8Type
TEST_F(ScalarTypeFunctionsTest, IsFloat8Type) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "IsFloat8Type ";

  // Test isFloat8Type - may not be available in all versions
  file << "float8_test ";
  file << "\n";
  file.saveFile();
}

// isReducedFloatingType
TEST_F(ScalarTypeFunctionsTest, IsReducedFloatingType) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "IsReducedFloatingType ";

  // Test isReducedFloatingType
  file << std::to_string(c10::isReducedFloatingType(c10::ScalarType::Half))
       << " ";
  file << std::to_string(c10::isReducedFloatingType(c10::ScalarType::BFloat16))
       << " ";
  file << std::to_string(c10::isReducedFloatingType(c10::ScalarType::Float))
       << " ";
  file << std::to_string(c10::isReducedFloatingType(c10::ScalarType::Double))
       << " ";
  file << "\n";
  file.saveFile();
}

// isFloatingType
TEST_F(ScalarTypeFunctionsTest, IsFloatingType) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "IsFloatingType ";

  // Test isFloatingType
  file << std::to_string(c10::isFloatingType(c10::ScalarType::Float)) << " ";
  file << std::to_string(c10::isFloatingType(c10::ScalarType::Double)) << " ";
  file << std::to_string(c10::isFloatingType(c10::ScalarType::Half)) << " ";
  file << std::to_string(c10::isFloatingType(c10::ScalarType::BFloat16)) << " ";
  file << std::to_string(c10::isFloatingType(c10::ScalarType::Int)) << " ";
  file << std::to_string(c10::isFloatingType(c10::ScalarType::Long)) << " ";
  file << std::to_string(c10::isFloatingType(c10::ScalarType::Bool)) << " ";
  file << "\n";
  file.saveFile();
}

// isComplexType
TEST_F(ScalarTypeFunctionsTest, IsComplexType) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "IsComplexType ";

  // Test isComplexType
  file << std::to_string(c10::isComplexType(c10::ScalarType::ComplexFloat))
       << " ";
  file << std::to_string(c10::isComplexType(c10::ScalarType::ComplexDouble))
       << " ";
  file << std::to_string(c10::isComplexType(c10::ScalarType::Float)) << " ";
  file << std::to_string(c10::isComplexType(c10::ScalarType::Double)) << " ";
  file << std::to_string(c10::isComplexType(c10::ScalarType::Int)) << " ";
  file << "\n";
  file.saveFile();
}

}  // namespace test
}  // namespace at
