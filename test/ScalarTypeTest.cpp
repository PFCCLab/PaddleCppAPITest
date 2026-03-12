#include <ATen/ATen.h>
#include <ATen/core/Tensor.h>
#include <ATen/ops/ones.h>
#include <gtest/gtest.h>
#include <torch/all.h>

#include <string>
#include <vector>

#include "../src/file_manager.h"

extern paddle_api_test::ThreadSafeParam g_custom_param;

namespace at {
namespace test {

using paddle_api_test::FileManerger;
using paddle_api_test::ThreadSafeParam;

class ScalarTypeTest : public ::testing::Test {
 protected:
  void SetUp() override {
    std::vector<int64_t> shape = {2, 3, 4};
    tensor = at::ones(shape, at::kFloat);
  }

  at::Tensor tensor;
};

// 测试 is_complex
TEST_F(ScalarTypeTest, IsComplex) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();

  // Float tensor should not be complex
  file << std::to_string(tensor.is_complex()) << " ";

  // Test with actual complex tensor
  at::Tensor complex_tensor =
      at::ones({2, 3}, at::TensorOptions().dtype(at::kComplexFloat));
  file << std::to_string(complex_tensor.is_complex()) << " ";

  at::Tensor complex_double_tensor =
      at::ones({2, 3}, at::TensorOptions().dtype(at::kComplexDouble));
  file << std::to_string(complex_double_tensor.is_complex()) << " ";
  file.saveFile();
}

// 测试 is_floating_point
TEST_F(ScalarTypeTest, IsFloatingPoint) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();

  // Float tensor should be floating point
  file << std::to_string(tensor.is_floating_point()) << " ";

  // Test with double tensor
  at::Tensor double_tensor =
      at::ones({2, 3}, at::TensorOptions().dtype(at::kDouble));
  file << std::to_string(double_tensor.is_floating_point()) << " ";

  // Test with integer tensor
  at::Tensor int_tensor = at::ones({2, 3}, at::TensorOptions().dtype(at::kInt));
  file << std::to_string(int_tensor.is_floating_point()) << " ";

  // Test with long tensor
  at::Tensor long_tensor =
      at::ones({2, 3}, at::TensorOptions().dtype(at::kLong));
  file << std::to_string(long_tensor.is_floating_point()) << " ";
  file.saveFile();
}

// 测试 is_signed
TEST_F(ScalarTypeTest, IsSigned) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();

  // Float tensor should be signed
  file << std::to_string(tensor.is_signed()) << " ";

  // Test with int tensor (signed)
  at::Tensor int_tensor = at::ones({2, 3}, at::TensorOptions().dtype(at::kInt));
  file << std::to_string(int_tensor.is_signed()) << " ";

  // Test with long tensor (signed)
  at::Tensor long_tensor =
      at::ones({2, 3}, at::TensorOptions().dtype(at::kLong));
  file << std::to_string(long_tensor.is_signed()) << " ";

  // Test with byte tensor (unsigned)
  at::Tensor byte_tensor =
      at::ones({2, 3}, at::TensorOptions().dtype(at::kByte));
  file << std::to_string(byte_tensor.is_signed()) << " ";

  // Test with bool tensor (unsigned)
  at::Tensor bool_tensor =
      at::ones({2, 3}, at::TensorOptions().dtype(at::kBool));
  file << std::to_string(bool_tensor.is_signed()) << " ";
  file.saveFile();
}

// 测试 c10::toString
TEST_F(ScalarTypeTest, ToString) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();

  // Test toString for various ScalarTypes
  file << c10::toString(c10::ScalarType::Byte) << " ";
  file << c10::toString(c10::ScalarType::Char) << " ";
  file << c10::toString(c10::ScalarType::Short) << " ";
  file << c10::toString(c10::ScalarType::Int) << " ";
  file << c10::toString(c10::ScalarType::Long) << " ";
  file << c10::toString(c10::ScalarType::Half) << " ";
  file << c10::toString(c10::ScalarType::Float) << " ";
  file << c10::toString(c10::ScalarType::Double) << " ";
  file << c10::toString(c10::ScalarType::ComplexFloat) << " ";
  file << c10::toString(c10::ScalarType::ComplexDouble) << " ";
  file << c10::toString(c10::ScalarType::Bool) << " ";
  file << c10::toString(c10::ScalarType::BFloat16) << " ";
  file << c10::toString(c10::ScalarType::QInt8) << " ";
  file << c10::toString(c10::ScalarType::QUInt8) << " ";
  file << c10::toString(c10::ScalarType::Float8_e5m2) << " ";
  file << c10::toString(c10::ScalarType::UInt16) << " ";
  file << c10::toString(c10::ScalarType::UInt32) << " ";
  file << c10::toString(c10::ScalarType::UInt64) << " ";
  file.saveFile();
}

// 测试 c10::elementSize
TEST_F(ScalarTypeTest, ElementSize) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();

  // Test elementSize for various ScalarTypes
  file << std::to_string(c10::elementSize(c10::ScalarType::Byte)) << " ";   // 1
  file << std::to_string(c10::elementSize(c10::ScalarType::Char)) << " ";   // 1
  file << std::to_string(c10::elementSize(c10::ScalarType::Short)) << " ";  // 2
  file << std::to_string(c10::elementSize(c10::ScalarType::Int)) << " ";    // 4
  file << std::to_string(c10::elementSize(c10::ScalarType::Long)) << " ";   // 8
  file << std::to_string(c10::elementSize(c10::ScalarType::Half)) << " ";   // 2
  file << std::to_string(c10::elementSize(c10::ScalarType::Float)) << " ";  // 4
  file << std::to_string(c10::elementSize(c10::ScalarType::Double))
       << " ";  // 8
  file << std::to_string(c10::elementSize(c10::ScalarType::ComplexFloat))
       << " ";  // 8
  file << std::to_string(c10::elementSize(c10::ScalarType::ComplexDouble))
       << " ";                                                             // 16
  file << std::to_string(c10::elementSize(c10::ScalarType::Bool)) << " ";  // 1
  file << std::to_string(c10::elementSize(c10::ScalarType::BFloat16))
       << " ";                                                              // 2
  file << std::to_string(c10::elementSize(c10::ScalarType::QInt8)) << " ";  // 1
  file << std::to_string(c10::elementSize(c10::ScalarType::QUInt8))
       << " ";  // 1
  file << std::to_string(c10::elementSize(c10::ScalarType::QInt32))
       << " ";  // 4
  file << std::to_string(c10::elementSize(c10::ScalarType::UInt16))
       << " ";  // 2
  file << std::to_string(c10::elementSize(c10::ScalarType::UInt32))
       << " ";  // 4
  file << std::to_string(c10::elementSize(c10::ScalarType::UInt64))
       << " ";  // 8
  file.saveFile();
}

// 测试 c10::isIntegralType
TEST_F(ScalarTypeTest, IsIntegralType) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();

  // Test integral types (without bool)
  file << std::to_string(c10::isIntegralType(c10::ScalarType::Byte, false))
       << " ";
  file << std::to_string(c10::isIntegralType(c10::ScalarType::Char, false))
       << " ";
  file << std::to_string(c10::isIntegralType(c10::ScalarType::Int, false))
       << " ";
  file << std::to_string(c10::isIntegralType(c10::ScalarType::Long, false))
       << " ";
  file << std::to_string(c10::isIntegralType(c10::ScalarType::Short, false))
       << " ";
  file << std::to_string(c10::isIntegralType(c10::ScalarType::UInt16, false))
       << " ";
  file << std::to_string(c10::isIntegralType(c10::ScalarType::UInt32, false))
       << " ";
  file << std::to_string(c10::isIntegralType(c10::ScalarType::UInt64, false))
       << " ";

  // Test non-integral types (should be false)
  file << std::to_string(c10::isIntegralType(c10::ScalarType::Float, false))
       << " ";
  file << std::to_string(c10::isIntegralType(c10::ScalarType::Double, false))
       << " ";
  file << std::to_string(c10::isIntegralType(c10::ScalarType::Bool, false))
       << " ";

  // Test with includeBool = true
  file << std::to_string(c10::isIntegralType(c10::ScalarType::Bool, true))
       << " ";
  file.saveFile();
}

#ifndef USE_PADDLE_API

// 测试 c10::isFloat8Type
TEST_F(ScalarTypeTest, IsFloat8Type) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();

  // Test Float8 types
  file << std::to_string(c10::isFloat8Type(c10::ScalarType::Float8_e5m2))
       << " ";
  file << std::to_string(c10::isFloat8Type(c10::ScalarType::Float8_e5m2fnuz))
       << " ";
  file << std::to_string(c10::isFloat8Type(c10::ScalarType::Float8_e4m3fn))
       << " ";
  file << std::to_string(c10::isFloat8Type(c10::ScalarType::Float8_e4m3fnuz))
       << " ";

  // Test non-Float8 types (should be false)
  file << std::to_string(c10::isFloat8Type(c10::ScalarType::Float)) << " ";
  file << std::to_string(c10::isFloat8Type(c10::ScalarType::Double)) << " ";
  file << std::to_string(c10::isFloat8Type(c10::ScalarType::Half)) << " ";
  file << std::to_string(c10::isFloat8Type(c10::ScalarType::Int)) << " ";
  file.saveFile();
}

// 测试 c10::isReducedFloatingType
TEST_F(ScalarTypeTest, IsReducedFloatingType) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();

  // Test reduced floating types
  file << std::to_string(c10::isReducedFloatingType(c10::ScalarType::Half))
       << " ";
  file << std::to_string(c10::isReducedFloatingType(c10::ScalarType::BFloat16))
       << " ";
  file << std::to_string(
              c10::isReducedFloatingType(c10::ScalarType::Float8_e5m2))
       << " ";

  // Test normal floating types (should be false)
  file << std::to_string(c10::isReducedFloatingType(c10::ScalarType::Float))
       << " ";
  file << std::to_string(c10::isReducedFloatingType(c10::ScalarType::Double))
       << " ";

  // Test non-floating types (should be false)
  file << std::to_string(c10::isReducedFloatingType(c10::ScalarType::Int))
       << " ";
  file << std::to_string(c10::isReducedFloatingType(c10::ScalarType::Bool))
       << " ";
  file.saveFile();
}

// 测试 c10::isFloatingType
TEST_F(ScalarTypeTest, IsFloatingType) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();

  // Test floating types
  file << std::to_string(c10::isFloatingType(c10::ScalarType::Float)) << " ";
  file << std::to_string(c10::isFloatingType(c10::ScalarType::Double)) << " ";
  file << std::to_string(c10::isFloatingType(c10::ScalarType::Half)) << " ";
  file << std::to_string(c10::isFloatingType(c10::ScalarType::BFloat16)) << " ";
  file << std::to_string(c10::isFloatingType(c10::ScalarType::Float8_e5m2))
       << " ";

  // Test non-floating types (should be false)
  file << std::to_string(c10::isFloatingType(c10::ScalarType::Int)) << " ";
  file << std::to_string(c10::isFloatingType(c10::ScalarType::Long)) << " ";
  file << std::to_string(c10::isFloatingType(c10::ScalarType::Bool)) << " ";
  file.saveFile();
}

// 测试 c10::isComplexType
TEST_F(ScalarTypeTest, IsComplexType) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();

  // Test complex types
  file << std::to_string(c10::isComplexType(c10::ScalarType::ComplexHalf))
       << " ";
  file << std::to_string(c10::isComplexType(c10::ScalarType::ComplexFloat))
       << " ";
  file << std::to_string(c10::isComplexType(c10::ScalarType::ComplexDouble))
       << " ";

  // Test non-complex types (should be false)
  file << std::to_string(c10::isComplexType(c10::ScalarType::Float)) << " ";
  file << std::to_string(c10::isComplexType(c10::ScalarType::Double)) << " ";
  file << std::to_string(c10::isComplexType(c10::ScalarType::Int)) << " ";
  file << std::to_string(c10::isComplexType(c10::ScalarType::Bool)) << " ";
  file.saveFile();
}

// 测试 c10::isQIntType
TEST_F(ScalarTypeTest, IsQIntType) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();

  // Test quantized int types
  file << std::to_string(c10::isQIntType(c10::ScalarType::QInt8)) << " ";
  file << std::to_string(c10::isQIntType(c10::ScalarType::QUInt8)) << " ";
  file << std::to_string(c10::isQIntType(c10::ScalarType::QInt32)) << " ";
  file << std::to_string(c10::isQIntType(c10::ScalarType::QUInt4x2)) << " ";
  file << std::to_string(c10::isQIntType(c10::ScalarType::QUInt2x4)) << " ";

  // Test non-qint types (should be false)
  file << std::to_string(c10::isQIntType(c10::ScalarType::Int)) << " ";
  file << std::to_string(c10::isQIntType(c10::ScalarType::Float)) << " ";
  file << std::to_string(c10::isQIntType(c10::ScalarType::Byte)) << " ";
  file.saveFile();
}

// 测试 c10::isBitsType
TEST_F(ScalarTypeTest, IsBitsType) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();

  // Test bits types
  file << std::to_string(c10::isBitsType(c10::ScalarType::Bits1x8)) << " ";
  file << std::to_string(c10::isBitsType(c10::ScalarType::Bits2x4)) << " ";
  file << std::to_string(c10::isBitsType(c10::ScalarType::Bits4x2)) << " ";
  file << std::to_string(c10::isBitsType(c10::ScalarType::Bits8)) << " ";
  file << std::to_string(c10::isBitsType(c10::ScalarType::Bits16)) << " ";

  // Test non-bits types (should be false)
  file << std::to_string(c10::isBitsType(c10::ScalarType::Int)) << " ";
  file << std::to_string(c10::isBitsType(c10::ScalarType::Float)) << " ";
  file << std::to_string(c10::isBitsType(c10::ScalarType::Byte)) << " ";
  file.saveFile();
}

// 测试 c10::isBarebonesUnsignedType
TEST_F(ScalarTypeTest, IsBarebonesUnsignedType) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();

  // Test barebones unsigned types
  file << std::to_string(c10::isBarebonesUnsignedType(c10::ScalarType::UInt1))
       << " ";
  file << std::to_string(c10::isBarebonesUnsignedType(c10::ScalarType::UInt2))
       << " ";
  file << std::to_string(c10::isBarebonesUnsignedType(c10::ScalarType::UInt3))
       << " ";
  file << std::to_string(c10::isBarebonesUnsignedType(c10::ScalarType::UInt4))
       << " ";
  file << std::to_string(c10::isBarebonesUnsignedType(c10::ScalarType::UInt5))
       << " ";
  file << std::to_string(c10::isBarebonesUnsignedType(c10::ScalarType::UInt6))
       << " ";
  file << std::to_string(c10::isBarebonesUnsignedType(c10::ScalarType::UInt7))
       << " ";
  file << std::to_string(c10::isBarebonesUnsignedType(c10::ScalarType::UInt16))
       << " ";
  file << std::to_string(c10::isBarebonesUnsignedType(c10::ScalarType::UInt32))
       << " ";
  file << std::to_string(c10::isBarebonesUnsignedType(c10::ScalarType::UInt64))
       << " ";

  // Test non-unsigned types (should be false)
  file << std::to_string(c10::isBarebonesUnsignedType(c10::ScalarType::Int))
       << " ";
  file << std::to_string(c10::isBarebonesUnsignedType(c10::ScalarType::Float))
       << " ";
  file << std::to_string(c10::isBarebonesUnsignedType(c10::ScalarType::Byte))
       << " ";
  file.saveFile();
}

// 测试 c10::toQIntType
TEST_F(ScalarTypeTest, ToQIntType) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();

  // Test conversion to QInt types
  file << c10::toString(c10::toQIntType(c10::ScalarType::Byte)) << " ";
  file << c10::toString(c10::toQIntType(c10::ScalarType::Char)) << " ";
  file << c10::toString(c10::toQIntType(c10::ScalarType::Int)) << " ";

  // Test non-quantized types (should return same type)
  file << c10::toString(c10::toQIntType(c10::ScalarType::Float)) << " ";
  file << c10::toString(c10::toQIntType(c10::ScalarType::Double)) << " ";
  file << c10::toString(c10::toQIntType(c10::ScalarType::Long)) << " ";
  file.saveFile();
}

// 测试 c10::toUnderlying
TEST_F(ScalarTypeTest, ToUnderlying) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();

  // Test conversion to underlying types
  file << c10::toString(c10::toUnderlying(c10::ScalarType::QUInt8)) << " ";
  file << c10::toString(c10::toUnderlying(c10::ScalarType::QUInt4x2)) << " ";
  file << c10::toString(c10::toUnderlying(c10::ScalarType::QUInt2x4)) << " ";
  file << c10::toString(c10::toUnderlying(c10::ScalarType::QInt8)) << " ";
  file << c10::toString(c10::toUnderlying(c10::ScalarType::QInt32)) << " ";

  // Test non-quantized types (should return same type)
  file << c10::toString(c10::toUnderlying(c10::ScalarType::Float)) << " ";
  file << c10::toString(c10::toUnderlying(c10::ScalarType::Int)) << " ";
  file.saveFile();
}

// 测试 c10::isSignedType
TEST_F(ScalarTypeTest, IsSignedType) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();

  // Test signed types
  file << std::to_string(c10::isSignedType(c10::ScalarType::Char)) << " ";
  file << std::to_string(c10::isSignedType(c10::ScalarType::Short)) << " ";
  file << std::to_string(c10::isSignedType(c10::ScalarType::Int)) << " ";
  file << std::to_string(c10::isSignedType(c10::ScalarType::Long)) << " ";
  file << std::to_string(c10::isSignedType(c10::ScalarType::Float)) << " ";
  file << std::to_string(c10::isSignedType(c10::ScalarType::Double)) << " ";
  file << std::to_string(c10::isSignedType(c10::ScalarType::Half)) << " ";
  file << std::to_string(c10::isSignedType(c10::ScalarType::BFloat16)) << " ";
  file << std::to_string(c10::isSignedType(c10::ScalarType::ComplexFloat))
       << " ";
  file << std::to_string(c10::isSignedType(c10::ScalarType::ComplexDouble))
       << " ";

  // Test unsigned types
  file << std::to_string(c10::isSignedType(c10::ScalarType::Byte)) << " ";
  file << std::to_string(c10::isSignedType(c10::ScalarType::UInt16)) << " ";
  file << std::to_string(c10::isSignedType(c10::ScalarType::UInt32)) << " ";
  file << std::to_string(c10::isSignedType(c10::ScalarType::UInt64)) << " ";

  // Test bool (unsigned)
  file << std::to_string(c10::isSignedType(c10::ScalarType::Bool)) << " ";
  file.saveFile();
}

// 测试 c10::isUnderlying
TEST_F(ScalarTypeTest, IsUnderlying) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();

  // Test isUnderlying
  file << std::to_string(
              c10::isUnderlying(c10::ScalarType::Byte, c10::ScalarType::QUInt8))
       << " ";
  file << std::to_string(
              c10::isUnderlying(c10::ScalarType::Char, c10::ScalarType::QInt8))
       << " ";
  file << std::to_string(
              c10::isUnderlying(c10::ScalarType::Int, c10::ScalarType::QInt32))
       << " ";
  file << std::to_string(
              c10::isUnderlying(c10::ScalarType::Byte, c10::ScalarType::QInt8))
       << " ";  // false
  file << std::to_string(c10::isUnderlying(c10::ScalarType::Float,
                                           c10::ScalarType::QUInt8))
       << " ";  // false
  file.saveFile();
}

// 测试 c10::toRealValueType
TEST_F(ScalarTypeTest, ToRealValueType) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();

  // Test conversion from complex to real types
  file << c10::toString(c10::toRealValueType(c10::ScalarType::ComplexHalf))
       << " ";
  file << c10::toString(c10::toRealValueType(c10::ScalarType::ComplexFloat))
       << " ";
  file << c10::toString(c10::toRealValueType(c10::ScalarType::ComplexDouble))
       << " ";

  // Test non-complex types (should return same type)
  file << c10::toString(c10::toRealValueType(c10::ScalarType::Float)) << " ";
  file << c10::toString(c10::toRealValueType(c10::ScalarType::Double)) << " ";
  file << c10::toString(c10::toRealValueType(c10::ScalarType::Int)) << " ";
  file.saveFile();
}

// 测试 c10::toComplexType
TEST_F(ScalarTypeTest, ToComplexType) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();

  // Test conversion to complex types
  file << c10::toString(c10::toComplexType(c10::ScalarType::Half)) << " ";
  file << c10::toString(c10::toComplexType(c10::ScalarType::Float)) << " ";
  file << c10::toString(c10::toComplexType(c10::ScalarType::Double)) << " ";
  file << c10::toString(c10::toComplexType(c10::ScalarType::BFloat16)) << " ";

  // Test complex to complex (should return same type)
  file << c10::toString(c10::toComplexType(c10::ScalarType::ComplexHalf))
       << " ";
  file << c10::toString(c10::toComplexType(c10::ScalarType::ComplexFloat))
       << " ";
  file << c10::toString(c10::toComplexType(c10::ScalarType::ComplexDouble))
       << " ";
  file.saveFile();
}

// 测试 c10::canCast
TEST_F(ScalarTypeTest, CanCast) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();

  // Test valid casts
  file << std::to_string(
              c10::canCast(c10::ScalarType::Int, c10::ScalarType::Long))
       << " ";
  file << std::to_string(
              c10::canCast(c10::ScalarType::Float, c10::ScalarType::Double))
       << " ";
  file << std::to_string(c10::canCast(c10::ScalarType::ComplexFloat,
                                      c10::ScalarType::ComplexDouble))
       << " ";
  file << std::to_string(
              c10::canCast(c10::ScalarType::Bool, c10::ScalarType::Int))
       << " ";

  // Test disallowed: complex to non-complex
  file << std::to_string(c10::canCast(c10::ScalarType::ComplexFloat,
                                      c10::ScalarType::Float))
       << " ";  // false

  // Test disallowed: float to integral
  file << std::to_string(
              c10::canCast(c10::ScalarType::Float, c10::ScalarType::Int))
       << " ";  // false
  file << std::to_string(
              c10::canCast(c10::ScalarType::Double, c10::ScalarType::Long))
       << " ";  // false

  // Test disallowed: to bool
  file << std::to_string(
              c10::canCast(c10::ScalarType::Int, c10::ScalarType::Bool))
       << " ";  // false
  file << std::to_string(
              c10::canCast(c10::ScalarType::Float, c10::ScalarType::Bool))
       << " ";  // false

  file.saveFile();
}

// 测试 NumScalarTypes 常量
TEST_F(ScalarTypeTest, NumScalarTypes) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();

  file << std::to_string(c10::NumScalarTypes) << " ";
  file.saveFile();
}

#endif  // USE_PADDLE_API

}  // namespace test
}  // namespace at
