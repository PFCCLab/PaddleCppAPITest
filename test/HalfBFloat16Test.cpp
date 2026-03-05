#include <ATen/ATen.h>
#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
#include <gtest/gtest.h>

#include <cmath>
#include <string>

#include "../src/file_manager.h"

extern paddle_api_test::ThreadSafeParam g_custom_param;

namespace at {
namespace test {

using paddle_api_test::FileManerger;
using paddle_api_test::ThreadSafeParam;

class HalfBFloat16Test : public ::testing::Test {
 protected:
  void SetUp() override {}
};

// ===================== Half (float16) =====================

// Half 基本构造和转换
TEST_F(HalfBFloat16Test, HalfBasic) {
  c10::Half h(3.14f);
  float f = static_cast<float>(h);

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();
  // 由于 fp16 精度，结果是近似值
  file << std::to_string(f) << " ";
  file.saveFile();
}

// Half 零
TEST_F(HalfBFloat16Test, HalfZero) {
  c10::Half h(0.0f);
  float f = static_cast<float>(h);
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << std::to_string(f) << " ";
  file.saveFile();
}

// Half 负数
TEST_F(HalfBFloat16Test, HalfNegative) {
  c10::Half h(-2.5f);
  float f = static_cast<float>(h);
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << std::to_string(f) << " ";
  file.saveFile();
}

// Half 小值
TEST_F(HalfBFloat16Test, HalfSmallValue) {
  c10::Half h(0.001f);
  float f = static_cast<float>(h);
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  // 近似保留
  file << std::to_string(std::abs(f - 0.001f) < 0.001f ? 1 : 0) << " ";
  file.saveFile();
}

// Half 命名空间别名
TEST_F(HalfBFloat16Test, HalfNamespace) {
  c10::Half h1(1.0f);
  c10::Half h2(2.0f);
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << std::to_string(static_cast<float>(h1)) << " ";
  file << std::to_string(static_cast<float>(h2)) << " ";
  file.saveFile();
}

// ===================== BFloat16 =====================

// BFloat16 基本构造和转换
TEST_F(HalfBFloat16Test, BFloat16Basic) {
  c10::BFloat16 b(3.14f);
  float f = static_cast<float>(b);

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << std::to_string(f) << " ";
  file.saveFile();
}

// BFloat16 零
TEST_F(HalfBFloat16Test, BFloat16Zero) {
  c10::BFloat16 b(0.0f);
  float f = static_cast<float>(b);
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << std::to_string(f) << " ";
  file.saveFile();
}

// BFloat16 负数
TEST_F(HalfBFloat16Test, BFloat16Negative) {
  c10::BFloat16 b(-2.5f);
  float f = static_cast<float>(b);
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << std::to_string(f) << " ";
  file.saveFile();
}

// BFloat16 命名空间别名
TEST_F(HalfBFloat16Test, BFloat16Namespace) {
  c10::BFloat16 b1(1.0f);
  c10::BFloat16 b2(2.0f);
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << std::to_string(static_cast<float>(b1)) << " ";
  file << std::to_string(static_cast<float>(b2)) << " ";
  file.saveFile();
}

// ScalarType 对应关系
// [DIFF] PyTorch输出: 5 11, PaddlePaddle输出: 5 15 (BFloat16枚举值不同)
TEST_F(HalfBFloat16Test, ScalarTypeMapping) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << std::to_string(static_cast<int>(at::kHalf)) << " ";
  // file << std::to_string(static_cast<int>(at::kBFloat16)) << " "; // [DIFF]
  file.saveFile();
}

}  // namespace test
}  // namespace at
