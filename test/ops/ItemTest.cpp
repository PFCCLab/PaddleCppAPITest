#include <ATen/ATen.h>
#include <ATen/core/Tensor.h>
#include <ATen/ops/zeros.h>
#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "../../src/file_manager.h"

extern paddle_api_test::ThreadSafeParam g_custom_param;

namespace at {
namespace test {

using paddle_api_test::FileManerger;
using paddle_api_test::ThreadSafeParam;

class ItemTest : public ::testing::Test {
 protected:
  void SetUp() override {
    scalar_float = at::zeros({}, at::kFloat);
    scalar_float.fill_(3.14f);

    scalar_int = at::zeros({}, at::kInt);
    scalar_int.fill_(42);

    scalar_double = at::zeros({}, at::kDouble);
    scalar_double.fill_(2.718281828);
  }

  at::Tensor scalar_float;
  at::Tensor scalar_int;
  at::Tensor scalar_double;
};

// 测试 item() 从 float 0-dim tensor 获取标量（返回 at::Scalar）
TEST_F(ItemTest, ItemFloatScalar) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();

  at::Scalar s = scalar_float.item();
  file << std::to_string(s.to<float>()) << " ";
  file.saveFile();
}

// 测试 item<float>() 模板形式
TEST_F(ItemTest, ItemTemplateFloat) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();

  float val = scalar_float.item<float>();
  file << std::to_string(val) << " ";
  file.saveFile();
}

// 测试 item<int>() 从 int tensor
TEST_F(ItemTest, ItemTemplateInt) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();

  int val = scalar_int.item<int>();
  file << std::to_string(val) << " ";
  file.saveFile();
}

// 测试 item<double>() 获取 double 精度值
TEST_F(ItemTest, ItemTemplateDouble) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();

  double val = scalar_double.item<double>();
  // 保留 9 位有效数字
  file << std::to_string(val) << " ";
  file.saveFile();
}

// 测试 item<int64_t>()
TEST_F(ItemTest, ItemTemplateInt64) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();

  at::Tensor t = at::zeros({}, at::kLong);
  t.fill_(static_cast<int64_t>(1234567890));
  int64_t val = t.item<int64_t>();
  file << std::to_string(val) << " ";
  file.saveFile();
}

// 测试 item() 对单元素 1-dim tensor（squeeze 后语义）
TEST_F(ItemTest, ItemFromSingleElementTensor) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();

  at::Tensor t = at::zeros({1}, at::kFloat);
  t.fill_(7.5f);
  float val = t.item<float>();
  file << std::to_string(val) << " ";
  file.saveFile();
}

// 测试 item() 跨类型转换：double tensor 通过 item<float>()
TEST_F(ItemTest, ItemCrossTypeCast) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();

  float val = scalar_double.item<float>();
  file << std::to_string(val) << " ";
  file.saveFile();
}

}  // namespace test
}  // namespace at
