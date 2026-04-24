#include <ATen/ATen.h>
#include <ATen/core/Tensor.h>
#include <ATen/ops/sparse_coo_tensor.h>
#include <ATen/ops/zeros.h>
#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "src/file_manager.h"

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
  file << "ItemFloatScalar ";

  at::Scalar s = scalar_float.item();
  file << std::to_string(s.to<float>()) << " ";
  file << "\n";
  file.saveFile();
}

// 测试 item<float>() 模板形式
TEST_F(ItemTest, ItemTemplateFloat) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "ItemTemplateFloat ";

  float val = scalar_float.item<float>();
  file << std::to_string(val) << " ";
  file << "\n";
  file.saveFile();
}

// 测试 item<int>() 从 int tensor
TEST_F(ItemTest, ItemTemplateInt) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "ItemTemplateInt ";

  int val = scalar_int.item<int>();
  file << std::to_string(val) << " ";
  file << "\n";
  file.saveFile();
}

// 测试 item<double>() 获取 double 精度值
TEST_F(ItemTest, ItemTemplateDouble) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "ItemTemplateDouble ";

  double val = scalar_double.item<double>();
  // 保留 9 位有效数字
  file << std::to_string(val) << " ";
  file << "\n";
  file.saveFile();
}

// 测试 item<int64_t>()
TEST_F(ItemTest, ItemTemplateInt64) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "ItemTemplateInt64 ";

  at::Tensor t = at::zeros({}, at::kLong);
  t.fill_(static_cast<int64_t>(1234567890));
  int64_t val = t.item<int64_t>();
  file << std::to_string(val) << " ";
  file << "\n";
  file.saveFile();
}

// 测试 item() 对单元素 1-dim tensor（squeeze 后语义）
TEST_F(ItemTest, ItemFromSingleElementTensor) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "ItemFromSingleElementTensor ";

  at::Tensor t = at::zeros({1}, at::kFloat);
  t.fill_(7.5f);
  float val = t.item<float>();
  file << std::to_string(val) << " ";
  file << "\n";
  file.saveFile();
}

// 测试 item() 跨类型转换：double tensor 通过 item<float>()
TEST_F(ItemTest, ItemCrossTypeCast) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "ItemCrossTypeCast ";

  float val = scalar_double.item<float>();
  file << std::to_string(val) << " ";
  file << "\n";
  file.saveFile();
}

TEST_F(ItemTest, ItemAdditionalScalarDtypes) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "ItemAdditionalScalarDtypes ";

  try {
    at::Tensor half_tensor = at::zeros({}, at::kHalf);
    at::Tensor bfloat_tensor = at::zeros({}, at::kBFloat16);
    at::Tensor int8_tensor = at::zeros({}, at::kChar);
    at::Tensor int16_tensor = at::zeros({}, at::kShort);
    at::Tensor uint8_tensor = at::zeros({}, at::kByte);

    file << std::to_string(half_tensor.item<float>()) << " ";
    file << std::to_string(bfloat_tensor.item<float>()) << " ";
    file << std::to_string(static_cast<int>(int8_tensor.item<int8_t>())) << " ";
    file << std::to_string(static_cast<int>(int16_tensor.item<int16_t>()))
         << " ";
    file << std::to_string(static_cast<int>(uint8_tensor.item<uint8_t>()))
         << " ";
  } catch (const std::exception&) {
    file << "exception ";
  }
  file << "\n";
  file.saveFile();
}

TEST_F(ItemTest, ItemComplexScalarDtypes) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "ItemComplexScalarDtypes ";

  try {
    at::Tensor complex64_tensor = at::zeros({}, at::kComplexFloat);
    at::Tensor complex128_tensor = at::zeros({}, at::kComplexDouble);
    (void)complex64_tensor.item();
    (void)complex128_tensor.item();
    file << "ok ";
  } catch (const std::exception&) {
    file << "exception ";
  }
  file << "\n";
  file.saveFile();
}

TEST_F(ItemTest, ItemSparseScalarPaths) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "ItemSparseScalarPaths ";

  try {
    at::Tensor empty_indices = at::zeros({1, 0}, at::kLong);
    at::Tensor empty_values = at::zeros({0}, at::kFloat);
    at::Tensor empty_sparse =
        at::sparse_coo_tensor(empty_indices, empty_values, {1});
    file << std::to_string(empty_sparse.item<float>()) << " ";
  } catch (const std::exception&) {
    file << "empty_exception ";
  }

  try {
    at::Tensor indices = at::zeros({1, 1}, at::kLong);
    at::Tensor values = at::zeros({1}, at::kFloat);
    values[0] = 7.0f;
    at::Tensor sparse = at::sparse_coo_tensor(indices, values, {1});
    file << std::to_string(sparse.coalesce().item<float>()) << " ";
  } catch (const std::exception&) {
    file << "coalesced_exception ";
  }

  try {
    at::Tensor duplicate_indices = at::zeros({1, 2}, at::kLong);
    at::Tensor duplicate_values = at::zeros({2}, at::kFloat);
    duplicate_values[0] = 2.0f;
    duplicate_values[1] = 3.0f;
    at::Tensor duplicate_sparse =
        at::sparse_coo_tensor(duplicate_indices, duplicate_values, {1});
    file << std::to_string(duplicate_sparse.item<float>()) << " ";
  } catch (const std::exception&) {
    file << "duplicate_exception ";
  }

  file << "\n";
  file.saveFile();
}

}  // namespace test
}  // namespace at
