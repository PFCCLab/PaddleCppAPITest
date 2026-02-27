#include <ATen/ATen.h>
#include <ATen/core/Tensor.h>
#include <ATen/ops/ones.h>
#include <ATen/ops/zeros.h>
#include <gtest/gtest.h>

#include <optional>
#include <string>
#include <vector>

#include "../../src/file_manager.h"

extern paddle_api_test::ThreadSafeParam g_custom_param;

namespace at {
namespace test {

using paddle_api_test::FileManerger;
using paddle_api_test::ThreadSafeParam;

class ToTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // 2x3 float tensor，values 1..6
    tensor_float = at::zeros({2, 3}, at::kFloat);
    float* d = tensor_float.data_ptr<float>();
    for (int64_t i = 0; i < 6; ++i) d[i] = static_cast<float>(i + 1);

    // 2x3 double tensor
    tensor_double = tensor_float.to(at::kDouble);
  }

  at::Tensor tensor_float;
  at::Tensor tensor_double;
};

// --------------------------------------------------------------------------
// 重载 1：to(TensorOptions)
// --------------------------------------------------------------------------

// 测试 to(TensorOptions) 转换数据类型
TEST_F(ToTest, ToTensorOptionsDtype) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();

  at::TensorOptions opts = at::TensorOptions().dtype(at::kDouble);
  at::Tensor result = tensor_float.to(opts);
  file << std::to_string(static_cast<int>(result.scalar_type())) << " ";
  file << std::to_string(result.numel()) << " ";
  file << std::to_string(result[0][0].item<double>()) << " ";
  file.saveFile();
}

// 测试 to(TensorOptions) 不指定 dtype 时返回原类型
TEST_F(ToTest, ToTensorOptionsEmpty) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();

  at::Tensor result = tensor_float.to(at::TensorOptions{});
  file << std::to_string(static_cast<int>(result.scalar_type())) << " ";
  file << std::to_string(result.numel()) << " ";
  file.saveFile();
}

// 测试 to(TensorOptions, non_blocking=false, copy=true) 强制拷贝
TEST_F(ToTest, ToTensorOptionsForceCopy) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();

  at::TensorOptions opts = at::TensorOptions().dtype(at::kFloat);
  at::Tensor result = tensor_float.to(opts,
                                      /*non_blocking=*/false,
                                      /*copy=*/true);
  // 强制拷贝后数据独立
  result.fill_(0.f);
  file << std::to_string(tensor_float[0][0].item<float>()) << " ";  // 仍为 1
  file << std::to_string(static_cast<int>(result.scalar_type())) << " ";
  file.saveFile();
}

// --------------------------------------------------------------------------
// 重载 2：to(optional<ScalarType>, optional<Layout>, optional<Device>,
//            optional<bool>, non_blocking, copy, optional<MemoryFormat>)
// --------------------------------------------------------------------------

TEST_F(ToTest, ToFullOptional) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();

  at::Tensor result = tensor_float.to(
      /*dtype=*/std::make_optional(at::kDouble),
      /*layout=*/std::nullopt,
      /*device=*/std::nullopt,
      /*pin_memory=*/std::nullopt,
      /*non_blocking=*/false,
      /*copy=*/false,
      /*memory_format=*/std::nullopt);
  file << std::to_string(static_cast<int>(result.scalar_type())) << " ";
  file << std::to_string(result.numel()) << " ";
  file.saveFile();
}

// --------------------------------------------------------------------------
// 重载 3：to(Device, ScalarType, non_blocking, copy, memory_format)
// --------------------------------------------------------------------------

TEST_F(ToTest, ToDeviceAndDtype) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();

  at::Device cpu_dev(at::kCPU);
  at::Tensor result = tensor_float.to(cpu_dev,
                                      at::kInt,
                                      /*non_blocking=*/false,
                                      /*copy=*/false);
  file << std::to_string(static_cast<int>(result.scalar_type())) << " ";
  file << std::to_string(result.numel()) << " ";
  file << std::to_string(result[0][0].item<int>()) << " ";  // 1
  file.saveFile();
}

// --------------------------------------------------------------------------
// 重载 4：to(ScalarType, non_blocking, copy, memory_format)
// --------------------------------------------------------------------------

// 测试 to(ScalarType) float -> double
TEST_F(ToTest, ToScalarTypeFloatToDouble) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();

  at::Tensor result = tensor_float.to(at::kDouble);
  file << std::to_string(static_cast<int>(result.scalar_type())) << " ";
  file << std::to_string(result.numel()) << " ";
  file << std::to_string(result[0][0].item<double>()) << " ";
  file.saveFile();
}

// 测试 to(ScalarType) float -> int
TEST_F(ToTest, ToScalarTypeFloatToInt) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();

  at::Tensor result = tensor_float.to(at::kInt);
  file << std::to_string(static_cast<int>(result.scalar_type())) << " ";
  file << std::to_string(result[0][0].item<int>()) << " ";
  file << std::to_string(result[0][1].item<int>()) << " ";
  file.saveFile();
}

// 测试 to(ScalarType) float -> long
TEST_F(ToTest, ToScalarTypeFloatToLong) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();

  at::Tensor result = tensor_float.to(at::kLong);
  file << std::to_string(static_cast<int>(result.scalar_type())) << " ";
  file << std::to_string(result[0][0].item<int64_t>()) << " ";
  file.saveFile();
}

// 测试 to(ScalarType, non_blocking=false, copy=true) 强制拷贝同类型
TEST_F(ToTest, ToScalarTypeSameTypeCopy) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();

  at::Tensor result =
      tensor_float.to(at::kFloat, /*non_blocking=*/false, /*copy=*/true);
  result.fill_(0.f);
  // 拷贝后互不影响
  file << std::to_string(tensor_float[0][0].item<float>()) << " ";  // 1
  file << std::to_string(result[0][0].item<float>()) << " ";        // 0
  file.saveFile();
}

// --------------------------------------------------------------------------
// 重载 5：to(const Tensor& other, non_blocking, copy, memory_format)
// --------------------------------------------------------------------------

// 测试 to(other)：转换为与 other 相同的 dtype/device
TEST_F(ToTest, ToOtherTensor) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();

  // other 是 double 类型
  at::Tensor other = at::zeros({1}, at::kDouble);
  at::Tensor result = tensor_float.to(other);
  file << std::to_string(static_cast<int>(result.scalar_type())) << " ";
  file << std::to_string(result.numel()) << " ";
  file << std::to_string(result[0][0].item<double>()) << " ";
  file.saveFile();
}

// 测试 to(other) 相同类型时不触发数据拷贝（返回原或视图）
TEST_F(ToTest, ToOtherSameType) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();

  at::Tensor other = at::zeros({1}, at::kFloat);
  at::Tensor result = tensor_float.to(other);
  file << std::to_string(static_cast<int>(result.scalar_type())) << " ";
  file << std::to_string(result.numel()) << " ";
  file.saveFile();
}

}  // namespace test
}  // namespace at
