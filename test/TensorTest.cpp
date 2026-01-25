#include <ATen/ATen.h>
#include <ATen/core/Tensor.h>
#include <ATen/ops/ones.h>
#include <gtest/gtest.h>
#include <torch/all.h>

#include <vector>

#include "../src/file_manager.h"
namespace at {
namespace test {

using paddle_api_test::FileManerger;
class TensorTest : public ::testing::Test {
 protected:
  void SetUp() override {
    std::vector<int64_t> shape = {2, 3, 4};

    tensor = at::ones(shape, at::kFloat);
    // std::cout << "tensor dim: " << tensor.dim() << std::endl;
  }

  at::Tensor tensor;
};

TEST_F(TensorTest, ConstructFromPaddleTensor) {
  EXPECT_EQ(tensor.dim(), 3);
  EXPECT_EQ(tensor.numel(), 24);  // 2*3*4
}

// 测试 data_ptr
TEST_F(TensorTest, DataPtr) {
  // Tensor tensor(paddle_tensor_);

  void* ptr = tensor.data_ptr();
  EXPECT_NE(ptr, nullptr);

  float* float_ptr = tensor.data_ptr<float>();
  EXPECT_NE(float_ptr, nullptr);
}

// 测试 strides
TEST_F(TensorTest, Strides) {
  // Tensor tensor(paddle_tensor_);

  c10::IntArrayRef strides = tensor.strides();
  EXPECT_GT(strides.size(), 0U);  // 使用无符号字面量
}

// 测试 sizes
TEST_F(TensorTest, Sizes) {
  // Tensor tensor(paddle_tensor_);

  c10::IntArrayRef sizes = tensor.sizes();
  EXPECT_EQ(sizes.size(), 3U);
  EXPECT_EQ(sizes[0], 2U);
  EXPECT_EQ(sizes[1], 3U);
  EXPECT_EQ(sizes[2], 4U);
}

// 测试 toType
TEST_F(TensorTest, ToType) {
  // Tensor tensor(paddle_tensor_);

  Tensor double_tensor = tensor.toType(c10::ScalarType::Double);
  EXPECT_EQ(double_tensor.dtype(), c10::ScalarType::Double);
}

// 测试 numel
TEST_F(TensorTest, Numel) {
  // Tensor tensor(paddle_tensor_);

  EXPECT_EQ(tensor.numel(), 24U);  // 2*3*4
}

// 测试 device
TEST_F(TensorTest, Device) {
  // Tensor tensor(paddle_tensor_);

  c10::Device device = tensor.device();
  EXPECT_EQ(device.type(), c10::DeviceType::CPU);
}

// 测试 get_device
TEST_F(TensorTest, GetDevice) {
  // Tensor tensor(paddle_tensor_);

  c10::DeviceIndex device_idx = tensor.get_device();
  EXPECT_GE(device_idx, -1);
}

// 测试 dim 和 ndimension
TEST_F(TensorTest, DimAndNdimension) {
  // Tensor tensor(paddle_tensor_);

  EXPECT_EQ(tensor.dim(), 3);
  EXPECT_EQ(tensor.ndimension(), 3);
  EXPECT_EQ(tensor.dim(), tensor.ndimension());
}

// 测试 contiguous
TEST_F(TensorTest, Contiguous) {
  // Tensor tensor(paddle_tensor_);

  at::Tensor cont_tensor = tensor.contiguous();
  EXPECT_TRUE(cont_tensor.is_contiguous());
}

// 测试 is_contiguous
TEST_F(TensorTest, IsContiguous) {
  // Tensor tensor(paddle_tensor_);

  EXPECT_TRUE(tensor.is_contiguous());
}

// 测试 scalar_type
TEST_F(TensorTest, ScalarType) {
  // Tensor tensor(paddle_tensor_);

  c10::ScalarType stype = tensor.scalar_type();
  EXPECT_EQ(stype, c10::ScalarType::Float);
}

// 测试 fill_
TEST_F(TensorTest, Fill) {
  // Tensor tensor(paddle_tensor_);

  tensor.fill_(5.0);
  float* data = tensor.data_ptr<float>();
  EXPECT_FLOAT_EQ(data[0], 5.0f);
}

// 测试 zero_
TEST_F(TensorTest, Zero) {
  // Tensor tensor(paddle_tensor_);

  tensor.zero_();
  float* data = tensor.data_ptr<float>();
  EXPECT_FLOAT_EQ(data[0], 0.0f);
}

// 测试 is_cpu
TEST_F(TensorTest, IsCpu) {
  // Tensor tensor(paddle_tensor_);

  EXPECT_TRUE(tensor.is_cpu());
}

// 测试 is_cuda (在 CPU tensor 上应该返回 false)
TEST_F(TensorTest, IsCuda) {
  // Tensor tensor(paddle_tensor_);

  EXPECT_FALSE(tensor.is_cuda());
}

// 测试 reshape
TEST_F(TensorTest, Reshape) {
  // Tensor tensor(paddle_tensor_);

  at::Tensor reshaped = tensor.reshape({6, 4});
  EXPECT_EQ(reshaped.sizes()[0], 6);
  EXPECT_EQ(reshaped.sizes()[1], 4);
  EXPECT_EQ(reshaped.numel(), 24);
}

// 测试 transpose
TEST_F(TensorTest, Transpose) {
  // Tensor tensor(paddle_tensor_);

  at::Tensor transposed = tensor.transpose(0, 2);
  EXPECT_EQ(transposed.sizes()[0], 4);
  EXPECT_EQ(transposed.sizes()[2], 2);
}

// 测试 is_complex
TEST_F(TensorTest, IsComplex) {
  // Float tensor should not be complex
  EXPECT_FALSE(tensor.is_complex());

  // Test with actual complex tensor
  at::Tensor complex_tensor =
      at::ones({2, 3}, at::TensorOptions().dtype(at::kComplexFloat));
  EXPECT_TRUE(complex_tensor.is_complex());

  at::Tensor complex_double_tensor =
      at::ones({2, 3}, at::TensorOptions().dtype(at::kComplexDouble));
  EXPECT_TRUE(complex_double_tensor.is_complex());
}

// 测试 is_floating_point
TEST_F(TensorTest, IsFloatingPoint) {
  // Float tensor should be floating point
  EXPECT_TRUE(tensor.is_floating_point());

  // Test with double tensor
  at::Tensor double_tensor =
      at::ones({2, 3}, at::TensorOptions().dtype(at::kDouble));
  EXPECT_TRUE(double_tensor.is_floating_point());

  // Test with integer tensor
  at::Tensor int_tensor = at::ones({2, 3}, at::TensorOptions().dtype(at::kInt));
  EXPECT_FALSE(int_tensor.is_floating_point());

  // Test with long tensor
  at::Tensor long_tensor =
      at::ones({2, 3}, at::TensorOptions().dtype(at::kLong));
  EXPECT_FALSE(long_tensor.is_floating_point());
}

// 测试 is_signed
TEST_F(TensorTest, IsSigned) {
  // Float tensor should be signed
  EXPECT_TRUE(tensor.is_signed());

  // Test with int tensor (signed)
  at::Tensor int_tensor = at::ones({2, 3}, at::TensorOptions().dtype(at::kInt));
  EXPECT_TRUE(int_tensor.is_signed());

  // Test with long tensor (signed)
  at::Tensor long_tensor =
      at::ones({2, 3}, at::TensorOptions().dtype(at::kLong));
  EXPECT_TRUE(long_tensor.is_signed());

  // Test with byte tensor (unsigned)
  at::Tensor byte_tensor =
      at::ones({2, 3}, at::TensorOptions().dtype(at::kByte));
  EXPECT_FALSE(byte_tensor.is_signed());

  // Test with bool tensor (unsigned)
  at::Tensor bool_tensor =
      at::ones({2, 3}, at::TensorOptions().dtype(at::kBool));
  EXPECT_FALSE(bool_tensor.is_signed());
}

}  // namespace test
}  // namespace at
