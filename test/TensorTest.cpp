#include <ATen/ATen.h>
#include <ATen/core/Tensor.h>
#include <ATen/ops/ones.h>
#include <gtest/gtest.h>
#include <torch/all.h>

#include <vector>

namespace at {
namespace test {

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

// 测试 sym_size
TEST_F(TensorTest, SymSize) {
  // 获取符号化的单个维度大小
  c10::SymInt sym_size_0 = tensor.sym_size(0);
  c10::SymInt sym_size_1 = tensor.sym_size(1);
  c10::SymInt sym_size_2 = tensor.sym_size(2);

  // 验证符号化大小与实际大小一致
  EXPECT_EQ(sym_size_0, 2);
  EXPECT_EQ(sym_size_1, 3);
  EXPECT_EQ(sym_size_2, 4);

  // 测试负索引
  c10::SymInt sym_size_neg1 = tensor.sym_size(-1);
  EXPECT_EQ(sym_size_neg1, 4);
}

// 测试 sym_stride
TEST_F(TensorTest, SymStride) {
  // 获取符号化的单个维度步长
  c10::SymInt sym_stride_0 = tensor.sym_stride(0);
  c10::SymInt sym_stride_1 = tensor.sym_stride(1);
  c10::SymInt sym_stride_2 = tensor.sym_stride(2);

  // 验证符号化步长
  EXPECT_GT(sym_stride_0, 0);
  EXPECT_GT(sym_stride_1, 0);
  EXPECT_GT(sym_stride_2, 0);

  // 测试负索引
  c10::SymInt sym_stride_neg1 = tensor.sym_stride(-1);
  EXPECT_EQ(sym_stride_neg1, 1);  // 最后一维步长通常为1
}

// 测试 sym_sizes
TEST_F(TensorTest, SymSizes) {
  // 获取符号化的所有维度大小
  c10::SymIntArrayRef sym_sizes = tensor.sym_sizes();

  // 验证维度数量
  EXPECT_EQ(sym_sizes.size(), 3U);

  // 验证每个维度的大小
  EXPECT_EQ(sym_sizes[0], 2);
  EXPECT_EQ(sym_sizes[1], 3);
  EXPECT_EQ(sym_sizes[2], 4);
}

// 测试 sym_strides
TEST_F(TensorTest, SymStrides) {
  // 获取符号化的所有维度步长
  c10::SymIntArrayRef sym_strides = tensor.sym_strides();

  // 验证维度数量
  EXPECT_EQ(sym_strides.size(), 3U);

  // 验证步长值都大于0
  for (size_t i = 0; i < sym_strides.size(); ++i) {
    EXPECT_GT(sym_strides[i], 0);
  }
}

// 测试 sym_numel
TEST_F(TensorTest, SymNumel) {
  // 获取符号化的元素总数
  c10::SymInt sym_numel = tensor.sym_numel();

  // 验证符号化元素数与实际元素数一致
  EXPECT_EQ(sym_numel, 24);  // 2*3*4

  // 验证与 numel() 结果一致
  EXPECT_EQ(sym_numel, tensor.numel());
}

}  // namespace test
}  // namespace at
