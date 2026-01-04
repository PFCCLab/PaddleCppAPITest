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

// 测试 cpu
TEST_F(TensorTest, Cpu) {
  at::Tensor cpu_tensor = tensor.cpu();

  EXPECT_TRUE(cpu_tensor.is_cpu());
  EXPECT_EQ(cpu_tensor.device().type(), c10::DeviceType::CPU);
  EXPECT_EQ(cpu_tensor.numel(), tensor.numel());
  EXPECT_FLOAT_EQ(cpu_tensor.data_ptr<float>()[0], tensor.data_ptr<float>()[0]);
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

// 测试 toBackend
TEST_F(TensorTest, ToBackend) {
  // 测试转换到 CPU backend
  at::Tensor cpu_tensor = tensor.toBackend(c10::Backend::CPU);
  EXPECT_TRUE(cpu_tensor.is_cpu());
  EXPECT_EQ(cpu_tensor.device().type(), c10::DeviceType::CPU);
  EXPECT_EQ(cpu_tensor.numel(), tensor.numel());

  // 测试多次调用 toBackend(CPU) - 应该返回相同的 tensor
  at::Tensor cpu_tensor2 = cpu_tensor.toBackend(c10::Backend::CPU);
  EXPECT_EQ(cpu_tensor.data_ptr(), cpu_tensor2.data_ptr());

  // 验证数据内容
  EXPECT_FLOAT_EQ(cpu_tensor.data_ptr<float>()[0], 1.0f);
}

// 测试 item
TEST_F(TensorTest, Item) {
  // 创建一个单元素 tensor
  std::vector<float> single_data = {3.14f};
  paddle::Tensor single_paddle_tensor(std::make_shared<phi::DenseTensor>(
      std::make_shared<phi::Allocation>(single_data.data(),
                                        single_data.size() * sizeof(float),
                                        phi::CPUPlace{}),
      phi::DenseTensorMeta(phi::DataType::FLOAT32, phi::make_ddim({1}))));
  at::Tensor single_tensor(single_paddle_tensor);

  at::Scalar scalar_value = single_tensor.item();
  EXPECT_FLOAT_EQ(scalar_value.to<float>(), 3.14f);

  // 测试多元素 tensor 应该抛出异常
  EXPECT_THROW(tensor.item(), std::exception);

  // 测试不同数据类型 - int32
  at::Tensor int_tensor = at::ones({1}, at::kInt);
  at::Scalar int_scalar = int_tensor.item();
  EXPECT_EQ(int_scalar.to<int>(), 1);

  // 测试不同数据类型 - int64
  at::Tensor long_tensor = at::ones({1}, at::kLong);
  at::Scalar long_scalar = long_tensor.item();
  EXPECT_EQ(long_scalar.to<int64_t>(), 1);

  // 测试不同数据类型 - double
  at::Tensor double_tensor = at::ones({1}, at::kDouble);
  at::Scalar double_scalar = double_tensor.item();
  EXPECT_DOUBLE_EQ(double_scalar.to<double>(), 1.0);

  // 测试不同数据类型 - bool
  at::Tensor bool_tensor = at::ones({1}, at::kBool);
  at::Scalar bool_scalar = bool_tensor.item();
  EXPECT_TRUE(bool_scalar.to<bool>());

  // 测试更多数据类型 - int32 自定义值
  std::vector<int32_t> int32_data = {42};
  paddle::Tensor int32_paddle_tensor(std::make_shared<phi::DenseTensor>(
      std::make_shared<phi::Allocation>(int32_data.data(),
                                        int32_data.size() * sizeof(int32_t),
                                        phi::CPUPlace{}),
      phi::DenseTensorMeta(phi::DataType::INT32, phi::make_ddim({1}))));
  at::Tensor int32_tensor(int32_paddle_tensor);
  EXPECT_EQ(int32_tensor.item().to<int32_t>(), 42);

  // 测试更多数据类型 - int64 大数值
  std::vector<int64_t> int64_data = {123456789L};
  paddle::Tensor int64_paddle_tensor(std::make_shared<phi::DenseTensor>(
      std::make_shared<phi::Allocation>(int64_data.data(),
                                        int64_data.size() * sizeof(int64_t),
                                        phi::CPUPlace{}),
      phi::DenseTensorMeta(phi::DataType::INT64, phi::make_ddim({1}))));
  at::Tensor int64_tensor(int64_paddle_tensor);
  EXPECT_EQ(int64_tensor.item().to<int64_t>(), 123456789L);

  // 测试更多数据类型 - float64
  std::vector<double> float64_data = {2.71828};
  paddle::Tensor float64_paddle_tensor(std::make_shared<phi::DenseTensor>(
      std::make_shared<phi::Allocation>(float64_data.data(),
                                        float64_data.size() * sizeof(double),
                                        phi::CPUPlace{}),
      phi::DenseTensorMeta(phi::DataType::FLOAT64, phi::make_ddim({1}))));
  at::Tensor float64_tensor(float64_paddle_tensor);
  EXPECT_DOUBLE_EQ(float64_tensor.item().to<double>(), 2.71828);

  // 测试更多数据类型 - bool false
  bool bool_false_data = false;
  paddle::Tensor bool_false_paddle_tensor(std::make_shared<phi::DenseTensor>(
      std::make_shared<phi::Allocation>(
          &bool_false_data, sizeof(bool), phi::CPUPlace{}),
      phi::DenseTensorMeta(phi::DataType::BOOL, phi::make_ddim({1}))));
  at::Tensor bool_false_tensor(bool_false_paddle_tensor);
  EXPECT_FALSE(bool_false_tensor.item().to<bool>());

  // 测试多元素 tensor 二维
  at::Tensor multi_elem_2d = at::ones({2, 1}, at::kFloat);
  EXPECT_THROW(multi_elem_2d.item(), std::exception);
}

// 测试 data 方法
TEST_F(TensorTest, Data) {
  // Tensor tensor(paddle_tensor_);

  void* float_data = tensor.data_ptr<float>();
  EXPECT_NE(float_data, nullptr);

  // 验证数据内容
  float* data_as_float = static_cast<float*>(float_data);
  EXPECT_FLOAT_EQ(data_as_float[0], 1.0f);

  // 测试不同类型的 tensor
  at::Tensor int_tensor = at::ones({2, 3}, at::kInt);
  void* int_data = int_tensor.data_ptr<int>();
  EXPECT_NE(int_data, nullptr);

  int* data_as_int = static_cast<int*>(int_data);
  EXPECT_EQ(data_as_int[0], 1);
}

// 测试 meta 方法
TEST_F(TensorTest, Meta) {
  // Tensor tensor(paddle_tensor_);

  // meta() 应该抛出异常，因为 Paddle 不支持
  EXPECT_THROW(tensor.meta(), std::exception);
}

// 测试 to 方法 (TensorOptions 版本)
TEST_F(TensorTest, ToWithOptions) {
  // Tensor tensor(paddle_tensor_);

  // 测试转换到不同的数据类型
  at::Tensor double_tensor = tensor.to(at::TensorOptions().dtype(at::kDouble));
  EXPECT_EQ(double_tensor.dtype(), at::kDouble);
  EXPECT_EQ(double_tensor.numel(), tensor.numel());

  // 测试 copy 参数
  at::Tensor copied_tensor =
      tensor.to(at::TensorOptions().dtype(at::kFloat), false, true);
  EXPECT_EQ(copied_tensor.dtype(), at::kFloat);
  EXPECT_EQ(copied_tensor.numel(), tensor.numel());
  // 验证是复制而不是引用
  EXPECT_NE(copied_tensor.data_ptr(), tensor.data_ptr());
}

// 测试 to 方法 (ScalarType 版本)
TEST_F(TensorTest, ToWithScalarType) {
  // Tensor tensor(paddle_tensor_);

  // 测试转换到 double
  at::Tensor double_tensor = tensor.to(at::kDouble);
  EXPECT_EQ(double_tensor.dtype(), at::kDouble);
  EXPECT_EQ(double_tensor.numel(), tensor.numel());

  // 测试转换到 int
  at::Tensor int_tensor = tensor.to(at::kInt);
  EXPECT_EQ(int_tensor.dtype(), at::kInt);
  EXPECT_EQ(int_tensor.numel(), tensor.numel());

  // 测试转换到 long
  at::Tensor long_tensor = tensor.to(at::kLong);
  EXPECT_EQ(long_tensor.dtype(), at::kLong);
  EXPECT_EQ(long_tensor.numel(), tensor.numel());

  // 验证数据内容 (float 1.0 -> int 1)
  int_tensor.fill_(5.7);  // 5.7 should be truncated to 5
  int* int_data = int_tensor.data_ptr<int>();
  EXPECT_EQ(int_data[0], 5);
}

// 测试 toBackend 行为
TEST_F(TensorTest, ToBackendBehavior) {
  // Tensor tensor(paddle_tensor_);

  // toBackend 总是会复制 tensor（当前实现）
  at::Tensor cpu_tensor1 = tensor.toBackend(c10::Backend::CPU);
  at::Tensor cpu_tensor2 = cpu_tensor1.toBackend(c10::Backend::CPU);

  // 验证都在 CPU 上
  EXPECT_TRUE(cpu_tensor1.is_cpu());
  EXPECT_TRUE(cpu_tensor2.is_cpu());
  EXPECT_EQ(cpu_tensor1.device().type(), c10::DeviceType::CPU);
  EXPECT_EQ(cpu_tensor2.device().type(), c10::DeviceType::CPU);

  // 验证数据内容相同（即使是不同的副本）
  EXPECT_FLOAT_EQ(cpu_tensor1.data_ptr<float>()[0], 1.0f);
  EXPECT_FLOAT_EQ(cpu_tensor2.data_ptr<float>()[0], 1.0f);

  // 验证形状和元素数量相同
  EXPECT_EQ(cpu_tensor1.numel(), tensor.numel());
  EXPECT_EQ(cpu_tensor2.numel(), tensor.numel());
}

// 测试 cpu 行为
TEST_F(TensorTest, CpuBehavior) {
  // Tensor tensor(paddle_tensor_);

  // 第一次调用 cpu()
  at::Tensor cpu_tensor1 = tensor.cpu();
  EXPECT_TRUE(cpu_tensor1.is_cpu());
  EXPECT_EQ(cpu_tensor1.device().type(), c10::DeviceType::CPU);

  // 再次调用 cpu()，当前实现会创建新的副本
  at::Tensor cpu_tensor2 = cpu_tensor1.cpu();
  EXPECT_TRUE(cpu_tensor2.is_cpu());

  // 验证数据内容
  EXPECT_FLOAT_EQ(cpu_tensor1.data_ptr<float>()[0], 1.0f);
  EXPECT_FLOAT_EQ(cpu_tensor2.data_ptr<float>()[0], 1.0f);

  // 验证是有效的 tensor
  EXPECT_EQ(cpu_tensor1.numel(), tensor.numel());
  EXPECT_EQ(cpu_tensor2.numel(), tensor.numel());
  EXPECT_EQ(cpu_tensor1.dim(), tensor.dim());
}

}  // namespace test
}  // namespace at
