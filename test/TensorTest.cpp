#include <ATen/ATen.h>
#include <ATen/core/Tensor.h>
#include <ATen/ops/ones.h>
#include <gtest/gtest.h>
#if !USE_PADDLE_API
#include <torch/all.h>
#endif

#include <string>
#include <vector>
#if USE_PADDLE_API
#include "paddle/phi/api/include/tensor.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/memory/malloc.h"
namespace phi {
inline std::ostream& operator<<(std::ostream& os, AllocationType type) {
  return os << static_cast<int>(type);
}
}  // namespace phi
#endif

#include "../src/file_manager.h"

extern paddle_api_test::ThreadSafeParam g_custom_param;

namespace at {
namespace test {

using paddle_api_test::FileManerger;
using paddle_api_test::ThreadSafeParam;
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
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();
  file << std::to_string(tensor.dim()) << " ";
  file << std::to_string(tensor.numel()) << " ";
  file.saveFile();
}

// 测试 data_ptr
TEST_F(TensorTest, DataPtr) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();
  void* ptr = tensor.data_ptr();
  file << std::to_string(ptr != nullptr) << " ";
  float* float_ptr = tensor.data_ptr<float>();
  file << std::to_string(float_ptr != nullptr) << " ";
  file.saveFile();
}

// 测试 strides
TEST_F(TensorTest, Strides) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();
  c10::IntArrayRef strides = tensor.strides();
  file << std::to_string(strides.size()) << " ";
  file.saveFile();
}

// 测试 sizes
TEST_F(TensorTest, Sizes) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();
  c10::IntArrayRef sizes = tensor.sizes();
  file << std::to_string(sizes.size()) << " ";
  file << std::to_string(sizes[0]) << " ";
  file << std::to_string(sizes[1]) << " ";
  file << std::to_string(sizes[2]) << " ";
  file.saveFile();
}

// 测试 toType
TEST_F(TensorTest, ToType) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();
  Tensor double_tensor = tensor.toType(c10::ScalarType::Double);
  file << std::to_string(static_cast<int>(double_tensor.scalar_type())) << " ";
  file.saveFile();
}

// 测试 numel
TEST_F(TensorTest, Numel) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();
  file << std::to_string(tensor.numel()) << " ";
  file.saveFile();
}

// 测试 device
TEST_F(TensorTest, Device) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();
  c10::Device device = tensor.device();
  file << std::to_string(static_cast<int>(device.type())) << " ";
  file.saveFile();
}

// 测试 get_device
TEST_F(TensorTest, GetDevice) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();
  c10::DeviceIndex device_idx = tensor.get_device();
  file << std::to_string(device_idx) << " ";
  file.saveFile();
}

// 测试 dim 和 ndimension
TEST_F(TensorTest, DimAndNdimension) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();
  file << std::to_string(tensor.dim()) << " ";
  file << std::to_string(tensor.ndimension()) << " ";
  file.saveFile();
}

// 测试 contiguous
TEST_F(TensorTest, Contiguous) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();
  at::Tensor cont_tensor = tensor.contiguous();
  file << std::to_string(cont_tensor.is_contiguous()) << " ";
  file.saveFile();
}

// 测试 is_contiguous
TEST_F(TensorTest, IsContiguous) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();
  file << std::to_string(tensor.is_contiguous()) << " ";
  file.saveFile();
}

// 测试 scalar_type
TEST_F(TensorTest, ScalarType) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();
  c10::ScalarType stype = tensor.scalar_type();
  file << std::to_string(static_cast<int>(stype)) << " ";
  file.saveFile();
}

// 测试 fill_
TEST_F(TensorTest, Fill) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();
  tensor.fill_(5.0);
  float* data = tensor.data_ptr<float>();
  file << std::to_string(data[0]) << " ";
  file.saveFile();
}

// 测试 zero_
TEST_F(TensorTest, Zero) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();
  tensor.zero_();
  float* data = tensor.data_ptr<float>();
  file << std::to_string(data[0]) << " ";
  file.saveFile();
}

// 测试 is_cpu
TEST_F(TensorTest, IsCpu) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();
  file << std::to_string(tensor.is_cpu()) << " ";
  file.saveFile();
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
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();
  file << std::to_string(tensor.is_cuda()) << " ";
  file.saveFile();
}

// 测试 reshape
TEST_F(TensorTest, Reshape) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();
  at::Tensor reshaped = tensor.reshape({6, 4});
  file << std::to_string(reshaped.sizes()[0]) << " ";
  file << std::to_string(reshaped.sizes()[1]) << " ";
  file << std::to_string(reshaped.numel()) << " ";
  file.saveFile();
}

// 测试 transpose
TEST_F(TensorTest, Transpose) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();
  at::Tensor transposed = tensor.transpose(0, 2);
  file << std::to_string(transposed.sizes()[0]) << " ";
  file << std::to_string(transposed.sizes()[2]) << " ";
  file.saveFile();
}

// 测试 toBackend
TEST_F(TensorTest, ToBackend) {
  // 测试转换到 CPU backend
  at::Tensor cpu_tensor = tensor.toBackend(c10::Backend::CPU);
  EXPECT_TRUE(cpu_tensor.is_cpu());
  EXPECT_EQ(cpu_tensor.device().type(), c10::DeviceType::CPU);
  EXPECT_EQ(cpu_tensor.numel(), tensor.numel());

  // 测试多次调用 toBackend(CPU) - 当前实现会创建新的副本
  at::Tensor cpu_tensor2 = cpu_tensor.toBackend(c10::Backend::CPU);
  // 验证都在 CPU 上且数据内容相同
  EXPECT_TRUE(cpu_tensor2.is_cpu());
  EXPECT_FLOAT_EQ(cpu_tensor2.data_ptr<float>()[0], 1.0f);

  // 验证数据内容
  EXPECT_FLOAT_EQ(cpu_tensor.data_ptr<float>()[0], 1.0f);
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

// 测试 cuda 方法
TEST_F(TensorTest, Cuda) {
  try {
    at::Tensor cuda_tensor = tensor.cuda();

    EXPECT_TRUE(cuda_tensor.is_cuda());
    EXPECT_EQ(cuda_tensor.device().type(), c10::DeviceType::CUDA);
    EXPECT_EQ(cuda_tensor.numel(), tensor.numel());

    at::Tensor cpu_check = cuda_tensor.cpu();
    EXPECT_FLOAT_EQ(cpu_check.data_ptr<float>()[0], 1.0f);

    at::Tensor cuda_tensor2 = cuda_tensor.cuda();
    EXPECT_TRUE(cuda_tensor2.is_cuda());
    EXPECT_EQ(cuda_tensor2.device().type(), c10::DeviceType::CUDA);
    EXPECT_EQ(cuda_tensor2.numel(), cuda_tensor.numel());
  } catch (const std::exception& e) {
    GTEST_SKIP() << "CUDA not available: " << e.what();
  } catch (...) {
    GTEST_SKIP() << "CUDA test failed with unknown error";
  }
}

// 测试 is_pinned 方法
TEST_F(TensorTest, IsPinned) {
  EXPECT_FALSE(tensor.is_pinned());

#ifdef PADDLE_WITH_CUDA

  at::Tensor pinned_tensor = tensor.pin_memory();
  EXPECT_TRUE(pinned_tensor.is_pinned());

  at::Tensor cuda_tensor = tensor.cuda();
  EXPECT_FALSE(cuda_tensor.is_pinned());
#endif
}

// 测试 pin_memory 方法
TEST_F(TensorTest, PinMemory) {
  try {
    at::Tensor pinned_tensor = tensor.pin_memory();
    EXPECT_TRUE(pinned_tensor.is_pinned());
    EXPECT_FALSE(pinned_tensor.is_cuda());
    EXPECT_EQ(pinned_tensor.numel(), tensor.numel());

    EXPECT_FLOAT_EQ(pinned_tensor.data_ptr<float>()[0], 1.0f);

    at::Tensor pinned_tensor2 = pinned_tensor.pin_memory();
    EXPECT_TRUE(pinned_tensor2.is_pinned());

    try {
      at::Tensor cuda_tensor = tensor.cuda();
    } catch (...) {
      // CUDA 不可用时跳过此测试
    }
  } catch (const std::exception& e) {
    GTEST_SKIP() << "Pinned memory not available: " << e.what();
  }
}

}  // namespace test
}  // namespace at
