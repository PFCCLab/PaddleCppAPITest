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

// 测试 squeeze
TEST_F(TensorTest, Squeeze) {
  // 创建一个包含大小为1的维度的tensor: shape = {2, 1, 3, 1, 4}
  at::Tensor tensor_with_ones = at::ones({2, 1, 3, 1, 4}, at::kFloat);

  // 移除所有大小为1的维度
  at::Tensor squeezed = tensor_with_ones.squeeze();
  EXPECT_EQ(squeezed.dim(), 3);
  EXPECT_EQ(squeezed.sizes()[0], 2);
  EXPECT_EQ(squeezed.sizes()[1], 3);
  EXPECT_EQ(squeezed.sizes()[2], 4);
  EXPECT_EQ(squeezed.numel(), 24);

  // 移除指定维度（维度1，大小为1）
  at::Tensor squeezed_dim1 = tensor_with_ones.squeeze(1);
  EXPECT_EQ(squeezed_dim1.dim(), 4);
  EXPECT_EQ(squeezed_dim1.sizes()[0], 2);
  EXPECT_EQ(squeezed_dim1.sizes()[1], 3);
  EXPECT_EQ(squeezed_dim1.sizes()[2], 1);
  EXPECT_EQ(squeezed_dim1.sizes()[3], 4);
}

// 测试 squeeze_ (原位操作)
TEST_F(TensorTest, SqueezeInplace) {
  // 创建一个包含大小为1的维度的tensor: shape = {2, 1, 3, 1, 4}
  at::Tensor tensor_with_ones = at::ones({2, 1, 3, 1, 4}, at::kFloat);

  // 记录原始数据指针
  void* original_ptr = tensor_with_ones.data_ptr();

  // 原位移除所有大小为1的维度
  tensor_with_ones.squeeze_();
  EXPECT_EQ(tensor_with_ones.dim(), 3);
  EXPECT_EQ(tensor_with_ones.sizes()[0], 2);
  EXPECT_EQ(tensor_with_ones.sizes()[1], 3);
  EXPECT_EQ(tensor_with_ones.sizes()[2], 4);
  EXPECT_EQ(tensor_with_ones.numel(), 24);

  // 验证是原位操作（数据指针未改变）
  EXPECT_EQ(tensor_with_ones.data_ptr(), original_ptr);

  // 测试原位移除指定维度
  at::Tensor tensor_with_ones2 = at::ones({2, 1, 3, 1, 4}, at::kFloat);
  tensor_with_ones2.squeeze_(1);
  EXPECT_EQ(tensor_with_ones2.dim(), 4);
  EXPECT_EQ(tensor_with_ones2.sizes()[1], 3);
}

// 测试 unsqueeze
TEST_F(TensorTest, Unsqueeze) {
  // 在维度0之前添加一个大小为1的维度
  at::Tensor unsqueezed0 = tensor.unsqueeze(0);
  EXPECT_EQ(unsqueezed0.dim(), 4);
  EXPECT_EQ(unsqueezed0.sizes()[0], 1);
  EXPECT_EQ(unsqueezed0.sizes()[1], 2);
  EXPECT_EQ(unsqueezed0.sizes()[2], 3);
  EXPECT_EQ(unsqueezed0.sizes()[3], 4);
  EXPECT_EQ(unsqueezed0.numel(), 24);

  // 在维度2之前添加一个大小为1的维度
  at::Tensor unsqueezed2 = tensor.unsqueeze(2);
  EXPECT_EQ(unsqueezed2.dim(), 4);
  EXPECT_EQ(unsqueezed2.sizes()[0], 2);
  EXPECT_EQ(unsqueezed2.sizes()[1], 3);
  EXPECT_EQ(unsqueezed2.sizes()[2], 1);
  EXPECT_EQ(unsqueezed2.sizes()[3], 4);

  // 在最后添加一个大小为1的维度（使用负索引-1）
  at::Tensor unsqueezed_last = tensor.unsqueeze(-1);
  EXPECT_EQ(unsqueezed_last.dim(), 4);
  EXPECT_EQ(unsqueezed_last.sizes()[0], 2);
  EXPECT_EQ(unsqueezed_last.sizes()[1], 3);
  EXPECT_EQ(unsqueezed_last.sizes()[2], 4);
  EXPECT_EQ(unsqueezed_last.sizes()[3], 1);
}

// 测试 unsqueeze_ (原位操作)
TEST_F(TensorTest, UnsqueezeInplace) {
  // 创建一个新的tensor用于原位操作
  at::Tensor test_tensor = at::ones({2, 3, 4}, at::kFloat);

  // 记录原始数据指针
  void* original_ptr = test_tensor.data_ptr();

  // 原位在维度0之前添加一个大小为1的维度
  test_tensor.unsqueeze_(0);
  EXPECT_EQ(test_tensor.dim(), 4);
  EXPECT_EQ(test_tensor.sizes()[0], 1);
  EXPECT_EQ(test_tensor.sizes()[1], 2);
  EXPECT_EQ(test_tensor.sizes()[2], 3);
  EXPECT_EQ(test_tensor.sizes()[3], 4);
  EXPECT_EQ(test_tensor.numel(), 24);

  // 验证是原位操作（数据指针未改变）
  EXPECT_EQ(test_tensor.data_ptr(), original_ptr);

  // 测试使用负索引的原位操作
  at::Tensor test_tensor2 = at::ones({2, 3, 4}, at::kFloat);
  test_tensor2.unsqueeze_(-1);
  EXPECT_EQ(test_tensor2.dim(), 4);
  EXPECT_EQ(test_tensor2.sizes()[3], 1);
}

}  // namespace test
}  // namespace at
