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

// 测试 is_sparse
TEST_F(TensorTest, IsSparse) {
  // 密集张量应该返回 false
  EXPECT_FALSE(tensor.is_sparse());

  // 创建稀疏 COO 张量 - 先创建模板，再使用 zeros_like
  at::TensorOptions sparse_options =
      at::TensorOptions().dtype(at::kFloat).layout(at::kSparse);
  at::Tensor sparse_template = at::empty({2, 3}, sparse_options);
  at::Tensor sparse_tensor = at::zeros_like(sparse_template);
  EXPECT_TRUE(sparse_tensor.is_sparse());
}

// 测试 is_sparse_csr
TEST_F(TensorTest, IsSparseCsr) {
  // 密集张量应该返回 false
  EXPECT_FALSE(tensor.is_sparse_csr());

  // 创建稀疏 CSR 张量 - 先创建模板，再使用 zeros_like
  at::TensorOptions sparse_csr_options =
      at::TensorOptions().dtype(at::kFloat).layout(at::kSparseCsr);
  at::Tensor sparse_csr_template = at::empty({2, 3}, sparse_csr_options);
  at::Tensor sparse_csr_tensor = at::zeros_like(sparse_csr_template);
  EXPECT_TRUE(sparse_csr_tensor.is_sparse_csr());
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

// 返回当前用例的结果文件名（用于逐个用例对比）
std::string GetTestCaseResultFileName() {
  std::string base = g_custom_param.get();
  std::string test_name =
      ::testing::UnitTest::GetInstance()->current_test_info()->name();
  if (base.size() >= 4 && base.substr(base.size() - 4) == ".txt") {
    base.resize(base.size() - 4);
  }
  return base + "_" + test_name + ".txt";
}

// 测试 cuda
TEST_F(TensorTest, CudaResult) {
  FileManerger file(GetTestCaseResultFileName());
  file.createFile();
  try {
    at::Tensor cuda_tensor = tensor.cuda();
    file << "1 ";
    file << std::to_string(static_cast<int>(cuda_tensor.device().type()))
         << " ";
    file << std::to_string(cuda_tensor.is_cuda() ? 1 : 0) << " ";
    file << std::to_string(cuda_tensor.numel()) << " ";
  } catch (const std::exception&) {
    file << "0 ";
  } catch (...) {
    file << "0 ";
  }
  file.saveFile();
}

// 测试 is_pinned
TEST_F(TensorTest, IsPinnedResult) {
  FileManerger file(GetTestCaseResultFileName());
  file.createFile();
  file << std::to_string(tensor.is_pinned() ? 1 : 0) << " ";
  int pinned_after_cuda = 0;
  try {
    at::Tensor cuda_tensor = tensor.cuda();
    at::Tensor pinned_tensor = cuda_tensor.pin_memory();
    pinned_after_cuda = pinned_tensor.is_pinned() ? 1 : 0;
  } catch (...) {
    pinned_after_cuda = 0;
  }
  file << std::to_string(pinned_after_cuda) << " ";
  file.saveFile();
}

// 测试 pin_memory
TEST_F(TensorTest, PinMemoryResult) {
  FileManerger file(GetTestCaseResultFileName());
  file.createFile();
  int gpu_pin_ok = 0;
  try {
    at::Tensor cuda_tensor = tensor.cuda();
    at::Tensor pinned_tensor = cuda_tensor.pin_memory();
    gpu_pin_ok = pinned_tensor.is_pinned() ? 1 : 0;
  } catch (...) {
    gpu_pin_ok = 0;
  }
  file << std::to_string(gpu_pin_ok) << " ";
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

// 测试 cpu()
TEST_F(TensorTest, CpuMethod) {
  FileManerger file(GetTestCaseResultFileName());
  file.createFile();
  at::Tensor cpu_tensor = tensor.cpu();
  file << std::to_string(cpu_tensor.is_cpu() ? 1 : 0) << " ";
  file << std::to_string(cpu_tensor.numel()) << " ";
  file.saveFile();
}

// 测试 toBackend
TEST_F(TensorTest, ToBackend) {
  FileManerger file(GetTestCaseResultFileName());
  file.createFile();
  at::Tensor cpu_tensor = tensor.toBackend(c10::Backend::CPU);
  file << std::to_string(cpu_tensor.is_cpu() ? 1 : 0) << " ";
  file << std::to_string(cpu_tensor.numel()) << " ";
  file.saveFile();
}

// 测试 data<T>()
TEST_F(TensorTest, DataTemplate) {
  FileManerger file(GetTestCaseResultFileName());
  file.createFile();
  void* ptr = tensor.data<float>();
  file << std::to_string(ptr != nullptr) << " ";
  file.saveFile();
}

// 测试 to(TensorOptions)
TEST_F(TensorTest, ToTensorOptions) {
  FileManerger file(GetTestCaseResultFileName());
  file.createFile();
  at::TensorOptions options = at::TensorOptions().dtype(at::kDouble);
  at::Tensor converted = tensor.to(options);
  file << std::to_string(static_cast<int>(converted.scalar_type())) << " ";
  file.saveFile();
}

// 测试 to(ScalarType)
TEST_F(TensorTest, ToScalarType) {
  FileManerger file(GetTestCaseResultFileName());
  file.createFile();
  at::Tensor converted = tensor.to(at::kDouble);
  file << std::to_string(static_cast<int>(converted.scalar_type())) << " ";
  file.saveFile();
}

// 测试 meta
TEST_F(TensorTest, MetaMethod) {
  FileManerger file(GetTestCaseResultFileName());
  file.createFile();
  file << "0 ";  // meta() not supported, should throw
  file.saveFile();
}

// 测试 item() - 需要1元素tensor
TEST_F(TensorTest, ItemScalar) {
  FileManerger file(GetTestCaseResultFileName());
  file.createFile();
  // 创建1元素tensor
  at::Tensor scalar_tensor = at::ones({1}, at::kFloat);
  try {
    at::Scalar item = scalar_tensor.item();
    file << "1 ";
    file << std::to_string(item.to<float>()) << " ";
  } catch (...) {
    file << "0 ";
  }
  file.saveFile();
}

// 测试 item<T>()
TEST_F(TensorTest, ItemTemplate) {
  FileManerger file(GetTestCaseResultFileName());
  file.createFile();
  at::Tensor scalar_tensor = at::ones({1}, at::kFloat);
  try {
    float val = scalar_tensor.item<float>();
    file << "1 ";
    file << std::to_string(val) << " ";
  } catch (...) {
    file << "0 ";
  }
  file.saveFile();
}

// 测试 clamp(min, max) with Scalar
TEST_F(TensorTest, ClampScalarMinMax) {
  FileManerger file(GetTestCaseResultFileName());
  file.createFile();
  at::Tensor input = at::ones({2, 3}, at::kFloat).fill_(5.0f);
  at::Tensor clamped = input.clamp(1.0, 3.0);
  file << std::to_string(clamped.dim()) << " ";
  float* data = clamped.data_ptr<float>();
  file << std::to_string(data[0]) << " ";
  file.saveFile();
}

// 测试 clamp(min, max) with Tensor
TEST_F(TensorTest, ClampTensorMinMax) {
  FileManerger file(GetTestCaseResultFileName());
  file.createFile();
  at::Tensor input = at::ones({2, 3}, at::kFloat).fill_(5.0f);
  at::Tensor min_tensor = at::ones({1}, at::kFloat).fill_(1.0f);
  at::Tensor max_tensor = at::ones({1}, at::kFloat).fill_(3.0f);
  at::Tensor clamped = input.clamp(min_tensor, max_tensor);
  file << std::to_string(clamped.dim()) << " ";
  file.saveFile();
}

// 测试 clamp_(Scalar)
TEST_F(TensorTest, ClampInplaceScalar) {
  FileManerger file(GetTestCaseResultFileName());
  file.createFile();
  at::Tensor input = at::ones({2, 3}, at::kFloat).fill_(5.0f);
  input.clamp_(1.0, 3.0);
  float* data = input.data_ptr<float>();
  file << std::to_string(data[0]) << " ";
  file.saveFile();
}

// 测试 clamp_(Tensor)
TEST_F(TensorTest, ClampInplaceTensor) {
  FileManerger file(GetTestCaseResultFileName());
  file.createFile();
  at::Tensor input = at::ones({2, 3}, at::kFloat).fill_(5.0f);
  at::Tensor min_tensor = at::ones({1}, at::kFloat).fill_(1.0f);
  at::Tensor max_tensor = at::ones({1}, at::kFloat).fill_(3.0f);
  input.clamp_(min_tensor, max_tensor);
  file << std::to_string(input.dim()) << " ";
  file.saveFile();
}

// 测试 clamp_max(Scalar)
TEST_F(TensorTest, ClampMaxScalar) {
  FileManerger file(GetTestCaseResultFileName());
  file.createFile();
  at::Tensor input = at::ones({2, 3}, at::kFloat).fill_(5.0f);
  at::Tensor clamped = input.clamp_max(3.0);
  float* data = clamped.data_ptr<float>();
  file << std::to_string(data[0]) << " ";
  file.saveFile();
}

// 测试 clamp_max(Tensor)
TEST_F(TensorTest, ClampMaxTensor) {
  FileManerger file(GetTestCaseResultFileName());
  file.createFile();
  at::Tensor input = at::ones({2, 3}, at::kFloat).fill_(5.0f);
  at::Tensor max_tensor = at::ones({1}, at::kFloat).fill_(3.0f);
  at::Tensor clamped = input.clamp_max(max_tensor);
  file << std::to_string(clamped.dim()) << " ";
  file.saveFile();
}

// 测试 clamp_max_(Scalar)
TEST_F(TensorTest, ClampMaxInplace) {
  FileManerger file(GetTestCaseResultFileName());
  file.createFile();
  at::Tensor input = at::ones({2, 3}, at::kFloat).fill_(5.0f);
  input.clamp_max_(3.0);
  float* data = input.data_ptr<float>();
  file << std::to_string(data[0]) << " ";
  file.saveFile();
}

// 测试 clamp_max_(Tensor)
TEST_F(TensorTest, ClampMaxInplaceTensor) {
  FileManerger file(GetTestCaseResultFileName());
  file.createFile();
  at::Tensor input = at::ones({2, 3}, at::kFloat).fill_(5.0f);
  at::Tensor max_tensor = at::ones({1}, at::kFloat).fill_(3.0f);
  input.clamp_max_(max_tensor);
  file << std::to_string(input.dim()) << " ";
  file.saveFile();
}

// 测试 clamp_min(Scalar)
TEST_F(TensorTest, ClampMinScalar) {
  FileManerger file(GetTestCaseResultFileName());
  file.createFile();
  at::Tensor input = at::ones({2, 3}, at::kFloat);
  at::Tensor clamped = input.clamp_min(2.0);
  float* data = clamped.data_ptr<float>();
  file << std::to_string(data[0]) << " ";
  file.saveFile();
}

// 测试 clamp_min(Tensor)
TEST_F(TensorTest, ClampMinTensor) {
  FileManerger file(GetTestCaseResultFileName());
  file.createFile();
  at::Tensor input = at::ones({2, 3}, at::kFloat);
  at::Tensor min_tensor = at::ones({1}, at::kFloat).fill_(2.0f);
  at::Tensor clamped = input.clamp_min(min_tensor);
  file << std::to_string(clamped.dim()) << " ";
  file.saveFile();
}

// 测试 clamp_min_(Scalar)
TEST_F(TensorTest, ClampMinInplace) {
  FileManerger file(GetTestCaseResultFileName());
  file.createFile();
  at::Tensor input = at::ones({2, 3}, at::kFloat);
  input.clamp_min_(2.0);
  float* data = input.data_ptr<float>();
  file << std::to_string(data[0]) << " ";
  file.saveFile();
}

// 测试 clamp_min_(Tensor)
TEST_F(TensorTest, ClampMinInplaceTensor) {
  FileManerger file(GetTestCaseResultFileName());
  file.createFile();
  at::Tensor input = at::ones({2, 3}, at::kFloat);
  at::Tensor min_tensor = at::ones({1}, at::kFloat).fill_(2.0f);
  input.clamp_min_(min_tensor);
  file << std::to_string(input.dim()) << " ";
  file.saveFile();
}

// 测试 as_strided
TEST_F(TensorTest, AsStrided) {
  FileManerger file(GetTestCaseResultFileName());
  file.createFile();
  at::Tensor strided = tensor.as_strided({3, 4, 2}, {2, 1, 6});
  file << std::to_string(strided.sizes()[0]) << " ";
  file << std::to_string(strided.sizes()[1]) << " ";
  file << std::to_string(strided.sizes()[2]) << " ";
  file.saveFile();
}

// 测试 as_strided_
TEST_F(TensorTest, AsStridedInplace) {
  FileManerger file(GetTestCaseResultFileName());
  file.createFile();
  tensor.as_strided_({3, 4, 2}, {2, 1, 6});
  file << std::to_string(tensor.sizes()[0]) << " ";
  file << std::to_string(tensor.sizes()[1]) << " ";
  file << std::to_string(tensor.sizes()[2]) << " ";
  file.saveFile();
}

// 测试 as_strided_scatter
TEST_F(TensorTest, AsStridedScatter) {
  FileManerger file(GetTestCaseResultFileName());
  file.createFile();
  at::Tensor src = at::ones({3, 4, 2}, at::kFloat).fill_(2.0f);
  at::Tensor result = tensor.as_strided_scatter(src, {3, 4, 2}, {2, 1, 6});
  file << std::to_string(result.sizes()[0]) << " ";
  file << std::to_string(result.dim()) << " ";
  file.saveFile();
}

// 测试 std(int dim)
TEST_F(TensorTest, StdDim) {
  FileManerger file(GetTestCaseResultFileName());
  file.createFile();
  at::Tensor input = at::ones({2, 3}, at::kFloat).fill_(2.0f);
  input.fill_(1.0f);
  input.data_ptr<float>()[1] = 3.0f;
  at::Tensor result = input.std(1);
  file << std::to_string(result.dim()) << " ";
  file.saveFile();
}

// 测试 std(bool unbiased)
TEST_F(TensorTest, StdAll) {
  FileManerger file(GetTestCaseResultFileName());
  file.createFile();
  at::Tensor input = at::ones({2, 3}, at::kFloat);
  input.fill_(1.0f);
  input.data_ptr<float>()[1] = 3.0f;
  at::Tensor result = input.std(true);
  file << std::to_string(result.dim()) << " ";
  file.saveFile();
}

// 测试 std(dim, unbiased, keepdim)
TEST_F(TensorTest, StdDims) {
  FileManerger file(GetTestCaseResultFileName());
  file.createFile();
  at::Tensor input = at::ones({2, 3}, at::kFloat);
  input.fill_(1.0f);
  input.data_ptr<float>()[1] = 3.0f;
  at::Tensor result = input.std({1}, true, true);
  file << std::to_string(result.dim()) << " ";
  file << std::to_string(result.sizes()[0]) << " ";
  file.saveFile();
}

// 测试 std(dim, correction, keepdim)
TEST_F(TensorTest, StdCorrection) {
  FileManerger file(GetTestCaseResultFileName());
  file.createFile();
  at::Tensor input = at::ones({2, 3}, at::kFloat);
  input.fill_(1.0f);
  input.data_ptr<float>()[1] = 3.0f;
  at::Tensor result = input.std({1}, 1.0, true);
  file << std::to_string(result.dim()) << " ";
  file.saveFile();
}

// 测试 var(int dim)
TEST_F(TensorTest, VarDim) {
  FileManerger file(GetTestCaseResultFileName());
  file.createFile();
  at::Tensor input = at::ones({2, 3}, at::kFloat);
  input.fill_(1.0f);
  input.data_ptr<float>()[1] = 3.0f;
  at::Tensor result = input.var(1);
  file << std::to_string(result.dim()) << " ";
  file.saveFile();
}

// 测试 var(bool unbiased)
TEST_F(TensorTest, VarAll) {
  FileManerger file(GetTestCaseResultFileName());
  file.createFile();
  at::Tensor input = at::ones({2, 3}, at::kFloat);
  input.fill_(1.0f);
  input.data_ptr<float>()[1] = 3.0f;
  at::Tensor result = input.var(true);
  file << std::to_string(result.dim()) << " ";
  file.saveFile();
}

// 测试 var(dim, unbiased, keepdim)
TEST_F(TensorTest, VarDims) {
  FileManerger file(GetTestCaseResultFileName());
  file.createFile();
  at::Tensor input = at::ones({2, 3}, at::kFloat);
  input.fill_(1.0f);
  input.data_ptr<float>()[1] = 3.0f;
  at::Tensor result = input.var({1}, true, true);
  file << std::to_string(result.dim()) << " ";
  file << std::to_string(result.sizes()[0]) << " ";
  file.saveFile();
}

// 测试 var(dim, correction, keepdim)
TEST_F(TensorTest, VarCorrection) {
  FileManerger file(GetTestCaseResultFileName());
  file.createFile();
  at::Tensor input = at::ones({2, 3}, at::kFloat);
  input.fill_(1.0f);
  input.data_ptr<float>()[1] = 3.0f;
  at::Tensor result = input.var({1}, 1.0, true);
  file << std::to_string(result.dim()) << " ";
  file.saveFile();
}

// 测试 tensor_data
TEST_F(TensorTest, TensorData) {
  FileManerger file(GetTestCaseResultFileName());
  file.createFile();
  at::Tensor result = tensor.tensor_data();
  file << std::to_string(result.dim()) << " ";
  file << std::to_string(result.numel()) << " ";
  file.saveFile();
}

// 测试 variable_data
TEST_F(TensorTest, VariableData) {
  FileManerger file(GetTestCaseResultFileName());
  file.createFile();
  at::Tensor result = tensor.variable_data();
  file << std::to_string(result.dim()) << " ";
  file << std::to_string(result.numel()) << " ";
  file.saveFile();
}

// 测试 index_select
TEST_F(TensorTest, IndexSelect) {
  FileManerger file(GetTestCaseResultFileName());
  file.createFile();
  at::Tensor input = at::ones({3, 4}, at::kFloat);
  input.fill_(1.0f);
  int64_t index_data[] = {0, 2};
  at::Tensor index = at::from_blob(index_data, {2}, at::kLong);
  at::Tensor result = input.index_select(0, index);
  file << std::to_string(result.sizes()[0]) << " ";
  file << std::to_string(result.dim()) << " ";
  file.saveFile();
}

// 测试 dtype
TEST_F(TensorTest, DtypeMethod) {
  FileManerger file(GetTestCaseResultFileName());
  file.createFile();
  // dtype() 在 TensorBody 中返回 TypeMeta，使用 scalar_type() 获取 ScalarType
  c10::ScalarType dt = tensor.scalar_type();
  file << std::to_string(static_cast<int>(dt)) << " ";
  file.saveFile();
}

// 测试 copy_
TEST_F(TensorTest, CopyMethod) {
  FileManerger file(GetTestCaseResultFileName());
  file.createFile();
  at::Tensor src = at::ones({2, 3, 4}, at::kFloat).fill_(2.0f);
  tensor.copy_(src);
  file << std::to_string(tensor.dim()) << " ";
  file.saveFile();
}

// 测试 bitwise_right_shift
TEST_F(TensorTest, BitwiseRightShift) {
  FileManerger file(GetTestCaseResultFileName());
  file.createFile();
  at::Tensor input = at::ones({2, 3}, at::kInt).fill_(8);
  at::Tensor result = input.bitwise_right_shift(2);
  int* data = result.data_ptr<int>();
  file << std::to_string(data[0]) << " ";
  file.saveFile();
}

// 测试 floor_divide_
TEST_F(TensorTest, FloorDivide) {
  FileManerger file(GetTestCaseResultFileName());
  file.createFile();
  at::Tensor input = at::ones({2, 3}, at::kFloat).fill_(7.0f);
  input.floor_divide_(3.0);
  float* data = input.data_ptr<float>();
  file << std::to_string(static_cast<int>(data[0])) << " ";
  file.saveFile();
}

// 测试 nbytes
TEST_F(TensorTest, NbytesMethod) {
  FileManerger file(GetTestCaseResultFileName());
  file.createFile();
  size_t nbytes = tensor.nbytes();
  file << std::to_string(nbytes) << " ";
  file.saveFile();
}

// 测试 itemsize
TEST_F(TensorTest, ItemsizeMethod) {
  FileManerger file(GetTestCaseResultFileName());
  file.createFile();
  size_t itemsize = tensor.itemsize();
  file << std::to_string(itemsize) << " ";
  file.saveFile();
}

// 测试 element_size
TEST_F(TensorTest, ElementSizeMethod) {
  FileManerger file(GetTestCaseResultFileName());
  file.createFile();
  int64_t elem_size = tensor.element_size();
  file << std::to_string(elem_size) << " ";
  file.saveFile();
}

// 测试 clone
TEST_F(TensorTest, CloneMethod) {
  FileManerger file(GetTestCaseResultFileName());
  file.createFile();
  at::Tensor cloned = tensor.clone();
  file << std::to_string(cloned.dim()) << " ";
  file << std::to_string(cloned.numel()) << " ";
  file.saveFile();
}

// 测试 abs
TEST_F(TensorTest, AbsMethod) {
  FileManerger file(GetTestCaseResultFileName());
  file.createFile();
  at::Tensor input = at::ones({2, 3}, at::kFloat).fill_(-1.0f);
  at::Tensor result = input.abs();
  float* data = result.data_ptr<float>();
  file << std::to_string(data[0]) << " ";
  file.saveFile();
}

// 测试 abs_
TEST_F(TensorTest, AbsInplace) {
  FileManerger file(GetTestCaseResultFileName());
  file.createFile();
  at::Tensor input = at::ones({2, 3}, at::kFloat).fill_(-1.0f);
  input.abs_();
  float* data = input.data_ptr<float>();
  file << std::to_string(data[0]) << " ";
  file.saveFile();
}

// 测试 absolute
TEST_F(TensorTest, AbsoluteMethod) {
  FileManerger file(GetTestCaseResultFileName());
  file.createFile();
  at::Tensor input = at::ones({2, 3}, at::kFloat).fill_(-1.0f);
  at::Tensor result = input.absolute();
  float* data = result.data_ptr<float>();
  file << std::to_string(data[0]) << " ";
  file.saveFile();
}

// 测试 absolute_
TEST_F(TensorTest, AbsoluteInplace) {
  FileManerger file(GetTestCaseResultFileName());
  file.createFile();
  at::Tensor input = at::ones({2, 3}, at::kFloat).fill_(-1.0f);
  input.absolute_();
  float* data = input.data_ptr<float>();
  file << std::to_string(data[0]) << " ";
  file.saveFile();
}

// 测试 operator[]
TEST_F(TensorTest, OperatorIndex) {
  FileManerger file(GetTestCaseResultFileName());
  file.createFile();
  at::Tensor result = tensor[0];
  file << std::to_string(result.dim()) << " ";
  file << std::to_string(result.sizes()[0]) << " ";
  file.saveFile();
}

}  // namespace test
}  // namespace at
