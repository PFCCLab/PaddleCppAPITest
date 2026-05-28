#include <ATen/ATen.h>
#include <ATen/core/Tensor.h>
#include <ATen/ops/ones.h>
#include <c10/core/Stream.h>

#if defined(__has_include)
#if __has_include(<c10/cuda/CUDAStream.h>) && \
    __has_include(<c10/cuda/impl/cuda_cmake_macros.h>)
#define PCAT_HAS_TORCH_CUDA_STREAM 1
#include <c10/cuda/CUDAStream.h>
#else
#define PCAT_HAS_TORCH_CUDA_STREAM 0
#endif
#else
#define PCAT_HAS_TORCH_CUDA_STREAM 1
#include <c10/cuda/CUDAStream.h>
#endif

#include <gtest/gtest.h>
#include <torch/all.h>

#include <cstdint>
#include <string>
#include <vector>

#include "src/file_manager.h"

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
  file << "ConstructFromPaddleTensor ";
  file << std::to_string(tensor.dim()) << " ";
  file << std::to_string(tensor.numel()) << " ";
  file << "\n";
  file.saveFile();
}

// 测试 data_ptr
TEST_F(TensorTest, DataPtr) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "DataPtr ";
  void* ptr = tensor.data_ptr();
  file << std::to_string(ptr != nullptr) << " ";
  float* float_ptr = tensor.data_ptr<float>();
  file << std::to_string(float_ptr != nullptr) << " ";
  file << "\n";
  file.saveFile();
}

// 测试 strides
TEST_F(TensorTest, Strides) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "Strides ";
  c10::IntArrayRef strides = tensor.strides();
  file << std::to_string(strides.size()) << " ";
  file << "\n";
  file.saveFile();
}

// 测试 sizes
TEST_F(TensorTest, Sizes) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "Sizes ";
  c10::IntArrayRef sizes = tensor.sizes();
  file << std::to_string(sizes.size()) << " ";
  file << std::to_string(sizes[0]) << " ";
  file << std::to_string(sizes[1]) << " ";
  file << std::to_string(sizes[2]) << " ";
  file << "\n";
  file.saveFile();
}

// 测试 toType
TEST_F(TensorTest, ToType) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "ToType ";
  Tensor double_tensor = tensor.toType(c10::ScalarType::Double);
  file << std::to_string(static_cast<int>(double_tensor.scalar_type())) << " ";
  file << "\n";
  file.saveFile();
}

// 测试 numel
TEST_F(TensorTest, Numel) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "Numel ";
  file << std::to_string(tensor.numel()) << " ";
  file << "\n";
  file.saveFile();
}

// 测试 device
TEST_F(TensorTest, Device) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "Device ";
  c10::Device device = tensor.device();
  file << std::to_string(static_cast<int>(device.type())) << " ";
  file << "\n";
  file.saveFile();
}

// 测试 get_device
TEST_F(TensorTest, GetDevice) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "GetDevice ";
  c10::DeviceIndex device_idx = tensor.get_device();
  file << std::to_string(device_idx) << " ";
  file << "\n";
  file.saveFile();
}

// 测试 dim 和 ndimension
TEST_F(TensorTest, DimAndNdimension) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "DimAndNdimension ";
  file << std::to_string(tensor.dim()) << " ";
  file << std::to_string(tensor.ndimension()) << " ";
  file << "\n";
  file.saveFile();
}

// 测试 contiguous
TEST_F(TensorTest, Contiguous) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "Contiguous ";
  at::Tensor cont_tensor = tensor.contiguous();
  file << std::to_string(cont_tensor.is_contiguous()) << " ";
  file << "\n";
  file.saveFile();
}

// 测试 is_contiguous
TEST_F(TensorTest, IsContiguous) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "IsContiguous ";
  file << std::to_string(tensor.is_contiguous()) << " ";
  file << "\n";
  file.saveFile();
}

// 测试 scalar_type
TEST_F(TensorTest, ScalarType) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "ScalarType ";
  c10::ScalarType stype = tensor.scalar_type();
  file << std::to_string(static_cast<int>(stype)) << " ";
  file << "\n";
  file.saveFile();
}

// 测试 fill_
TEST_F(TensorTest, Fill) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "Fill ";
  tensor.fill_(5.0);
  float* data = tensor.data_ptr<float>();
  file << std::to_string(data[0]) << " ";
  file << "\n";
  file.saveFile();
}

// 测试 zero_
TEST_F(TensorTest, Zero) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "Zero ";
  tensor.zero_();
  float* data = tensor.data_ptr<float>();
  file << std::to_string(data[0]) << " ";
  file << "\n";
  file.saveFile();
}

// 测试 is_cpu
TEST_F(TensorTest, IsCpu) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "IsCpu ";
  file << std::to_string(tensor.is_cpu()) << " ";
  file << "\n";
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
  file.openAppend();
  file << "IsCuda ";
  file << std::to_string(tensor.is_cuda()) << " ";
  file << "\n";
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
  file.openAppend();
  file << "Reshape ";
  at::Tensor reshaped = tensor.reshape({6, 4});
  file << std::to_string(reshaped.sizes()[0]) << " ";
  file << std::to_string(reshaped.sizes()[1]) << " ";
  file << std::to_string(reshaped.numel()) << " ";
  file << "\n";
  file.saveFile();
}

// 测试 transpose
TEST_F(TensorTest, Transpose) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "Transpose ";
  at::Tensor transposed = tensor.transpose(0, 2);
  file << std::to_string(transposed.sizes()[0]) << " ";
  file << std::to_string(transposed.sizes()[2]) << " ";
  file << "\n";
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
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "CudaResult ";
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
  file << "\n";
  file.saveFile();
}

// 测试 record_stream
TEST_F(TensorTest, RecordStreamResult) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "RecordStreamResult ";
#if PCAT_HAS_TORCH_CUDA_STREAM
  try {
    at::Tensor cuda_tensor = tensor.cuda();
    at::Stream stream = c10::cuda::getCurrentCUDAStream(0);
    cuda_tensor.record_stream(stream);
    file << "1 ";
  } catch (const std::exception&) {
    file << "0 ";
  } catch (...) {
    file << "0 ";
  }
#else
  file << "0 ";
#endif
  file << "\n";
  file.saveFile();
}

// 测试 register_hook 在不需要梯度的 tensor 上抛异常
TEST_F(TensorTest, RegisterHookNoGradResult) {
  FileManerger file(GetTestCaseResultFileName());
  file.openAppend();
  file << "RegisterHookNoGradResult ";
  try {
    auto handle =
        tensor.register_hook([](const at::Tensor& grad) { return grad; });
    file << "0 ";
    file << std::to_string(handle) << " ";
  } catch (const std::exception&) {
    file << "1 ";
  }
  file << "\n";
  file.saveFile();
}

// 测试 is_pinned
TEST_F(TensorTest, IsPinnedResult) {
  FileManerger file(GetTestCaseResultFileName());
  file.openAppend();
  file << "IsPinnedResult ";
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
  file << "\n";
  file.saveFile();
}

// 测试 pin_memory
TEST_F(TensorTest, PinMemoryResult) {
  FileManerger file(GetTestCaseResultFileName());
  file.openAppend();
  file << "PinMemoryResult ";
  int gpu_pin_ok = 0;
  try {
    at::Tensor cuda_tensor = tensor.cuda();
    at::Tensor pinned_tensor = cuda_tensor.pin_memory();
    gpu_pin_ok = pinned_tensor.is_pinned() ? 1 : 0;
  } catch (...) {
    gpu_pin_ok = 0;
  }
  file << std::to_string(gpu_pin_ok) << " ";
  file << "\n";
  file.saveFile();
}

// 测试 sym_size
TEST_F(TensorTest, SymSize) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "SymSize ";
  // 获取符号化的单个维度大小
  c10::SymInt sym_size_0 = tensor.sym_size(0);
  c10::SymInt sym_size_1 = tensor.sym_size(1);
  c10::SymInt sym_size_2 = tensor.sym_size(2);
  file << std::to_string(sym_size_0.guard_int(__FILE__, __LINE__)) << " ";
  file << std::to_string(sym_size_1.guard_int(__FILE__, __LINE__)) << " ";
  file << std::to_string(sym_size_2.guard_int(__FILE__, __LINE__)) << " ";
  // 测试负索引
  c10::SymInt sym_size_neg1 = tensor.sym_size(-1);
  file << std::to_string(sym_size_neg1.guard_int(__FILE__, __LINE__)) << " ";
  file << "\n";
  file.saveFile();
}

// 测试 sym_stride
TEST_F(TensorTest, SymStride) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "SymStride ";
  // 获取符号化的单个维度步长
  c10::SymInt sym_stride_0 = tensor.sym_stride(0);
  c10::SymInt sym_stride_1 = tensor.sym_stride(1);
  c10::SymInt sym_stride_2 = tensor.sym_stride(2);
  file << std::to_string(sym_stride_0.guard_int(__FILE__, __LINE__)) << " ";
  file << std::to_string(sym_stride_1.guard_int(__FILE__, __LINE__)) << " ";
  file << std::to_string(sym_stride_2.guard_int(__FILE__, __LINE__)) << " ";
  // 测试负索引
  c10::SymInt sym_stride_neg1 = tensor.sym_stride(-1);
  file << std::to_string(sym_stride_neg1.guard_int(__FILE__, __LINE__)) << " ";
  file << "\n";
  file.saveFile();
}

// 测试 sym_sizes
TEST_F(TensorTest, SymSizes) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "SymSizes ";
  // 获取符号化的所有维度大小
  c10::SymIntArrayRef sym_sizes = tensor.sym_sizes();
  file << std::to_string(sym_sizes.size()) << " ";
  for (size_t i = 0; i < sym_sizes.size(); ++i) {
    file << std::to_string(sym_sizes[i].guard_int(__FILE__, __LINE__)) << " ";
  }
  file << "\n";
  file.saveFile();
}

// 测试 sym_strides
TEST_F(TensorTest, SymStrides) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "SymStrides ";
  // 获取符号化的所有维度步长
  c10::SymIntArrayRef sym_strides = tensor.sym_strides();
  file << std::to_string(sym_strides.size()) << " ";
  for (size_t i = 0; i < sym_strides.size(); ++i) {
    file << std::to_string(sym_strides[i].guard_int(__FILE__, __LINE__)) << " ";
  }
  file << "\n";
  file.saveFile();
}

// 测试 sym_numel
TEST_F(TensorTest, SymNumel) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "SymNumel ";
  // 获取符号化的元素总数
  c10::SymInt sym_numel = tensor.sym_numel();
  file << std::to_string(sym_numel.guard_int(__FILE__, __LINE__)) << " ";
  file << std::to_string(tensor.numel()) << " ";
  file << "\n";
  file.saveFile();
}

// 测试 cpu()
TEST_F(TensorTest, CpuMethod) {
  FileManerger file(GetTestCaseResultFileName());
  file.openAppend();
  file << "CpuMethod ";
  at::Tensor cpu_tensor = tensor.cpu();
  file << std::to_string(cpu_tensor.is_cpu() ? 1 : 0) << " ";
  file << std::to_string(cpu_tensor.numel()) << " ";
  file << "\n";
  file.saveFile();
}

// 测试 toBackend
TEST_F(TensorTest, ToBackend) {
  FileManerger file(GetTestCaseResultFileName());
  file.openAppend();
  file << "ToBackend ";
  at::Tensor cpu_tensor = tensor.toBackend(c10::Backend::CPU);
  file << std::to_string(cpu_tensor.is_cpu() ? 1 : 0) << " ";
  file << std::to_string(cpu_tensor.numel()) << " ";
  file << "\n";
  file.saveFile();
}

// 测试 data<T>()
TEST_F(TensorTest, DataTemplate) {
  FileManerger file(GetTestCaseResultFileName());
  file.openAppend();
  file << "DataTemplate ";
  void* ptr = tensor.data_ptr<float>();
  file << std::to_string(ptr != nullptr) << " ";
  file << "\n";
  file.saveFile();
}

// 测试 to(TensorOptions)
TEST_F(TensorTest, ToTensorOptions) {
  FileManerger file(GetTestCaseResultFileName());
  file.openAppend();
  file << "ToTensorOptions ";
  at::TensorOptions options = at::TensorOptions().dtype(at::kDouble);
  at::Tensor converted = tensor.to(options);
  file << std::to_string(static_cast<int>(converted.scalar_type())) << " ";
  file << "\n";
  file.saveFile();
}

// 测试 to(ScalarType)
TEST_F(TensorTest, ToScalarType) {
  FileManerger file(GetTestCaseResultFileName());
  file.openAppend();
  file << "ToScalarType ";
  at::Tensor converted = tensor.to(at::kDouble);
  file << std::to_string(static_cast<int>(converted.scalar_type())) << " ";
  file << "\n";
  file.saveFile();
}

// 测试 meta
TEST_F(TensorTest, MetaMethod) {
  // [DIFF] Paddle 没有 meta 设备，也不对齐 Tensor::meta() 语义；
  // 该用例保留为已知剩余差异。
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "MetaMethod ";
  try {
    at::Tensor meta_tensor = tensor.meta();
    file << "1 ";
    file << std::to_string(static_cast<int>(meta_tensor.device().type()))
         << " ";
    file << std::to_string(meta_tensor.numel()) << " ";
  } catch (const std::exception&) {
    file << "0 ";
  } catch (...) {
    file << "0 ";
  }
  file << "\n";
  file.saveFile();
}

// 测试 item() - 需要1元素tensor
TEST_F(TensorTest, ItemScalar) {
  FileManerger file(GetTestCaseResultFileName());
  file.openAppend();
  file << "ItemScalar ";
  // 创建1元素tensor
  at::Tensor scalar_tensor = at::ones({1}, at::kFloat);
  try {
    at::Scalar item = scalar_tensor.item();
    file << "1 ";
    file << std::to_string(item.to<float>()) << " ";
  } catch (...) {
    file << "0 ";
  }
  file << "\n";
  file.saveFile();
}

// 测试 item<T>()
TEST_F(TensorTest, ItemTemplate) {
  FileManerger file(GetTestCaseResultFileName());
  file.openAppend();
  file << "ItemTemplate ";
  at::Tensor scalar_tensor = at::ones({1}, at::kFloat);
  try {
    float val = scalar_tensor.item<float>();
    file << "1 ";
    file << std::to_string(val) << " ";
  } catch (...) {
    file << "0 ";
  }
  file << "\n";
  file.saveFile();
}

// 测试 toBackend
TEST_F(TensorTest, ToBackendExpect) {
  FileManerger file(GetTestCaseResultFileName());
  file.openAppend();
  file << "ToBackendExpect ";

  at::Tensor cpu_tensor = tensor.toBackend(c10::Backend::CPU);
  file << std::to_string(cpu_tensor.is_cpu() ? 1 : 0) << " ";
  file << std::to_string(static_cast<int>(cpu_tensor.device().type())) << " ";
  file << std::to_string(cpu_tensor.numel()) << " ";

  at::Tensor cpu_tensor2 = cpu_tensor.toBackend(c10::Backend::CPU);
  file << std::to_string(cpu_tensor2.is_cpu() ? 1 : 0) << " ";
  file << std::to_string(static_cast<int>(cpu_tensor2.scalar_type())) << " ";
  file << std::to_string(cpu_tensor2.numel()) << " ";

  file << std::to_string(cpu_tensor.data_ptr<float>()[0]) << " ";
  file << "\n";
  file.saveFile();
}

// 测试 item
TEST_F(TensorTest, Item) {
  FileManerger file(GetTestCaseResultFileName());
  file.openAppend();
  file << "Item ";

  at::Tensor single_tensor = at::ones({1}, at::kFloat).fill_(3.14f);
  try {
    at::Scalar scalar_value = single_tensor.item();
    file << "1 " << std::to_string(scalar_value.to<float>()) << " ";
  } catch (...) {
    file << "0 ";
  }

  try {
    (void)tensor.item();
    file << "0 ";
  } catch (...) {
    file << "1 ";
  }

  at::Tensor int_tensor = at::ones({1}, at::kInt);
  file << std::to_string(int_tensor.item().to<int>()) << " ";

  at::Tensor long_tensor = at::ones({1}, at::kLong);
  file << std::to_string(long_tensor.item().to<int64_t>()) << " ";

  at::Tensor double_tensor = at::ones({1}, at::kDouble);
  file << std::to_string(double_tensor.item().to<double>()) << " ";

  at::Tensor bool_tensor = at::ones({1}, at::kBool);
  file << std::to_string(bool_tensor.item().to<bool>() ? 1 : 0) << " ";

  at::Tensor int32_tensor = at::ones({1}, at::kInt).fill_(42);
  file << std::to_string(int32_tensor.item().to<int32_t>()) << " ";

  at::Tensor int64_tensor = at::ones({1}, at::kLong).fill_(123456789L);
  file << std::to_string(int64_tensor.item().to<int64_t>()) << " ";

  at::Tensor float64_tensor = at::ones({1}, at::kDouble).fill_(2.71828);
  file << std::to_string(float64_tensor.item().to<double>()) << " ";

  at::Tensor bool_false_tensor = at::zeros({1}, at::kBool);
  file << std::to_string(bool_false_tensor.item().to<bool>() ? 1 : 0) << " ";

  at::Tensor multi_elem_2d = at::ones({2, 1}, at::kFloat);
  try {
    (void)multi_elem_2d.item();
    file << "0 ";
  } catch (...) {
    file << "1 ";
  }

  file << "\n";
  file.saveFile();
}

// 测试 data 方法
TEST_F(TensorTest, Data) {
  FileManerger file(GetTestCaseResultFileName());
  file.openAppend();
  file << "Data ";

  void* float_data = tensor.data_ptr<float>();
  file << std::to_string(float_data != nullptr ? 1 : 0) << " ";

  float* data_as_float = static_cast<float*>(float_data);
  file << std::to_string(data_as_float[0]) << " ";

  at::Tensor int_tensor = at::ones({2, 3}, at::kInt);
  void* int_data = int_tensor.data_ptr<int>();
  file << std::to_string(int_data != nullptr ? 1 : 0) << " ";

  int* data_as_int = static_cast<int*>(int_data);
  file << std::to_string(data_as_int[0]) << " ";
  file << "\n";
  file.saveFile();
}

// 测试 meta 方法
TEST_F(TensorTest, Meta) {
  FileManerger file(GetTestCaseResultFileName());
  file.openAppend();
  file << "Meta ";

  try {
    (void)tensor.meta();
    file << "0 ";
  } catch (const std::exception&) {
    file << "1 ";
  }
  file << "\n";
  file.saveFile();
}

// 测试 to 方法 (TensorOptions 版本)
TEST_F(TensorTest, ToWithOptions) {
  FileManerger file(GetTestCaseResultFileName());
  file.openAppend();
  file << "ToWithOptions ";

  at::Tensor double_tensor = tensor.to(at::TensorOptions().dtype(at::kDouble));
  file << std::to_string(static_cast<int>(double_tensor.scalar_type())) << " ";
  file << std::to_string(double_tensor.numel()) << " ";

  at::Tensor copied_tensor =
      tensor.to(at::TensorOptions().dtype(at::kFloat), false, true);
  file << std::to_string(static_cast<int>(copied_tensor.scalar_type())) << " ";
  file << std::to_string(copied_tensor.numel()) << " ";
  file << "\n";
  file.saveFile();
}

// 测试 to 方法 (ScalarType 版本)
TEST_F(TensorTest, ToWithScalarType) {
  FileManerger file(GetTestCaseResultFileName());
  file.openAppend();
  file << "ToWithScalarType ";

  at::Tensor double_tensor = tensor.to(at::kDouble);
  file << std::to_string(static_cast<int>(double_tensor.scalar_type())) << " ";
  file << std::to_string(double_tensor.numel()) << " ";

  at::Tensor int_tensor = tensor.to(at::kInt);
  file << std::to_string(static_cast<int>(int_tensor.scalar_type())) << " ";
  file << std::to_string(int_tensor.numel()) << " ";

  at::Tensor long_tensor = tensor.to(at::kLong);
  file << std::to_string(static_cast<int>(long_tensor.scalar_type())) << " ";
  file << std::to_string(long_tensor.numel()) << " ";

  int_tensor.fill_(5.7);
  int* int_data = int_tensor.data_ptr<int>();
  file << std::to_string(int_data[0]) << " ";
  file << "\n";
  file.saveFile();
}

// 测试 toBackend 行为
TEST_F(TensorTest, ToBackendBehavior) {
  FileManerger file(GetTestCaseResultFileName());
  file.openAppend();
  file << "ToBackendBehavior ";

  at::Tensor cpu_tensor1 = tensor.toBackend(c10::Backend::CPU);
  at::Tensor cpu_tensor2 = cpu_tensor1.toBackend(c10::Backend::CPU);

  file << std::to_string(cpu_tensor1.is_cpu() ? 1 : 0) << " ";
  file << std::to_string(cpu_tensor2.is_cpu() ? 1 : 0) << " ";
  file << std::to_string(cpu_tensor1.data_ptr<float>()[0]) << " ";
  file << std::to_string(cpu_tensor2.data_ptr<float>()[0]) << " ";
  file << std::to_string(cpu_tensor1.numel()) << " ";
  file << std::to_string(cpu_tensor2.numel()) << " ";
  file << "\n";
  file.saveFile();
}

// 测试 cpu 行为
TEST_F(TensorTest, CpuBehavior) {
  FileManerger file(GetTestCaseResultFileName());
  file.openAppend();
  file << "CpuBehavior ";

  at::Tensor cpu_tensor1 = tensor.cpu();

  at::Tensor cpu_tensor2 = cpu_tensor1.cpu();

  file << std::to_string(cpu_tensor1.is_cpu() ? 1 : 0) << " ";
  file << std::to_string(cpu_tensor2.is_cpu() ? 1 : 0) << " ";
  file << std::to_string(cpu_tensor1.data_ptr<float>()[0]) << " ";
  file << std::to_string(cpu_tensor2.data_ptr<float>()[0]) << " ";
  file << std::to_string(cpu_tensor1.numel()) << " ";
  file << std::to_string(cpu_tensor2.numel()) << " ";
  file << std::to_string(cpu_tensor1.dim()) << " ";
  file << "\n";
  file.saveFile();
}

// 测试 defined
TEST_F(TensorTest, Defined) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "Defined ";
  file << std::to_string(tensor.defined()) << " ";
  file << "\n";
  file.saveFile();
}

// 测试 reset
TEST_F(TensorTest, Reset) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "Reset ";
  tensor.reset();
  file << std::to_string(tensor.defined()) << " ";
  file << "\n";
  file.saveFile();
}

}  // namespace test
}  // namespace at
