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

// 测试 any
TEST_F(TensorTest, Any) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();
  at::Tensor test_tensor = at::ones({2, 2}, at::kFloat);
  test_tensor.fill_(0.0);
  test_tensor.data_ptr<float>()[0] = 1.0;
  bool any_result = test_tensor.any().item<bool>();
  file << std::to_string(any_result) << " ";
  auto any_dim_result = test_tensor.any(0);
  file << std::to_string(any_dim_result.sizes()[0]) << " ";
  file.saveFile();
}

// 测试 chunk
TEST_F(TensorTest, Chunk) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();
  at::Tensor test_tensor = at::ones({4, 4}, at::kFloat);
  std::vector<at::Tensor> chunks = test_tensor.chunk(2, 0);
  file << std::to_string(chunks.size()) << " ";
  file << std::to_string(chunks[0].sizes()[0]) << " ";
  file << std::to_string(chunks[1].sizes()[0]) << " ";
  file.saveFile();
}

// 测试 rename - Paddle不支持Dimname，返回原tensor
TEST_F(TensorTest, Rename) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();
  at::Tensor renamed = tensor.rename(std::nullopt);
  file << std::to_string(renamed.sizes().size()) << " ";
  file.saveFile();
}

// 测试 new_empty
TEST_F(TensorTest, NewEmpty) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();
  at::Tensor empty_tensor = tensor.new_empty({3, 4});
  file << std::to_string(empty_tensor.sizes()[0]) << " ";
  file << std::to_string(empty_tensor.sizes()[1]) << " ";
  file << std::to_string(empty_tensor.dtype() == tensor.dtype()) << " ";
  file.saveFile();
}

// 测试 new_full
TEST_F(TensorTest, NewFull) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();
  at::Tensor full_tensor = tensor.new_full({2, 3}, 7.5);
  file << std::to_string(full_tensor.sizes()[0]) << " ";
  file << std::to_string(full_tensor.sizes()[1]) << " ";
  file << std::to_string(full_tensor.data_ptr<float>()[0]) << " ";
  file.saveFile();
}

// 测试 new_zeros
TEST_F(TensorTest, NewZeros) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();
  at::Tensor zeros_tensor = tensor.new_zeros({2, 3});
  file << std::to_string(zeros_tensor.sizes()[0]) << " ";
  file << std::to_string(zeros_tensor.sizes()[1]) << " ";
  file << std::to_string(zeros_tensor.data_ptr<float>()[0]) << " ";
  file.saveFile();
}

// 测试 new_ones
TEST_F(TensorTest, NewOnes) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();
  at::Tensor ones_tensor = tensor.new_ones({2, 3});
  file << std::to_string(ones_tensor.sizes()[0]) << " ";
  file << std::to_string(ones_tensor.sizes()[1]) << " ";
  file << std::to_string(ones_tensor.data_ptr<float>()[0]) << " ";
  file.saveFile();
}

// 测试 resize_ - Paddle不支持，会抛出异常
TEST_F(TensorTest, Resize) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();
  try {
    tensor.resize_({4, 5});
    file << "0 ";
  } catch (const std::exception& e) {
    file << "1 ";
  }
  file.saveFile();
}

// 测试 expand
TEST_F(TensorTest, Expand) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();
  at::Tensor small = at::ones({1, 3}, at::kFloat);
  at::Tensor expanded = small.expand({4, 3});
  file << std::to_string(expanded.sizes()[0]) << " ";
  file << std::to_string(expanded.sizes()[1]) << " ";
  file.saveFile();
}

// 测试 expand_as - 只测试维度等于1的情况（libtorch和paddle都支持）
TEST_F(TensorTest, ExpandAs) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();
  // 只有维度等于1时才能扩展，这是libtorch的语义
  at::Tensor small = at::ones({1, 3}, at::kFloat);
  at::Tensor target = at::ones({4, 3}, at::kFloat);
  at::Tensor expanded = small.expand_as(target);
  file << std::to_string(expanded.sizes()[0]) << " ";
  file << std::to_string(expanded.sizes()[1]) << " ";
  file.saveFile();
}

}  // namespace test
}  // namespace at
