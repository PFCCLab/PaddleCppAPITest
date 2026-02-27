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

// 测试 all() - 检查所有元素是否为真（非零）
TEST_F(TensorTest, All) {
  FileManerger file(GetTestCaseResultFileName());
  file.createFile();

  // 测试全1张量 - all() 应返回 true
  at::Tensor all_ones = at::ones({2, 2}, at::kInt);
  bool result1 = all_ones.all().item<bool>();
  file << std::to_string(result1) << " ";

  // 测试全0张量 - all() 应返回 false
  at::Tensor all_zeros = at::zeros({2, 2}, at::kInt);
  bool result2 = all_zeros.all().item<bool>();
  file << std::to_string(result2) << " ";

  // 测试混合张量（有0有1）- all() 应返回 false
  std::vector<int> data3 = {1, 0, 1, 1};
  at::Tensor mixed = at::from_blob(data3.data(), {2, 2}, at::kInt).clone();
  bool result3 = mixed.all().item<bool>();
  file << std::to_string(result3) << " ";

  // 测试全为负数张量 - all() 应返回 true（非零）
  std::vector<int> data4 = {-1, -2, -3, -4};
  at::Tensor all_neg = at::from_blob(data4.data(), {2, 2}, at::kInt).clone();
  bool result4 = all_neg.all().item<bool>();
  file << std::to_string(result4) << " ";

  file.saveFile();
}

// 测试 all(dim, keepdim) - 沿指定维度检查
TEST_F(TensorTest, AllDim) {
  FileManerger file(GetTestCaseResultFileName());
  file.createFile();

  std::vector<int> data = {1, 0, 1, 1, 1, 1};
  at::Tensor tensor = at::from_blob(data.data(), {2, 3}, at::kInt).clone();

  // 沿 dim=0 检查 - 每列所有行
  at::Tensor result_dim0 = tensor.all(0, false);
  file << std::to_string(result_dim0.sizes()[0]) << " ";
  file << std::to_string(result_dim0.sizes()[1]) << " ";
  // 第一列有0，应为false；第二列全为1，应为true；第三列有0，应为false
  file << std::to_string(result_dim0[0].item<bool>()) << " ";
  file << std::to_string(result_dim0[1].item<bool>()) << " ";
  file << std::to_string(result_dim0[2].item<bool>()) << " ";

  // 沿 dim=1 检查 - 每行所有列
  at::Tensor result_dim1 = tensor.all(1, false);
  file << std::to_string(result_dim1.sizes()[0]) << " ";
  // 第一行有0，应为false；第二行全为1，应为true
  file << std::to_string(result_dim1[0].item<bool>()) << " ";
  file << std::to_string(result_dim1[1].item<bool>()) << " ";

  // 测试 keepdim=true
  at::Tensor result_keepdim = tensor.all(1, true);
  file << std::to_string(result_keepdim.sizes()[0]) << " ";
  file << std::to_string(result_keepdim.sizes()[1]) << " ";

  file.saveFile();
}

// 测试 all(at::OptionalIntArrayRef dim, bool keepdim)
TEST_F(TensorTest, AllOptionalDim) {
  FileManerger file(GetTestCaseResultFileName());
  file.createFile();

  std::vector<int> data = {1, 0, 1, 1, 1, 1};
  at::Tensor tensor = at::from_blob(data.data(), {2, 3}, at::kInt).clone();

  // 不指定维度 - 检查所有元素
  at::Tensor result_no_dim = tensor.all(c10::nullopt, false);
  file << std::to_string(result_no_dim.item<bool>()) << " ";

  // 指定单个维度
  at::Tensor result_single_dim = tensor.all({0}, false);
  file << std::to_string(result_single_dim[0].item<bool>()) << " ";
  file << std::to_string(result_single_dim[1].item<bool>()) << " ";
  file << std::to_string(result_single_dim[2].item<bool>()) << " ";

  // 指定多个维度
  at::Tensor result_multi_dim = tensor.all({0, 1}, false);
  file << std::to_string(result_multi_dim.item<bool>()) << " ";

  file.saveFile();
}

// 测试 allclose - 检查两个张量是否接近
TEST_F(TensorTest, Allclose) {
  FileManerger file(GetTestCaseResultFileName());
  file.createFile();

  // 测试1: 完全相同的张量 - 应返回 true
  std::vector<float> data1 = {1.0f, 2.0f, 3.0f};
  at::Tensor t1 = at::from_blob(data1.data(), {3}, at::kFloat).clone();
  at::Tensor t1_copy = at::from_blob(data1.data(), {3}, at::kFloat).clone();
  bool result1 = t1.allclose(t1_copy);
  file << std::to_string(result1) << " ";

  // 测试2: 在默认 rtol/atol 范围内的张量 - 应返回 true
  std::vector<float> data2 = {1.0f, 2.0f, 3.0f};
  std::vector<float> data2_slight = {1.0f + 1e-6f, 2.0f - 1e-6f, 3.0f};
  at::Tensor t2 = at::from_blob(data2.data(), {3}, at::kFloat).clone();
  at::Tensor t2_slight =
      at::from_blob(data2_slight.data(), {3}, at::kFloat).clone();
  bool result2 = t2.allclose(t2_slight);
  file << std::to_string(result2) << " ";

  // 测试3: 超出默认容差的张量 - 应返回 false
  std::vector<float> data3 = {1.0f, 2.0f, 3.0f};
  std::vector<float> data3_diff = {1.5f, 2.0f, 3.0f};  // 差异 0.5 > 默认 atol
  at::Tensor t3 = at::from_blob(data3.data(), {3}, at::kFloat).clone();
  at::Tensor t3_diff =
      at::from_blob(data3_diff.data(), {3}, at::kFloat).clone();
  bool result3 = t3.allclose(t3_diff);
  file << std::to_string(result3) << " ";

  // 测试4: 使用较大 rtol 的张量 - 应返回 true
  bool result4 = t3.allclose(t3_diff, 0.5, 0.1, false);
  file << std::to_string(result4) << " ";

  // 测试7: 多维张量
  std::vector<float> data7 = {1.0f, 2.0f, 3.0f, 4.0f};
  at::Tensor t7 = at::from_blob(data7.data(), {2, 2}, at::kFloat).clone();
  at::Tensor t7_copy = at::from_blob(data7.data(), {2, 2}, at::kFloat).clone();
  bool result7 = t7.allclose(t7_copy);
  file << std::to_string(result7) << " ";

  file.saveFile();
}

}  // namespace test
}  // namespace at
