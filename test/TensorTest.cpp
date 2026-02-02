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

// 测试 var(bool unbiased)
TEST_F(TensorTest, VarUnbiased) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();
  std::vector<int64_t> shape = {2, 3};
  at::Tensor test_tensor = at::ones(shape, at::kFloat);
  test_tensor.data_ptr<float>()[0] = 1.0f;
  test_tensor.data_ptr<float>()[1] = 2.0f;
  test_tensor.data_ptr<float>()[2] = 3.0f;
  test_tensor.data_ptr<float>()[3] = 4.0f;
  test_tensor.data_ptr<float>()[4] = 5.0f;
  test_tensor.data_ptr<float>()[5] = 6.0f;
  at::Tensor var_result = test_tensor.var(true);
  file << std::to_string(var_result.dim()) << " ";
  file << std::to_string(var_result.data_ptr<float>()[0]) << " ";
  at::Tensor var_result_biased = test_tensor.var(false);
  file << std::to_string(var_result_biased.dim()) << " ";
  file << std::to_string(var_result_biased.data_ptr<float>()[0]) << " ";
  file.saveFile();
}

// 测试 var(OptionalIntArrayRef dim, bool unbiased, bool keepdim)
TEST_F(TensorTest, VarDim) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();
  std::vector<int64_t> shape = {2, 3};
  at::Tensor test_tensor = at::ones(shape, at::kFloat);
  for (int i = 0; i < 6; ++i) {
    test_tensor.data_ptr<float>()[i] = static_cast<float>(i + 1);
  }
  at::Tensor var_result = test_tensor.var({0}, true, false);
  file << std::to_string(var_result.dim()) << " ";
  file << std::to_string(var_result.size(0)) << " ";
  at::Tensor var_result_keepdim = test_tensor.var({1}, true, true);
  file << std::to_string(var_result_keepdim.dim()) << " ";
  file << std::to_string(var_result_keepdim.size(0)) << " ";
  file << std::to_string(var_result_keepdim.size(1)) << " ";
  file.saveFile();
}

// 测试 var(OptionalIntArrayRef dim, optional<Scalar> correction, bool keepdim)
TEST_F(TensorTest, VarCorrection) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();
  std::vector<int64_t> shape = {2, 3};
  at::Tensor test_tensor = at::ones(shape, at::kFloat);
  for (int i = 0; i < 6; ++i) {
    test_tensor.data_ptr<float>()[i] = static_cast<float>(i + 1);
  }
  at::Tensor var_result = test_tensor.var({0}, at::Scalar(1.0), false);
  file << std::to_string(var_result.dim()) << " ";
  file << std::to_string(var_result.size(0)) << " ";
  at::Tensor var_result_pop = test_tensor.var({0}, at::Scalar(0.0), false);
  file << std::to_string(var_result_pop.dim()) << " ";
  file << std::to_string(var_result_pop.size(0)) << " ";
  file.saveFile();
}

}  // namespace test
}  // namespace at
