#include <ATen/ATen.h>
#include <ATen/core/Tensor.h>
#include <ATen/ops/tensor.h>
#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "../../src/file_manager.h"

extern paddle_api_test::ThreadSafeParam g_custom_param;

namespace at {
namespace test {

using paddle_api_test::FileManerger;
using paddle_api_test::ThreadSafeParam;

class TensorFactoryTest : public ::testing::Test {
 protected:
  void SetUp() override {}
};

static void write_tensor_info_to_file(FileManerger* file, const at::Tensor& t) {
  *file << std::to_string(t.dim()) << " ";
  *file << std::to_string(t.numel()) << " ";
  for (int64_t i = 0; i < t.dim(); ++i) {
    *file << std::to_string(t.sizes()[i]) << " ";
  }
  *file << std::to_string(static_cast<int>(t.scalar_type())) << " ";
}

// at::tensor(ArrayRef<float>, options)
TEST_F(TensorFactoryTest, TensorFromFloatArrayRef) {
  std::vector<float> data = {1.0f, 2.5f, 3.7f, 4.0f, 5.5f};
  at::Tensor t = at::tensor(at::ArrayRef<float>(data), at::kFloat);
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();
  write_tensor_info_to_file(&file, t);
  float* ptr = t.data_ptr<float>();
  for (int64_t i = 0; i < t.numel(); ++i) {
    file << std::to_string(ptr[i]) << " ";
  }
  file.saveFile();
}

// at::tensor(ArrayRef<double>)
TEST_F(TensorFactoryTest, TensorFromDoubleArrayRef) {
  std::vector<double> data = {1.1, 2.2, 3.3, 4.4};
  at::Tensor t = at::tensor(at::ArrayRef<double>(data));
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  write_tensor_info_to_file(&file, t);
  double* ptr = t.data_ptr<double>();
  for (int64_t i = 0; i < t.numel(); ++i) {
    file << std::to_string(ptr[i]) << " ";
  }
  file.saveFile();
}

// at::tensor(ArrayRef<int>)
TEST_F(TensorFactoryTest, TensorFromIntArrayRef) {
  std::vector<int> data = {-10, 0, 5, 100, -32768, 32767};
  at::Tensor t = at::tensor(at::ArrayRef<int>(data));
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  write_tensor_info_to_file(&file, t);
  int* ptr = t.data_ptr<int>();
  for (int64_t i = 0; i < t.numel(); ++i) {
    file << std::to_string(ptr[i]) << " ";
  }
  file.saveFile();
}

// at::tensor(ArrayRef<int64_t>)
TEST_F(TensorFactoryTest, TensorFromLongArrayRef) {
  std::vector<int64_t> data = {-100000, 0, 100000, 999999999};
  at::Tensor t = at::tensor(at::ArrayRef<int64_t>(data));
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  write_tensor_info_to_file(&file, t);
  int64_t* ptr = t.data_ptr<int64_t>();
  for (int64_t i = 0; i < t.numel(); ++i) {
    file << std::to_string(ptr[i]) << " ";
  }
  file.saveFile();
}

// at::tensor(ArrayRef<bool>)
TEST_F(TensorFactoryTest, TensorFromBoolArrayRef) {
  std::vector<bool> data_vec = {true, false, true, true, false};
  // bool vector 的 data() 不可直接用, 用 C 数组
  bool data[] = {true, false, true, true, false};
  at::Tensor t = at::tensor(at::ArrayRef<bool>(data, 5));
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  write_tensor_info_to_file(&file, t);
  bool* ptr = t.data_ptr<bool>();
  for (int64_t i = 0; i < t.numel(); ++i) {
    file << std::to_string(static_cast<int>(ptr[i])) << " ";
  }
  file.saveFile();
}

// at::tensor(initializer_list<float>)
TEST_F(TensorFactoryTest, TensorFromInitializerListFloat) {
  at::Tensor t = at::tensor({1.0f, 2.0f, 3.0f});
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  write_tensor_info_to_file(&file, t);
  float* ptr = t.data_ptr<float>();
  for (int64_t i = 0; i < t.numel(); ++i) {
    file << std::to_string(ptr[i]) << " ";
  }
  file.saveFile();
}

// at::tensor(initializer_list<int64_t>)
TEST_F(TensorFactoryTest, TensorFromInitializerListLong) {
  at::Tensor t = at::tensor({10L, 20L, 30L, 40L});
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  write_tensor_info_to_file(&file, t);
  int64_t* ptr = t.data_ptr<int64_t>();
  for (int64_t i = 0; i < t.numel(); ++i) {
    file << std::to_string(ptr[i]) << " ";
  }
  file.saveFile();
}

// at::tensor(单个标量 float)
TEST_F(TensorFactoryTest, TensorFromScalarFloat) {
  at::Tensor t = at::tensor(3.14f);
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  write_tensor_info_to_file(&file, t);
  float* ptr = t.data_ptr<float>();
  file << std::to_string(ptr[0]) << " ";
  file.saveFile();
}

// at::tensor(单个标量 int64_t)
TEST_F(TensorFactoryTest, TensorFromScalarLong) {
  at::Tensor t = at::tensor(42L);
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  write_tensor_info_to_file(&file, t);
  int64_t* ptr = t.data_ptr<int64_t>();
  file << std::to_string(ptr[0]) << " ";
  file.saveFile();
}

// at::tensor with explicit options
TEST_F(TensorFactoryTest, TensorWithExplicitOptions) {
  std::vector<float> data = {1.0f, 2.0f, 3.0f};
  at::Tensor t = at::tensor(at::ArrayRef<float>(data), at::dtype(at::kFloat));
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  write_tensor_info_to_file(&file, t);
  float* ptr = t.data_ptr<float>();
  for (int64_t i = 0; i < t.numel(); ++i) {
    file << std::to_string(ptr[i]) << " ";
  }
  file.saveFile();
}

// 大 shape 测试
TEST_F(TensorFactoryTest, TensorLargeShape) {
  std::vector<float> data(10000);
  for (int64_t i = 0; i < 10000; ++i) {
    data[i] = static_cast<float>(i) * 0.01f;
  }
  at::Tensor t = at::tensor(at::ArrayRef<float>(data));
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  write_tensor_info_to_file(&file, t);
  float* ptr = t.data_ptr<float>();
  file << std::to_string(ptr[0]) << " ";
  file << std::to_string(ptr[4999]) << " ";
  file << std::to_string(ptr[9999]) << " ";
  file.saveFile();
}

// 特殊值
TEST_F(TensorFactoryTest, TensorSpecialValues) {
  float nan_val = std::numeric_limits<float>::quiet_NaN();
  float inf_val = std::numeric_limits<float>::infinity();
  float neg_inf_val = -std::numeric_limits<float>::infinity();
  std::vector<float> data = {nan_val, inf_val, neg_inf_val, 0.0f, -0.0f};
  at::Tensor t = at::tensor(at::ArrayRef<float>(data));
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  write_tensor_info_to_file(&file, t);
  float* ptr = t.data_ptr<float>();
  file << std::to_string(std::isnan(ptr[0]) ? 1 : 0) << " ";
  file << std::to_string(std::isinf(ptr[1]) ? 1 : 0) << " ";
  file << std::to_string(std::isinf(ptr[2]) && ptr[2] < 0 ? 1 : 0) << " ";
  file << std::to_string(ptr[3]) << " ";
  file << std::to_string(ptr[4]) << " ";
  file.saveFile();
}

}  // namespace test
}  // namespace at
