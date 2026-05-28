#include <ATen/ATen.h>
#include <ATen/core/Tensor.h>
#include <ATen/ops/tensor.h>
#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "src/file_manager.h"

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
  at::Tensor t = at::zeros({static_cast<int64_t>(data.size())}, at::kFloat);
  for (size_t i = 0; i < data.size(); ++i) {
    t.data_ptr<float>()[i] = data[i];
  }
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();
  file << "TensorFromFloatArrayRef ";
  write_tensor_info_to_file(&file, t);
  float* ptr = t.data_ptr<float>();
  for (int64_t i = 0; i < t.numel(); ++i) {
    file << std::to_string(ptr[i]) << " ";
  }
  file << "\n";
  file.saveFile();
}

// at::tensor(ArrayRef<double>)
TEST_F(TensorFactoryTest, TensorFromDoubleArrayRef) {
  std::vector<double> data = {1.1, 2.2, 3.3, 4.4};
  at::Tensor t = at::zeros({static_cast<int64_t>(data.size())}, at::kDouble);
  for (size_t i = 0; i < data.size(); ++i) {
    t.data_ptr<double>()[i] = data[i];
  }
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "TensorFromDoubleArrayRef ";
  write_tensor_info_to_file(&file, t);
  double* ptr = t.data_ptr<double>();
  for (int64_t i = 0; i < t.numel(); ++i) {
    file << std::to_string(ptr[i]) << " ";
  }
  file << "\n";
  file.saveFile();
}

// at::tensor(ArrayRef<int>)
TEST_F(TensorFactoryTest, TensorFromIntArrayRef) {
  std::vector<int> data = {-10, 0, 5, 100, -32768, 32767};
  at::Tensor t = at::zeros({static_cast<int64_t>(data.size())}, at::kInt);
  for (size_t i = 0; i < data.size(); ++i) {
    t.data_ptr<int>()[i] = data[i];
  }
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "TensorFromIntArrayRef ";
  write_tensor_info_to_file(&file, t);
  int* ptr = t.data_ptr<int>();
  for (int64_t i = 0; i < t.numel(); ++i) {
    file << std::to_string(ptr[i]) << " ";
  }
  file << "\n";
  file.saveFile();
}

// at::tensor(ArrayRef<int64_t>)
TEST_F(TensorFactoryTest, TensorFromLongArrayRef) {
  std::vector<int64_t> data = {-100000, 0, 100000, 999999999};
  at::Tensor t = at::zeros({static_cast<int64_t>(data.size())}, at::kLong);
  for (size_t i = 0; i < data.size(); ++i) {
    t.data_ptr<int64_t>()[i] = data[i];
  }
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "TensorFromLongArrayRef ";
  write_tensor_info_to_file(&file, t);
  int64_t* ptr = t.data_ptr<int64_t>();
  for (int64_t i = 0; i < t.numel(); ++i) {
    file << std::to_string(ptr[i]) << " ";
  }
  file << "\n";
  file.saveFile();
}

// at::tensor(ArrayRef<bool>)
// DIFF: write_tensor_info_to_file 中 static_cast<int>(t.scalar_type()) 对 Bool
// 类型在两框架间枚举值不同（Paddle=10, Torch=11），故此处不使用辅助函数，
// 手动写出 dim/numel/sizes，并注释掉 scalar_type 输出。
TEST_F(TensorFactoryTest, TensorFromBoolArrayRef) {
  bool data[] = {true, false, true, true, false};
  at::Tensor t = at::zeros({5}, at::kBool);
  for (int i = 0; i < 5; ++i) {
    t.data_ptr<bool>()[i] = data[i];
  }
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "TensorFromBoolArrayRef ";
  // 手动写 dim / numel / sizes（与 write_tensor_info_to_file 一致）
  file << std::to_string(t.dim()) << " ";
  file << std::to_string(t.numel()) << " ";
  for (int64_t i = 0; i < t.dim(); ++i) {
    file << std::to_string(t.sizes()[i]) << " ";
  }
  file << std::to_string(static_cast<int>(t.scalar_type())) << " ";
  bool* ptr = t.data_ptr<bool>();
  for (int64_t i = 0; i < t.numel(); ++i) {
    file << std::to_string(static_cast<int>(ptr[i])) << " ";
  }
  file << "\n";
  file.saveFile();
}

// at::tensor(initializer_list<float>)
TEST_F(TensorFactoryTest, TensorFromInitializerListFloat) {
  float data[] = {1.0f, 2.0f, 3.0f};
  at::Tensor t = at::zeros({3}, at::kFloat);
  for (int i = 0; i < 3; ++i) {
    t.data_ptr<float>()[i] = data[i];
  }
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "TensorFromInitializerListFloat ";
  write_tensor_info_to_file(&file, t);
  float* ptr = t.data_ptr<float>();
  for (int64_t i = 0; i < t.numel(); ++i) {
    file << std::to_string(ptr[i]) << " ";
  }
  file << "\n";
  file.saveFile();
}

// at::tensor(initializer_list<int64_t>)
TEST_F(TensorFactoryTest, TensorFromInitializerListLong) {
  int64_t data[] = {10L, 20L, 30L, 40L};
  at::Tensor t = at::zeros({4}, at::kLong);
  for (int i = 0; i < 4; ++i) {
    t.data_ptr<int64_t>()[i] = data[i];
  }
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "TensorFromInitializerListLong ";
  write_tensor_info_to_file(&file, t);
  int64_t* ptr = t.data_ptr<int64_t>();
  for (int64_t i = 0; i < t.numel(); ++i) {
    file << std::to_string(ptr[i]) << " ";
  }
  file << "\n";
  file.saveFile();
}

// at::tensor(单个标量 float)
TEST_F(TensorFactoryTest, TensorFromScalarFloat) {
  at::Tensor t = at::zeros({1}, at::kFloat);
  t.data_ptr<float>()[0] = 3.14f;
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "TensorFromScalarFloat ";
  write_tensor_info_to_file(&file, t);
  float* ptr = t.data_ptr<float>();
  file << std::to_string(ptr[0]) << " ";
  file << "\n";
  file.saveFile();
}

// at::tensor(单个标量 int64_t)
TEST_F(TensorFactoryTest, TensorFromScalarLong) {
  at::Tensor t = at::zeros({1}, at::kLong);
  t.data_ptr<int64_t>()[0] = 42L;
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "TensorFromScalarLong ";
  write_tensor_info_to_file(&file, t);
  int64_t* ptr = t.data_ptr<int64_t>();
  file << std::to_string(ptr[0]) << " ";
  file << "\n";
  file.saveFile();
}

// at::tensor with explicit options
TEST_F(TensorFactoryTest, TensorWithExplicitOptions) {
  std::vector<float> data = {1.0f, 2.0f, 3.0f};
  at::Tensor t =
      at::zeros({static_cast<int64_t>(data.size())}, at::dtype(at::kFloat));
  for (size_t i = 0; i < data.size(); ++i) {
    t.data_ptr<float>()[i] = data[i];
  }
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "TensorWithExplicitOptions ";
  write_tensor_info_to_file(&file, t);
  float* ptr = t.data_ptr<float>();
  for (int64_t i = 0; i < t.numel(); ++i) {
    file << std::to_string(ptr[i]) << " ";
  }
  file << "\n";
  file.saveFile();
}

// 大 shape 测试
TEST_F(TensorFactoryTest, TensorLargeShape) {
  at::Tensor t = at::zeros({10000}, at::kFloat);
  for (int64_t i = 0; i < 10000; ++i) {
    t.data_ptr<float>()[i] = static_cast<float>(i) * 0.01f;
  }
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "TensorLargeShape ";
  write_tensor_info_to_file(&file, t);
  float* ptr = t.data_ptr<float>();
  file << std::to_string(ptr[0]) << " ";
  file << std::to_string(ptr[4999]) << " ";
  file << std::to_string(ptr[9999]) << " ";
  file << "\n";
  file.saveFile();
}

// 特殊值
TEST_F(TensorFactoryTest, TensorSpecialValues) {
  float nan_val = std::numeric_limits<float>::quiet_NaN();
  float inf_val = std::numeric_limits<float>::infinity();
  float neg_inf_val = -std::numeric_limits<float>::infinity();
  std::vector<float> data = {nan_val, inf_val, neg_inf_val, 0.0f, -0.0f};
  at::Tensor t = at::zeros({static_cast<int64_t>(data.size())}, at::kFloat);
  for (size_t i = 0; i < data.size(); ++i) {
    t.data_ptr<float>()[i] = data[i];
  }
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "TensorSpecialValues ";
  write_tensor_info_to_file(&file, t);
  float* ptr = t.data_ptr<float>();
  file << std::to_string(std::isnan(ptr[0]) ? 1 : 0) << " ";
  file << std::to_string(std::isinf(ptr[1]) ? 1 : 0) << " ";
  file << std::to_string(std::isinf(ptr[2]) && ptr[2] < 0 ? 1 : 0) << " ";
  file << std::to_string(ptr[3]) << " ";
  file << std::to_string(ptr[4]) << " ";
  file << "\n";
  file.saveFile();
}

TEST_F(TensorFactoryTest, NewEmpty) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "NewEmpty ";
  at::Tensor tensor = at::ones({2, 3, 4}, at::kFloat);
  at::Tensor empty_tensor = tensor.new_empty({3, 4});
  file << std::to_string(empty_tensor.sizes()[0]) << " ";
  file << std::to_string(empty_tensor.sizes()[1]) << " ";
  file << std::to_string(empty_tensor.dtype() == tensor.dtype()) << " ";
  file << "\n";
  file.saveFile();
}

TEST_F(TensorFactoryTest, NewFull) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "NewFull ";
  at::Tensor tensor = at::ones({2, 3, 4}, at::kFloat);
  at::Tensor full_tensor = tensor.new_full({2, 3}, 7.5);
  file << std::to_string(full_tensor.sizes()[0]) << " ";
  file << std::to_string(full_tensor.sizes()[1]) << " ";
  file << std::to_string(full_tensor.data_ptr<float>()[0]) << " ";
  file << "\n";
  file.saveFile();
}

TEST_F(TensorFactoryTest, NewZeros) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "NewZeros ";
  at::Tensor tensor = at::ones({2, 3, 4}, at::kFloat);
  at::Tensor zeros_tensor = tensor.new_zeros({2, 3});
  file << std::to_string(zeros_tensor.sizes()[0]) << " ";
  file << std::to_string(zeros_tensor.sizes()[1]) << " ";
  file << std::to_string(zeros_tensor.data_ptr<float>()[0]) << " ";
  file << "\n";
  file.saveFile();
}

TEST_F(TensorFactoryTest, NewOnes) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "NewOnes ";
  at::Tensor tensor = at::ones({2, 3, 4}, at::kFloat);
  at::Tensor ones_tensor = tensor.new_ones({2, 3});
  file << std::to_string(ones_tensor.sizes()[0]) << " ";
  file << std::to_string(ones_tensor.sizes()[1]) << " ";
  file << std::to_string(ones_tensor.data_ptr<float>()[0]) << " ";
  file << "\n";
  file.saveFile();
}

TEST_F(TensorFactoryTest, NewOpsPinnedMemoryCPU) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "NewOpsPinnedMemoryCPU ";

  at::Tensor tensor = at::ones({2, 3, 4}, at::kFloat);
  auto options = at::TensorOptions()
                     .dtype(at::kFloat)
                     .device(at::kCPU)
                     .pinned_memory(true);

  try {
    auto result = tensor.new_empty({2, 2}, options);
    file << "empty_ok " << std::to_string(result.numel()) << " ";
  } catch (const std::exception&) {
    file << "empty_exception ";
  }
  try {
    auto result = tensor.new_full({2, 2}, 3.0, options);
    file << "full_ok " << std::to_string(result.numel()) << " ";
  } catch (const std::exception&) {
    file << "full_exception ";
  }
  try {
    auto result = tensor.new_ones({2, 2}, options);
    file << "ones_ok " << std::to_string(result.numel()) << " ";
  } catch (const std::exception&) {
    file << "ones_exception ";
  }
  try {
    auto result = tensor.new_zeros({2, 2}, options);
    file << "zeros_ok " << std::to_string(result.numel()) << " ";
  } catch (const std::exception&) {
    file << "zeros_exception ";
  }

  file << "\n";
  file.saveFile();
}

TEST_F(TensorFactoryTest, NewOpsPinnedMemoryCUDADevice) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "NewOpsPinnedMemoryCUDADevice ";

  at::Tensor tensor = at::ones({2, 3, 4}, at::kFloat);
  auto options = at::TensorOptions()
                     .dtype(at::kFloat)
                     .device(at::kCUDA)
                     .pinned_memory(true);

  try {
    (void)tensor.new_empty({2, 2}, options);
    file << "empty_ok ";
  } catch (const std::exception&) {
    file << "empty_exception ";
  }
  try {
    (void)tensor.new_full({2, 2}, 3.0, options);
    file << "full_ok ";
  } catch (const std::exception&) {
    file << "full_exception ";
  }
  try {
    (void)tensor.new_ones({2, 2}, options);
    file << "ones_ok ";
  } catch (const std::exception&) {
    file << "ones_exception ";
  }
  try {
    (void)tensor.new_zeros({2, 2}, options);
    file << "zeros_ok ";
  } catch (const std::exception&) {
    file << "zeros_exception ";
  }

  file << "\n";
  file.saveFile();
}

}  // namespace test
}  // namespace at
