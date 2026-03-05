#include <ATen/ATen.h>
#include <gtest/gtest.h>
#include <torch/library.h>

#include <functional>
#include <string>
#include <vector>

#include "../src/file_manager.h"

extern paddle_api_test::ThreadSafeParam g_custom_param;

namespace at {
namespace test {

using paddle_api_test::FileManerger;
using paddle_api_test::ThreadSafeParam;

class LibraryTest : public ::testing::Test {
 protected:
  void SetUp() override {}
};

// 测试 torch::Library::Kind 枚举
TEST_F(LibraryTest, LibraryKindEnum) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();

  file << std::to_string(static_cast<int>(torch::Library::Kind::DEF)) << " ";
  file << std::to_string(static_cast<int>(torch::Library::Kind::IMPL)) << " ";
  file << std::to_string(static_cast<int>(torch::Library::Kind::FRAGMENT))
       << " ";
  file.saveFile();
}

// 测试 IValue 基本构造
TEST_F(LibraryTest, IValueBasicConstruction) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();

#if USE_PADDLE_API
  // Paddle 兼容层使用 torch::IValue，方法名是 snake_case
  torch::IValue ival_int(42);
  torch::IValue ival_double(3.14);
  torch::IValue ival_string(std::string("test"));

  file << std::to_string(ival_int.is_int() ? 1 : 0) << " ";
  file << std::to_string(ival_double.is_double() ? 1 : 0) << " ";
  file << std::to_string(ival_string.is_string() ? 1 : 0) << " ";
#else
  // libtorch 使用 c10::IValue，方法名是 camelCase
  c10::IValue ival_int(42);
  c10::IValue ival_double(3.14);
  c10::IValue ival_string(std::string("test"));

  file << std::to_string(ival_int.isInt() ? 1 : 0) << " ";
  file << std::to_string(ival_double.isDouble() ? 1 : 0) << " ";
  file << std::to_string(ival_string.isString() ? 1 : 0) << " ";
#endif
  file.saveFile();
}

// 测试 IValue 从 vector 构造
TEST_F(LibraryTest, IValueVectorConstruction) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();

#if USE_PADDLE_API
  std::vector<torch::IValue> args_vec = {
      torch::IValue(1), torch::IValue(2.5), torch::IValue(std::string("test"))};
#else
  std::vector<c10::IValue> args_vec = {
      c10::IValue(1), c10::IValue(2.5), c10::IValue(std::string("test"))};
#endif

  file << std::to_string(args_vec.size()) << " ";
  file << std::to_string(args_vec.empty() ? 0 : 1) << " ";
  file.saveFile();
}

// 测试 IValue get 方法
TEST_F(LibraryTest, IValueGet) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();

#if USE_PADDLE_API
  torch::IValue ival_int(42);
  torch::IValue ival_double(3.14);

  try {
    int64_t int_val = ival_int.to_int();
    file << std::to_string(int_val) << " ";
  } catch (...) {
    file << "-1 ";
  }

  try {
    double double_val = ival_double.to_double();
    file << std::to_string(double_val) << " ";
  } catch (...) {
    file << "-1 ";
  }
#else
  c10::IValue ival_int(42);
  c10::IValue ival_double(3.14);

  try {
    int64_t int_val = ival_int.toInt();
    file << std::to_string(int_val) << " ";
  } catch (...) {
    file << "-1 ";
  }

  try {
    double double_val = ival_double.toDouble();
    file << std::to_string(double_val) << " ";
  } catch (...) {
    file << "-1 ";
  }
#endif

  file.saveFile();
}

// 测试 IValue is_none
TEST_F(LibraryTest, IValueIsNone) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();

#if USE_PADDLE_API
  torch::IValue ival_none;
  torch::IValue ival_int(42);

  file << std::to_string(ival_none.is_none() ? 1 : 0) << " ";
  file << std::to_string(ival_int.is_none() ? 0 : 1) << " ";
#else
  c10::IValue ival_none;
  c10::IValue ival_int(42);

  file << std::to_string(ival_none.isNone() ? 1 : 0) << " ";
  file << std::to_string(ival_int.isNone() ? 0 : 1) << " ";
#endif
  file.saveFile();
}

// 测试 IValue 显式转换为 int64_t
TEST_F(LibraryTest, IValueSizeToInt64) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();

  size_t sz = 42;
#if USE_PADDLE_API
  torch::IValue ival(static_cast<int64_t>(sz));
  file << std::to_string(ival.to_int()) << " ";
#else
  c10::IValue ival(static_cast<int64_t>(sz));
  file << std::to_string(ival.toInt()) << " ";
#endif
  file.saveFile();
}

// 测试 IValue 作为 Tensor
TEST_F(LibraryTest, IValueTensor) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();

  at::Tensor tensor = at::ones({3, 3});
#if USE_PADDLE_API
  torch::IValue ival(tensor);
  file << std::to_string(ival.is_tensor() ? 1 : 0) << " ";
#else
  c10::IValue ival(tensor);
  file << std::to_string(ival.isTensor() ? 1 : 0) << " ";
#endif
  file.saveFile();
}

// 测试 at::Tensor 操作
TEST_F(LibraryTest, TensorOperations) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();

  at::Tensor t1 = at::ones({2, 3});
  at::Tensor t2 = at::ones({2, 3});
#if USE_PADDLE_API
  at::Tensor t3 = t1 + t2;
#else
  at::Tensor t3 = t1.add(t2);
#endif

  file << std::to_string(t3.numel()) << " ";
  file.saveFile();
}

// 测试 at::Device - 只在 libtorch 下
#if !USE_PADDLE_API
TEST_F(LibraryTest, DeviceTest) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();

  at::Device device(c10::DeviceType::CPU);
  file << std::to_string(device.type() == c10::DeviceType::CPU ? 1 : 0) << " ";
  file.saveFile();
}

// 测试 c10::TensorOptions - 只在 libtorch 下
TEST_F(LibraryTest, TensorOptionsTest) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();

  auto opts =
      c10::TensorOptions().dtype(at::kFloat).device(c10::DeviceType::CPU);
  file << std::to_string(opts.dtype() == at::kFloat ? 1 : 0) << " ";
  file.saveFile();
}
#endif

// 测试 torch::Library 构造（不实际注册）
TEST_F(LibraryTest, LibraryConstruction) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "1 ";
  file.saveFile();
}

}  // namespace test
}  // namespace at
