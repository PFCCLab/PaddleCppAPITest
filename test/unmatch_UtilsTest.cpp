#include <ATen/ATen.h>
#include <ATen/Utils.h>
#include <gtest/gtest.h>

#include <iostream>

#include "../src/file_manager.h"

extern paddle_api_test::ThreadSafeParam g_custom_param;

namespace at {
namespace test {

// [DIFF] 文件级说明：本文件多个用例直接调用 at::detail::* 工具函数，
// 在 Paddle 侧存在未导出/未实例化符号，去掉 unmatch 前缀后会在链接阶段失败。

using paddle_api_test::FileManerger;
using paddle_api_test::ThreadSafeParam;

static void write_op_result_to_file(FileManerger* file,
                                    const at::Tensor& result) {
  *file << std::to_string(result.dim()) << " ";
  *file << std::to_string(result.numel()) << " ";
  float* data = result.data_ptr<float>();
  for (int64_t i = 0; i < result.numel(); ++i) {
    *file << std::to_string(data[i]) << " ";
  }
}

TEST(UtilsTest, TensorCPU) {
  // [DIFF] 用例级差异：TensorCPU 在 Paddle 常规链接中会报
  // undefined reference: at::detail::tensor_cpu<...>。
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();
  file << "TensorCPU ";

  // 1. float array
  std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
  at::ArrayRef<float> arr(data);
  // [DIFF] 问题行：该调用在 Paddle 侧缺少稳定链接符号（模板实例/导出缺失）。
  at::Tensor t1 = at::detail::tensor_cpu(arr, at::TensorOptions(at::kFloat));
  write_op_result_to_file(&file, t1);

  // 2. empty array
  std::vector<float> empty_data;
  at::ArrayRef<float> empty_arr(empty_data);
  // [DIFF] 问题行：同上，empty_arr 分支同样命中未定义符号。
  at::Tensor t2 =
      at::detail::tensor_cpu(empty_arr, at::TensorOptions(at::kFloat));
  write_op_result_to_file(&file, t2);

  file << "\n";
  file.saveFile();
}

TEST(UtilsTest, TensorBackend) {
  // [DIFF] 用例级差异：TensorBackend 在 Paddle 侧会报
  // undefined reference: at::detail::tensor_backend<...>。
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "TensorBackend ";

  // 1. float array
  std::vector<float> data = {5.0f};
  at::ArrayRef<float> arr(data);
  // [DIFF] 问题行：detail 层符号未稳定导出，常规构建链接失败。
  at::Tensor t1 =
      at::detail::tensor_backend(arr, at::TensorOptions(at::kFloat));
  write_op_result_to_file(&file, t1);

  file << "\n";
  file.saveFile();
}

TEST(UtilsTest, TensorComplexCPU) {
  // [DIFF] 用例级差异：复数路径在 Paddle 侧会报
  // undefined reference: at::detail::tensor_complex_cpu<...>。
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "TensorComplexCPU ";

#if USE_PADDLE_API
  std::vector<c10::complex<double>> data = {{1.0, 2.0}, {3.0, 4.0}};
  at::ArrayRef<c10::complex<double>> arr(data);
  // [DIFF] 问题行：复数模板实例在链接阶段不可见。
  at::Tensor t1 = at::detail::tensor_complex_cpu(
      arr, at::TensorOptions(at::kComplexDouble));

  file << std::to_string(t1.dim()) << " ";
  file << std::to_string(t1.numel()) << " ";
  auto* t1_data = t1.data_ptr<c10::complex<double>>();
  for (int64_t i = 0; i < t1.numel(); ++i) {
    file << std::to_string(t1_data[i].real) << " "
         << std::to_string(t1_data[i].imag) << " ";
  }
#else
  file << "1 2 1.000000 2.000000 3.000000 4.000000 ";
#endif

  file << "\n";
  file.saveFile();
}

TEST(UtilsTest, TensorComplexBackend) {
  // [DIFF] 用例级差异：TensorComplexBackend 在 Paddle 侧会报
  // undefined reference: at::detail::tensor_complex_backend<...>。
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "TensorComplexBackend ";

#if USE_PADDLE_API
  std::vector<c10::complex<double>> data = {{5.0, 6.0}};
  at::ArrayRef<c10::complex<double>> arr(data);
  // [DIFF] 问题行：复数 backend detail 符号未稳定导出。
  at::Tensor t1 = at::detail::tensor_complex_backend(
      arr, at::TensorOptions(at::kComplexDouble));

  file << std::to_string(t1.dim()) << " ";
  file << std::to_string(t1.numel()) << " ";
  auto* t1_data = t1.data_ptr<c10::complex<double>>();
  for (int64_t i = 0; i < t1.numel(); ++i) {
    file << std::to_string(t1_data[i].real) << " "
         << std::to_string(t1_data[i].imag) << " ";
  }
#else
  file << "1 1 5.000000 6.000000 ";
#endif

  file << "\n";
  file.saveFile();
}

}  // namespace test
}  // namespace at
