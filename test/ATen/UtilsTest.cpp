#include <ATen/ATen.h>
#include <ATen/Utils.h>
#include <ATen/ops/tensor.h>
#include <gtest/gtest.h>

#include <complex>
#include <string>
#include <vector>

#include "src/file_manager.h"

extern paddle_api_test::ThreadSafeParam g_custom_param;

namespace at {
namespace test {

using paddle_api_test::FileManerger;
using paddle_api_test::ThreadSafeParam;

static void WriteCudaRuntimeUnavailable(FileManerger* file) {
  *file << "cuda_runtime_unavailable ";
}

static void WriteTensorResult(FileManerger* file, const at::Tensor& result) {
  at::Tensor cpu_result = result.cpu();
  if (cpu_result.scalar_type() != at::kFloat) {
    cpu_result = cpu_result.to(at::kFloat);
  }
  cpu_result = cpu_result.contiguous();

  *file << result.dim() << " ";
  *file << result.numel() << " ";
  const auto* data = cpu_result.data_ptr<float>();
  for (int64_t i = 0; i < cpu_result.numel(); ++i) {
    *file << data[i] << " ";
  }
}

static void WriteComplexTensorResult(FileManerger* file,
                                     const at::Tensor& result) {
  at::Tensor cpu_result = result.cpu().contiguous();
  *file << result.dim() << " ";
  *file << result.numel() << " ";
  const auto* data = cpu_result.data_ptr<c10::complex<double>>();
  for (int64_t i = 0; i < cpu_result.numel(); ++i) {
    const auto value = static_cast<std::complex<double>>(data[i]);
    *file << value.real() << " ";
    *file << value.imag() << " ";
  }
}

TEST(UtilsTest, TensorCPU) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();
  file << "TensorCPU ";

  std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
  at::ArrayRef<float> arr(data);
  at::Tensor t1 = at::tensor(arr, at::TensorOptions(at::kFloat));
  WriteTensorResult(&file, t1);

  std::vector<float> empty_data;
  at::ArrayRef<float> empty_arr(empty_data);
  at::Tensor t2 = at::tensor(empty_arr, at::TensorOptions(at::kFloat));
  WriteTensorResult(&file, t2);

  file << "\n";
  file.saveFile();
}

TEST(UtilsTest, TensorBackend) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "TensorBackend ";

  std::vector<float> data = {5.0f};
  at::ArrayRef<float> arr(data);
  try {
    at::Tensor t1 = at::tensor(
        arr, at::TensorOptions(at::kFloat).device(c10::DeviceType::CUDA));
    WriteTensorResult(&file, t1);
  } catch (const std::exception&) {
    WriteCudaRuntimeUnavailable(&file);
  }

  file << "\n";
  file.saveFile();
}

TEST(UtilsTest, TensorComplexCPU) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "TensorComplexCPU ";

  std::vector<c10::complex<double>> data = {{1.0, 2.0}, {3.0, 4.0}};
  at::ArrayRef<c10::complex<double>> arr(data);
  at::Tensor t1 = at::tensor(arr, at::TensorOptions(at::kComplexDouble));
  WriteComplexTensorResult(&file, t1);

  file << "\n";
  file.saveFile();
}

TEST(UtilsTest, TensorComplexBackend) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "TensorComplexBackend ";

  std::vector<c10::complex<double>> data = {{5.0, 6.0}};
  at::ArrayRef<c10::complex<double>> arr(data);
  try {
    at::Tensor t1 = at::tensor(
        arr,
        at::TensorOptions(at::kComplexDouble).device(c10::DeviceType::CUDA));
    WriteComplexTensorResult(&file, t1);
  } catch (const std::exception&) {
    WriteCudaRuntimeUnavailable(&file);
  }

  file << "\n";
  file.saveFile();
}

}  // namespace test
}  // namespace at
