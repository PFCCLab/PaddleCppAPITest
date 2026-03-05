#include <ATen/ATen.h>
#include <ATen/core/Tensor.h>
#include <ATen/ops/zeros.h>
#include <c10/core/Stream.h>
#include <gtest/gtest.h>
#include <torch/all.h>

// Paddle compat 的 c10/cuda/CUDAStream.h 依赖 PADDLE_WITH_CUDA 宏，
// 不能在普通编译环境中直接包含。libtorch 的版本则依赖 cuda_runtime_api.h。
// 两者均只在 USE_PADDLE_API=0（libtorch build）下包含。
#if !USE_PADDLE_API
#include <c10/cuda/CUDAStream.h>
#endif

#include <string>
#include <vector>

#include "../src/file_manager.h"

extern paddle_api_test::ThreadSafeParam g_custom_param;

namespace at {
namespace test {

using paddle_api_test::FileManerger;
using paddle_api_test::ThreadSafeParam;

class RecordStreamTest : public ::testing::Test {
 protected:
  void SetUp() override { cpu_tensor = at::zeros({2, 3}, at::kFloat); }
  at::Tensor cpu_tensor;
};

// 返回一个指向 device 0 默认 CUDA stream 的 at::Stream
// libtorch: 通过 CUDAStream（有 operator Stream() 隐式转换）
// Paddle compat: CUDAStream 未提供隐式转换，手动以 DEFAULT stream id 0 构造
static at::Stream get_default_cuda_stream() {
#if USE_PADDLE_API
  // Paddle: 直接构造（id=0 = CUDA null/default stream）
  return at::Stream(at::Stream::DEFAULT, c10::Device(c10::DeviceType::CUDA, 0));
#else
  // libtorch: CUDAStream 隐式转换为 at::Stream
  return c10::cuda::getCurrentCUDAStream(0);
#endif
}

// --- 基础功能测试：CUDA tensor + CUDA stream ---

// kFloat, shape {2,3} (small)
TEST_F(RecordStreamTest, CudaFloat2x3) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();
  file << "CudaFloat2x3 ";
  if (!torch::cuda::is_available()) {
    file << "no_cuda";
    file << "\n";
    file.saveFile();
    return;
  }
  try {
    at::Tensor t = cpu_tensor.cuda();
    at::Stream stream = get_default_cuda_stream();
    t.record_stream(stream);
    file << "1";
  } catch (const std::exception& e) {
    file << "exception";
  }
  file << "\n";
  file.saveFile();
}

// kDouble, shape {4} (small, different dtype)
TEST_F(RecordStreamTest, CudaDouble4) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "CudaDouble4 ";
  if (!torch::cuda::is_available()) {
    file << "no_cuda";
    file << "\n";
    file.saveFile();
    return;
  }
  try {
    at::Tensor t = at::zeros({4}, at::kDouble).cuda();
    at::Stream stream = get_default_cuda_stream();
    t.record_stream(stream);
    file << "1";
  } catch (const std::exception& e) {
    file << "exception";
  }
  file << "\n";
  file.saveFile();
}

// kInt, shape {100,100} (large, >= 10000 elements)
TEST_F(RecordStreamTest, CudaInt100x100) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "CudaInt100x100 ";
  if (!torch::cuda::is_available()) {
    file << "no_cuda";
    file << "\n";
    file.saveFile();
    return;
  }
  try {
    at::Tensor t = at::zeros({100, 100}, at::kInt).cuda();
    at::Stream stream = get_default_cuda_stream();
    t.record_stream(stream);
    file << "1";
  } catch (const std::exception& e) {
    file << "exception";
  }
  file << "\n";
  file.saveFile();
}

// kLong, shape {} (0-d scalar tensor)
TEST_F(RecordStreamTest, CudaLongScalar) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "CudaLongScalar ";
  if (!torch::cuda::is_available()) {
    file << "no_cuda";
    file << "\n";
    file.saveFile();
    return;
  }
  try {
    at::Tensor t = at::zeros({}, at::kLong).cuda();
    at::Stream stream = get_default_cuda_stream();
    t.record_stream(stream);
    file << "1";
  } catch (const std::exception& e) {
    file << "exception";
  }
  file << "\n";
  file.saveFile();
}

// kFloat, shape {0} (空 tensor，边界 shape)
TEST_F(RecordStreamTest, CudaEmptyShape) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "CudaEmptyShape ";
  if (!torch::cuda::is_available()) {
    file << "no_cuda";
    file << "\n";
    file.saveFile();
    return;
  }
  try {
    at::Tensor t = at::zeros({0}, at::kFloat).cuda();
    at::Stream stream = get_default_cuda_stream();
    t.record_stream(stream);
    file << "1";
  } catch (const std::exception& e) {
    file << "exception";
  }
  file << "\n";
  file.saveFile();
}

// kFloat, shape {1,1,1} (全一维度，边界 shape)
TEST_F(RecordStreamTest, CudaAllOnes) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "CudaAllOnes ";
  if (!torch::cuda::is_available()) {
    file << "no_cuda";
    file << "\n";
    file.saveFile();
    return;
  }
  try {
    at::Tensor t = at::zeros({1, 1, 1}, at::kFloat).cuda();
    at::Stream stream = get_default_cuda_stream();
    t.record_stream(stream);
    file << "1";
  } catch (const std::exception& e) {
    file << "exception";
  }
  file << "\n";
  file.saveFile();
}

// kFloat, 非连续 tensor（经 transpose）
TEST_F(RecordStreamTest, CudaNonContiguous) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "CudaNonContiguous ";
  if (!torch::cuda::is_available()) {
    file << "no_cuda";
    file << "\n";
    file.saveFile();
    return;
  }
  try {
    at::Tensor base = at::zeros({3, 4}, at::kFloat).cuda();
    at::Tensor t = base.transpose(0, 1);  // 非连续
    at::Stream stream = get_default_cuda_stream();
    t.record_stream(stream);
    file << "1";
  } catch (const std::exception& e) {
    file << "exception";
  }
  file << "\n";
  file.saveFile();
}

// --- 异常路径：CPU tensor + CUDA stream（如有 CUDA） ---
// record_stream 在两个框架下对 CPU tensor 的处理行为
TEST_F(RecordStreamTest, CpuTensorCudaStream) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "CpuTensorCudaStream ";
  if (!torch::cuda::is_available()) {
    file << "no_cuda";
    file << "\n";
    file.saveFile();
    return;
  }
  try {
    at::Stream stream = get_default_cuda_stream();
    cpu_tensor.record_stream(stream);
    file << "1";
  } catch (const std::exception& e) {
    file << "exception";
  }
  file << "\n";
  file.saveFile();
}

// --- 异常路径：CPU tensor + CPU stream（无 CUDA 依赖） ---
// record_stream 是 CUDA-only API，CPU stream 应触发异常
TEST_F(RecordStreamTest, CpuTensorCpuStream) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "CpuTensorCpuStream ";
  c10::Stream stream(c10::Stream::DEFAULT,
                     c10::Device(c10::DeviceType::CPU, 0));
  try {
    cpu_tensor.record_stream(stream);
    file << "1";
  } catch (const std::exception& e) {
    file << "exception";
  }
  file << "\n";
  file.saveFile();
}

}  // namespace test
}  // namespace at
