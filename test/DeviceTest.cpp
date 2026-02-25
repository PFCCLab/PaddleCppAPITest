#include <ATen/ATen.h>
#include <ATen/core/Tensor.h>
#include <ATen/ops/zeros.h>
#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "../src/file_manager.h"

extern paddle_api_test::ThreadSafeParam g_custom_param;

namespace at {
namespace test {

using paddle_api_test::FileManerger;
using paddle_api_test::ThreadSafeParam;

class DeviceTest : public ::testing::Test {
 protected:
  void SetUp() override {}
};

// 辅助函数：将 Device 结果写入文件
static void write_device_result_to_file(const FileManerger& file,
                                        const c10::Device& device) {
  file << std::to_string(static_cast<int>(device.type())) << " ";
  file << std::to_string(static_cast<int>(device.index())) << " ";
  file << (device.is_cuda() ? "1" : "0") << " ";
  file << (device.is_cpu() ? "1" : "0") << " ";
  file << (device.has_index() ? "1" : "0") << " ";
  file << device.str() << " ";
}

// ==================== Device 构造函数测试 ====================

// 测试 Device(DeviceType, DeviceIndex) 构造函数
TEST_F(DeviceTest, ConstructorWithTypeAndIndex) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();

  // CPU 设备，默认 index=0
  c10::Device cpu_device(c10::kCPU);
  write_device_result_to_file(file, cpu_device);

  // CPU 设备，显式 index=0
  c10::Device cpu_device_0(c10::kCPU, 0);
  write_device_result_to_file(file, cpu_device_0);

  // CUDA 设备，index=0
  c10::Device cuda_device_0(c10::kCUDA, 0);
  write_device_result_to_file(file, cuda_device_0);

  // CUDA 设备，index=1
  c10::Device cuda_device_1(c10::kCUDA, 1);
  write_device_result_to_file(file, cuda_device_1);

  file.saveFile();
}

// 测试 Device(std::string) 字符串构造
TEST_F(DeviceTest, ConstructorWithString) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();

  // "cpu" 字符串
  c10::Device cpu_str("cpu");
  write_device_result_to_file(file, cpu_str);

  // "cpu:0" 字符串
  c10::Device cpu0_str("cpu:0");
  write_device_result_to_file(file, cpu0_str);

  // "cuda" 字符串
  c10::Device cuda_str("cuda");
  write_device_result_to_file(file, cuda_str);

  // "cuda:0" 字符串
  c10::Device cuda0_str("cuda:0");
  write_device_result_to_file(file, cuda0_str);

  // "cuda:1" 字符串
  c10::Device cuda1_str("cuda:1");
  write_device_result_to_file(file, cuda1_str);

  file.saveFile();
}

// ==================== Device 属性测试 ====================

// 测试 index() 和 has_index()
TEST_F(DeviceTest, IndexAndHasIndex) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();

  // CPU 设备
  c10::Device cpu_device(c10::kCPU);
  file << std::to_string(cpu_device.index()) << " ";
  file << (cpu_device.has_index() ? "1" : "0") << " ";

  // CUDA 设备 index=0
  c10::Device cuda_0(c10::kCUDA, 0);
  file << std::to_string(cuda_0.index()) << " ";
  file << (cuda_0.has_index() ? "1" : "0") << " ";

  // CUDA 设备 index=1
  c10::Device cuda_1(c10::kCUDA, 1);
  file << std::to_string(cuda_1.index()) << " ";
  file << (cuda_1.has_index() ? "1" : "0") << " ";

  // 字符串构造的设备
  c10::Device cpu_str("cpu");
  file << std::to_string(cpu_str.index()) << " ";
  file << (cpu_str.has_index() ? "1" : "0") << " ";

  c10::Device cuda0_str("cuda:0");
  file << std::to_string(cuda0_str.index()) << " ";
  file << (cuda0_str.has_index() ? "1" : "0") << " ";

  file.saveFile();
}

// 测试 type() 设备类型
TEST_F(DeviceTest, DeviceType) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();

  c10::Device cpu_device(c10::kCPU);
  file << std::to_string(static_cast<int>(cpu_device.type())) << " ";

  c10::Device cuda_device(c10::kCUDA, 0);
  file << std::to_string(static_cast<int>(cuda_device.type())) << " ";

  // 从字符串解析
  c10::Device cpu_str("cpu");
  file << std::to_string(static_cast<int>(cpu_str.type())) << " ";

  c10::Device cuda_str("cuda:0");
  file << std::to_string(static_cast<int>(cuda_str.type())) << " ";

  file.saveFile();
}

// 测试 is_cuda() 和 is_cpu()
TEST_F(DeviceTest, IsCudaAndIsCpu) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();

  // CPU 设备
  c10::Device cpu_device(c10::kCPU);
  file << (cpu_device.is_cpu() ? "1" : "0") << " ";
  file << (cpu_device.is_cuda() ? "1" : "0") << " ";

  // CUDA 设备
  c10::Device cuda_device(c10::kCUDA, 0);
  file << (cuda_device.is_cpu() ? "1" : "0") << " ";
  file << (cuda_device.is_cuda() ? "1" : "0") << " ";

  // 字符串构造的 CPU
  c10::Device cpu_str("cpu");
  file << (cpu_str.is_cpu() ? "1" : "0") << " ";
  file << (cpu_str.is_cuda() ? "1" : "0") << " ";

  // 字符串构造的 CUDA
  c10::Device cuda_str("cuda:0");
  file << (cuda_str.is_cpu() ? "1" : "0") << " ";
  file << (cuda_str.is_cuda() ? "1" : "0") << " ";

  file.saveFile();
}

// 测试 str() 字符串表示
TEST_F(DeviceTest, ToString) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();

  c10::Device cpu_device(c10::kCPU);
  file << cpu_device.str() << " ";

  c10::Device cuda_0(c10::kCUDA, 0);
  file << cuda_0.str() << " ";

  c10::Device cuda_1(c10::kCUDA, 1);
  file << cuda_1.str() << " ";

  c10::Device cpu_str("cpu:0");
  file << cpu_str.str() << " ";

  c10::Device cuda_str("cuda:1");
  file << cuda_str.str() << " ";

  file.saveFile();
}

// ==================== Device 比较测试 ====================

// 测试 operator==
TEST_F(DeviceTest, Equality) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();

  // 相同设备
  c10::Device cpu1(c10::kCPU);
  c10::Device cpu2(c10::kCPU);
  file << (cpu1 == cpu2 ? "1" : "0") << " ";

  // 相同类型不同 index
  c10::Device cuda0(c10::kCUDA, 0);
  c10::Device cuda1(c10::kCUDA, 1);
  file << (cuda0 == cuda1 ? "1" : "0") << " ";

  // 不同类型
  c10::Device cpu(c10::kCPU);
  c10::Device cuda(c10::kCUDA, 0);
  file << (cpu == cuda ? "1" : "0") << " ";

  // 字符串构造 vs 构造函数
  c10::Device cpu_cons(c10::kCPU);
  c10::Device cpu_str("cpu");
  file << (cpu_cons == cpu_str ? "1" : "0") << " ";

  c10::Device cuda0_cons(c10::kCUDA, 0);
  c10::Device cuda0_str("cuda:0");
  file << (cuda0_cons == cuda0_str ? "1" : "0") << " ";

  file.saveFile();
}

// ==================== Tensor 与 Device 交互测试 ====================

// 测试 Tensor 的 device() 方法
TEST_F(DeviceTest, TensorDevice) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();

  // 默认 CPU tensor
  at::Tensor cpu_tensor = at::zeros({2, 3});
  c10::Device cpu_dev = cpu_tensor.device();
  write_device_result_to_file(file, cpu_dev);

  // 指定 CPU device 的 tensor
  at::Tensor cpu_tensor2 =
      at::zeros({2, 3}, at::TensorOptions().device(c10::kCPU));
  c10::Device cpu_dev2 = cpu_tensor2.device();
  write_device_result_to_file(file, cpu_dev2);

  // 使用 TensorOptions 构造
  at::Tensor cpu_tensor3 =
      at::zeros({2, 3}, at::TensorOptions().device(c10::Device(c10::kCPU)));
  c10::Device cpu_dev3 = cpu_tensor3.device();
  write_device_result_to_file(file, cpu_dev3);

  file.saveFile();
}

// 测试 TensorOptions 设置 device
TEST_F(DeviceTest, TensorOptionsDevice) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();

  // 使用 TensorOptions 设置 CPU
  at::TensorOptions opts_cpu = at::TensorOptions().device(c10::kCPU);
  at::Tensor tensor_cpu = at::zeros({2, 2}, opts_cpu);
  c10::Device dev_cpu = tensor_cpu.device();
  write_device_result_to_file(file, dev_cpu);

  file.saveFile();
}

// ==================== Device 常量测试 ====================

// 测试 kCPU 和 kCUDA 常量
TEST_F(DeviceTest, DeviceConstants) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();

  // 使用 kCPU 常量
  c10::Device dev_kcpu(c10::kCPU);
  write_device_result_to_file(file, dev_kcpu);

  // 使用 kCUDA 常量
  c10::Device dev_kcuda(c10::kCUDA, 0);
  write_device_result_to_file(file, dev_kcuda);

  // 在 TensorOptions 中使用
  at::Tensor tensor_kcpu =
      at::zeros({2}, at::TensorOptions().device(c10::kCPU));
  write_device_result_to_file(file, tensor_kcpu.device());

  file.saveFile();
}

// ==================== Device 边界情况测试 ====================

// 测试负索引
TEST_F(DeviceTest, NegativeIndex) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();

  // CUDA 负索引 - 应该返回 -1
  c10::Device cuda_neg(c10::kCUDA, -1);
  file << std::to_string(static_cast<int>(cuda_neg.type())) << " ";
  file << std::to_string(static_cast<int>(cuda_neg.index())) << " ";
  file << (cuda_neg.has_index() ? "1" : "0") << " ";

  // CPU 负索引
  c10::Device cpu_neg(c10::kCPU, -1);
  file << std::to_string(static_cast<int>(cpu_neg.type())) << " ";
  file << std::to_string(static_cast<int>(cpu_neg.index())) << " ";
  file << (cpu_neg.has_index() ? "1" : "0") << " ";

  file.saveFile();
}

}  // namespace test
}  // namespace at
