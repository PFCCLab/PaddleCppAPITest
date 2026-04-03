#include <ATen/ATen.h>
#include <c10/core/Device.h>
#include <gtest/gtest.h>

#include <string>
#include <unordered_map>

#include "src/file_manager.h"

extern paddle_api_test::ThreadSafeParam g_custom_param;

namespace at {
namespace test {

using paddle_api_test::FileManerger;
using paddle_api_test::ThreadSafeParam;

class DeviceCompatTest : public ::testing::Test {
 protected:
  void SetUp() override {}
};

static void write_device_summary(FileManerger* file,
                                 const c10::Device& device) {
  *file << static_cast<int>(device.type()) << " ";
  *file << static_cast<int>(device.index()) << " ";
  *file << (device.has_index() ? "1" : "0") << " ";
  *file << device.str() << " ";
}

template <typename Fn>
static bool throws_any(Fn&& fn) {
  try {
    fn();
    return false;
  } catch (...) {
    return true;
  }
}

TEST_F(DeviceCompatTest, DeviceStr) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();
  file << "DeviceStr ";

  c10::Device cpu_device(c10::kCPU);
  auto cpu_str = cpu_device.str();

  c10::Device cpu_device_0(c10::kCPU, 0);
  auto cpu_0_str = cpu_device_0.str();

  c10::Device cuda_device_0(c10::kCUDA, 0);
  auto cuda_0_str = cuda_device_0.str();

  c10::Device cuda_device_1(c10::kCUDA, 1);
  auto cuda_1_str = cuda_device_1.str();

  // 当前 compat 已与 PyTorch 对齐：字符串输出统一为
  // `cpu cpu:0 cuda:0 cuda:1`。
  file << cpu_str << " ";
  file << cpu_0_str << " ";
  file << cuda_0_str << " ";
  file << cuda_1_str << " ";

  file << "\n";
  file.saveFile();
}

TEST_F(DeviceCompatTest, HasIndex) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "HasIndex ";

  c10::Device cpu_default(c10::kCPU);
  c10::Device cpu_0(c10::kCPU, 0);
  c10::Device cuda_default(c10::kCUDA);
  c10::Device cuda_1(c10::kCUDA, 1);

  bool cpu_default_has = cpu_default.has_index();
  bool cpu_0_has = cpu_0.has_index();
  bool cuda_default_has = cuda_default.has_index();
  bool cuda_1_has = cuda_1.has_index();

  file << std::to_string(cpu_default_has ? 1 : 0) << " ";
  file << std::to_string(cpu_0_has ? 1 : 0) << " ";
  file << std::to_string(cuda_default_has ? 1 : 0) << " ";
  file << std::to_string(cuda_1_has ? 1 : 0) << " ";

  file << "\n";
  file.saveFile();
}

TEST_F(DeviceCompatTest, StrictStringParsing) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "StrictStringParsing ";

  c10::Device privateuse_device("privateuseone:3");
  file << std::to_string(privateuse_device.is_privateuseone() ? 1 : 0) << " ";
  file << privateuse_device.str() << " ";
  file << std::to_string(throws_any([] { (void)c10::Device("cuda:-1"); }) ? 1
                                                                          : 0)
       << " ";
  file << std::to_string(throws_any([] { (void)c10::Device("cuda:01"); }) ? 1
                                                                          : 0)
       << " ";
  file << std::to_string(throws_any([] { (void)c10::Device("cuda:1:2"); }) ? 1
                                                                           : 0)
       << " ";
  file << std::to_string(throws_any([] { (void)c10::Device("cpu::0"); }) ? 1
                                                                         : 0)
       << " ";

  file << "\n";
  file.saveFile();
}

TEST_F(DeviceCompatTest, PredicatesAndHash) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "PredicatesAndHash ";

  c10::Device cpu(c10::kCPU);
  c10::Device cuda(c10::kCUDA, 0);
  c10::Device xpu(c10::kXPU, 1);
  c10::Device ipu(c10::kIPU, 2);
  c10::Device privateuse(c10::kPrivateUse1, 4);

  std::unordered_map<c10::Device, int> device_map;
  device_map.emplace(c10::Device(c10::kCUDA, 0), 7);
  device_map.emplace(c10::Device(c10::kCPU), 3);

  file << std::to_string(cpu.is_cpu() ? 1 : 0) << " ";
  file << std::to_string(cuda.is_cuda() ? 1 : 0) << " ";
  file << std::to_string(xpu.is_xpu() ? 1 : 0) << " ";
  file << std::to_string(ipu.is_ipu() ? 1 : 0) << " ";
  file << std::to_string(privateuse.is_privateuseone() ? 1 : 0) << " ";
  file << std::to_string(privateuse.is_mps() ? 1 : 0) << " ";
  file << std::to_string(cpu.supports_as_strided() ? 1 : 0) << " ";
  file << std::to_string(ipu.supports_as_strided() ? 1 : 0) << " ";
  file << std::to_string(cpu != cuda ? 1 : 0) << " ";
  file << std::to_string(cuda == c10::Device(c10::kCUDA, 0) ? 1 : 0) << " ";
  file << std::to_string(device_map.at(c10::Device(c10::kCUDA, 0))) << " ";
  file << std::to_string(device_map.at(c10::Device(c10::kCPU))) << " ";

  file << "\n";
  file.saveFile();
}

TEST_F(DeviceCompatTest, SetIndexAndTensorDevice) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "SetIndexAndTensorDevice ";

  c10::Device cpu(c10::kCPU);
  c10::Device cuda(c10::kCUDA);
  cpu.set_index(0);
  cuda.set_index(2);
  write_device_summary(&file, cpu);
  write_device_summary(&file, cuda);

  at::Tensor default_cpu_tensor = at::zeros({2, 3});
  at::Tensor explicit_cpu_tensor =
      at::zeros({2, 3}, at::TensorOptions().device(c10::kCPU));
  write_device_summary(&file, default_cpu_tensor.device());
  write_device_summary(&file, explicit_cpu_tensor.device());

  file << "\n";
  file.saveFile();
}

}  // namespace test
}  // namespace at
