#include <ATen/ATen.h>
#include <ATen/DeviceGuard.h>
#include <gtest/gtest.h>

#include <iostream>

#include "../src/file_manager.h"

extern paddle_api_test::ThreadSafeParam g_custom_param;

namespace at {
namespace test {

using paddle_api_test::FileManerger;
using paddle_api_test::ThreadSafeParam;

static void write_device_result_to_file(FileManerger* file,
                                        const std::optional<at::Device>& dev) {
  if (dev.has_value()) {
    *file << dev->type() << " " << static_cast<int>(dev->index()) << " ";
  } else {
    *file << "nullopt ";
  }
}

TEST(DeviceGuardTest, DeviceOfTensor) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();
  file << "DeviceOfTensor ";

  // 1. CPU Tensor
  at::Tensor t_cpu = at::ones({2, 3}, at::kFloat);
  std::optional<at::Device> dev_cpu = at::device_of(t_cpu);
  write_device_result_to_file(&file, dev_cpu);

  // 2. Empty Tensor
  at::Tensor t_empty = at::empty({0}, at::kFloat);
  std::optional<at::Device> dev_empty = at::device_of(t_empty);
  write_device_result_to_file(&file, dev_empty);

  file << "\n";
  file.saveFile();
}

TEST(DeviceGuardTest, DeviceOfOptionalTensor) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "DeviceOfOptionalTensor ";

  // 1. nullopt
  std::optional<at::Tensor> opt_t_null = std::nullopt;
  std::optional<at::Device> dev_null = at::device_of(opt_t_null);
  write_device_result_to_file(&file, dev_null);

  // 2. optional with tensor
  std::optional<at::Tensor> opt_t_value = at::ones({1}, at::kInt);
  std::optional<at::Device> dev_value = at::device_of(opt_t_value);
  write_device_result_to_file(&file, dev_value);

  file << "\n";
  file.saveFile();
}

}  // namespace test
}  // namespace at
