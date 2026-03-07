// Tests for:
//   at::Generator
//   at::cuda::detail::getDefaultCUDAGenerator
//   at::get_generator_or_default
//   at::CUDAGeneratorImpl

#include <ATen/core/Generator.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <gtest/gtest.h>

#include <optional>
#include <string>

#include "../src/file_manager.h"

extern paddle_api_test::ThreadSafeParam g_custom_param;

namespace at {
namespace test {

using paddle_api_test::FileManerger;
using paddle_api_test::ThreadSafeParam;

class GeneratorTest : public ::testing::Test {
 protected:
  void SetUp() override {
    test_gen_ = at::cuda::detail::createCUDAGenerator(0);
    test_gen_.set_current_seed(42);
  }
  at::Generator test_gen_;
};

// ============================================================
// at::Generator
// ============================================================

// 默认构造函数产生未定义（impl 为 nullptr）的 Generator
TEST_F(GeneratorTest, DefaultConstructorUndefined) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();
  file << "DefaultConstructorUndefined ";
  at::Generator gen;
  file << std::to_string(gen.defined()) << " ";
  file << "\n";
  file.saveFile();
}

// set_current_seed / current_seed 往返测试
TEST_F(GeneratorTest, SetAndGetSeed) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "SetAndGetSeed ";
  test_gen_.set_current_seed(12345);
  file << std::to_string(test_gen_.current_seed()) << " ";
  file << "\n";
  file.saveFile();
}

// set_offset / get_offset 往返测试
TEST_F(GeneratorTest, SetAndGetOffset) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "SetAndGetOffset ";
  test_gen_.set_offset(100);
  file << std::to_string(test_gen_.get_offset()) << " ";
  file << "\n";
  file.saveFile();
}

// device() 返回 CUDA 设备，设备字符串为 "cuda:0"
TEST_F(GeneratorTest, DeviceIsCUDA) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "DeviceIsCUDA ";
  file << test_gen_.device().str() << " ";
  file << "\n";
  file.saveFile();
}

// clone() 复制种子状态，修改克隆体不影响原体
TEST_F(GeneratorTest, CloneIndependence) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "CloneIndependence ";
  // SetUp 中已设置 seed = 42
  at::Generator cloned = test_gen_.clone();
  cloned.set_current_seed(999);
  // 原体 seed 仍为 42，克隆体 seed 为 999
  file << std::to_string(test_gen_.current_seed()) << " ";
  file << std::to_string(cloned.current_seed()) << " ";
  file << "\n";
  file.saveFile();
}

// operator== 同一底层实现时相等；不同实例不相等
TEST_F(GeneratorTest, EqualityOperators) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "EqualityOperators ";
  at::Generator other = at::cuda::detail::createCUDAGenerator(0);
  file << std::to_string(test_gen_ == test_gen_) << " ";  // 1
  file << std::to_string(test_gen_ != other) << " ";      // 1
  file << "\n";
  file.saveFile();
}

// ============================================================
// at::cuda::detail::getDefaultCUDAGenerator
// ============================================================

// 返回已定义的 Generator
TEST_F(GeneratorTest, GetDefaultCUDAGeneratorDefined) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "GetDefaultCUDAGeneratorDefined ";
  const at::Generator& gen = at::cuda::detail::getDefaultCUDAGenerator(0);
  file << std::to_string(gen.defined()) << " ";
  file << "\n";
  file.saveFile();
}

// 返回的 Generator 设备为 cuda:0
TEST_F(GeneratorTest, GetDefaultCUDAGeneratorDevice) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "GetDefaultCUDAGeneratorDevice ";
  const at::Generator& gen = at::cuda::detail::getDefaultCUDAGenerator(0);
  file << gen.device().str() << " ";
  file << "\n";
  file.saveFile();
}

// 多次调用返回同一实例（地址相同）
TEST_F(GeneratorTest, GetDefaultCUDAGeneratorSameInstance) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "GetDefaultCUDAGeneratorSameInstance ";
  const at::Generator& gen1 = at::cuda::detail::getDefaultCUDAGenerator(0);
  const at::Generator& gen2 = at::cuda::detail::getDefaultCUDAGenerator(0);
  file << std::to_string(&gen1 == &gen2) << " ";
  file << "\n";
  file.saveFile();
}

// ============================================================
// at::get_generator_or_default<T>
// ============================================================

// 提供有效 Generator 时，返回该 Generator 的 impl
TEST_F(GeneratorTest, GetOrDefaultUsesProvidedGenerator) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "GetOrDefaultUsesProvidedGenerator ";
  test_gen_.set_current_seed(55555);
  std::optional<at::Generator> opt_gen = test_gen_;
  const at::Generator& default_gen =
      at::cuda::detail::getDefaultCUDAGenerator(0);
  at::CUDAGeneratorImpl* impl =
      at::get_generator_or_default<at::CUDAGeneratorImpl>(opt_gen, default_gen);
  file << std::to_string(impl->current_seed() == 55555) << " ";
  file << "\n";
  file.saveFile();
}

// nullopt 时，返回 default_gen 的 impl
TEST_F(GeneratorTest, GetOrDefaultUsesDefaultWhenNullopt) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "GetOrDefaultUsesDefaultWhenNullopt ";
  std::optional<at::Generator> opt_gen = std::nullopt;
  at::Generator custom_default = at::cuda::detail::createCUDAGenerator(0);
  custom_default.set_current_seed(77777);
  at::CUDAGeneratorImpl* impl =
      at::get_generator_or_default<at::CUDAGeneratorImpl>(opt_gen,
                                                          custom_default);
  file << std::to_string(impl->current_seed() == 77777) << " ";
  file << "\n";
  file.saveFile();
}

// ============================================================
// at::CUDAGeneratorImpl
// ============================================================

// 静态方法 device_type() 返回 kCUDA
TEST_F(GeneratorTest, CUDAImplDeviceType) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "CUDAImplDeviceType ";
  file << std::to_string(at::CUDAGeneratorImpl::device_type() ==
                         c10::DeviceType::CUDA)
       << " ";
  file << "\n";
  file.saveFile();
}

// set_philox_offset_per_thread / philox_offset_per_thread 往返测试
TEST_F(GeneratorTest, CUDAImplPhiloxOffset) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "CUDAImplPhiloxOffset ";
  at::CUDAGeneratorImpl* impl = test_gen_.get<at::CUDAGeneratorImpl>();
  impl->set_philox_offset_per_thread(256);
  file << std::to_string(impl->philox_offset_per_thread()) << " ";
  file << "\n";
  file.saveFile();
}

// philox_cuda_state(increment) 将 offset 推进 increment
TEST_F(GeneratorTest, CUDAImplPhiloxCudaState) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "CUDAImplPhiloxCudaState ";
  at::CUDAGeneratorImpl* impl = test_gen_.get<at::CUDAGeneratorImpl>();
  impl->set_philox_offset_per_thread(0);
  impl->set_current_seed(12345);
  impl->philox_cuda_state(4);
  // offset 应已推进 4
  file << std::to_string(impl->philox_offset_per_thread()) << " ";
  file << "\n";
  file.saveFile();
}

// philox_engine_inputs(increment) 返回 (seed, offset_before)，并推进 offset
TEST_F(GeneratorTest, CUDAImplPhiloxEngineInputs) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "CUDAImplPhiloxEngineInputs ";
  at::CUDAGeneratorImpl* impl = test_gen_.get<at::CUDAGeneratorImpl>();
  impl->set_current_seed(99999);
  impl->set_philox_offset_per_thread(0);
  auto inputs = impl->philox_engine_inputs(8);
  // inputs.first == seed, inputs.second == offset before call
  file << std::to_string(inputs.first) << " ";
  file << std::to_string(inputs.second) << " ";
  // offset 推进 8
  file << std::to_string(impl->philox_offset_per_thread()) << " ";
  file << "\n";
  file.saveFile();
}

// clone() 保留 seed 与 offset；修改克隆体不影响原体
TEST_F(GeneratorTest, CUDAImplClone) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "CUDAImplClone ";
  at::CUDAGeneratorImpl* impl = test_gen_.get<at::CUDAGeneratorImpl>();
  impl->set_current_seed(12345);
  impl->set_philox_offset_per_thread(100);
  at::Generator cloned_gen = test_gen_.clone();
  at::CUDAGeneratorImpl* cloned_impl = cloned_gen.get<at::CUDAGeneratorImpl>();
  // 克隆体拥有相同的 seed 和 offset
  file << std::to_string(cloned_impl->current_seed()) << " ";
  file << std::to_string(cloned_impl->philox_offset_per_thread()) << " ";
  // 修改克隆体后，原体 seed 不变
  cloned_impl->set_current_seed(999);
  file << std::to_string(impl->current_seed()) << " ";
  file << "\n";
  file.saveFile();
}

}  // namespace test
}  // namespace at
