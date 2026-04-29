#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/PhiloxCudaState.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <gtest/gtest.h>

#include <functional>
#include <optional>
#include <sstream>
#include <string>
#include <tuple>

#include "src/file_manager.h"

extern paddle_api_test::ThreadSafeParam g_custom_param;

namespace at {
namespace test {

using paddle_api_test::FileManerger;
using paddle_api_test::ThreadSafeParam;

class CUDATest2 : public ::testing::Test {
 protected:
  void SetUp() override {}
};

TEST_F(CUDATest2, DeviceSynchronize) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();
  file << "DeviceSynchronize ";

  if (!at::cuda::is_available()) {
    file << "skip ";
    file << "\n";
    file.saveFile();
    return;
  }

  try {
    c10::cuda::device_synchronize();
    file << "1 ";
  } catch (const std::exception& e) {
    file << "exception " << e.what() << " ";
  }
  file << "\n";
  file.saveFile();
}

TEST_F(CUDATest2, StreamSynchronize) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "StreamSynchronize ";

  if (!at::cuda::is_available()) {
    file << "skip ";
    file << "\n";
    file.saveFile();
    return;
  }

  try {
    auto stream = c10::cuda::getCurrentCUDAStream();
    c10::cuda::stream_synchronize(stream.stream());
    file << "1 ";
  } catch (const std::exception& e) {
    file << "exception " << e.what() << " ";
  }
  file << "\n";
  file.saveFile();
}

TEST_F(CUDATest2, CUDAGuardDeviceCtor) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "CUDAGuardDeviceCtor ";

  if (!at::cuda::is_available()) {
    file << "skip ";
    file << "\n";
    file.saveFile();
    return;
  }

  c10::cuda::CUDAGuard guard(c10::Device(c10::DeviceType::CUDA, 0));
  auto original = guard.original_device();
  auto current = guard.current_device();
  file << static_cast<int>(original.index()) << " ";
  file << static_cast<int>(current.index()) << " ";
  file << current.is_cuda() << " ";
  file << "\n";
  file.saveFile();
}

TEST_F(CUDATest2, CUDAGuardLifecycle) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "CUDAGuardLifecycle ";

  if (!at::cuda::is_available()) {
    file << "skip ";
    file << "\n";
    file.saveFile();
    return;
  }

  c10::cuda::CUDAGuard guard(0);
  auto original = guard.original_device();
  guard.set_device(c10::Device(c10::DeviceType::CUDA, 0));
  guard.reset_device(c10::Device(c10::DeviceType::CUDA, 0));
  guard.set_index(0);
  auto current = guard.current_device();
  file << static_cast<int>(original.index()) << " ";
  file << static_cast<int>(current.index()) << " ";
  file << current.is_cuda() << " ";
  file << "\n";
  file.saveFile();
}

TEST_F(CUDATest2, OptionalCUDAGuardLifecycle) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "OptionalCUDAGuardLifecycle ";

  if (!at::cuda::is_available()) {
    file << "skip ";
    file << "\n";
    file.saveFile();
    return;
  }

  c10::cuda::OptionalCUDAGuard guard;
  file << guard.current_device().has_value() << " ";
  guard.set_device(c10::Device(c10::DeviceType::CUDA, 0));
  auto original = guard.original_device();
  auto current = guard.current_device();
  file << original.has_value() << " ";
  file << current.has_value() << " ";
  file << static_cast<int>(current->index()) << " ";
  guard.reset();
  file << guard.original_device().has_value() << " ";
  file << guard.current_device().has_value() << " ";
  file << "\n";
  file.saveFile();
}

TEST_F(CUDATest2, CUDAStreamRoundTrip) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "CUDAStreamRoundTrip ";

  if (!at::cuda::is_available()) {
    file << "skip ";
    file << "\n";
    file.saveFile();
    return;
  }

  auto current = c10::cuda::getCurrentCUDAStream(c10::DeviceIndex(-1));
  c10::cuda::CUDAStream unchecked(c10::cuda::CUDAStream::UNCHECKED,
                                  current.unwrap());
  auto packed = current.pack3();
  auto unpacked = c10::cuda::CUDAStream::unpack3(
      packed.stream_id, packed.device_index, packed.device_type);
  auto external = c10::cuda::getStreamFromExternal(current.stream(),
                                                   current.device_index());
  auto priority_range = c10::cuda::CUDAStream::priority_range();
  std::ostringstream oss;
  oss << current;
  auto stream_hash = std::hash<c10::cuda::CUDAStream>{}(current);
  auto unwrap_hash = std::hash<c10::Stream>{}(current.unwrap());

  file << static_cast<int>(current.device_index()) << " ";
  file << static_cast<int>(current.device_type()) << " ";
  file << (unchecked == current) << " ";
  file << (current == unpacked) << " ";
  file << (current == external) << " ";
  file << (current.stream() == static_cast<cudaStream_t>(current)) << " ";
  file << current.query() << " ";
  file << current.priority() << " ";
  file << std::get<0>(priority_range) << " ";
  file << std::get<1>(priority_range) << " ";
  file << (!oss.str().empty()) << " ";
  file << (stream_hash == unwrap_hash) << " ";
  current.synchronize();
  file << "\n";
  file.saveFile();
}

TEST_F(CUDATest2, CUDAStreamPoolAndCurrent) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "CUDAStreamPoolAndCurrent ";

  if (!at::cuda::is_available()) {
    file << "skip ";
    file << "\n";
    file.saveFile();
    return;
  }

  auto original = c10::cuda::getCurrentCUDAStream(0);
  auto pooled = c10::cuda::getStreamFromPool(false, 0);
  c10::cuda::setCurrentCUDAStream(pooled);
  auto current = c10::cuda::getCurrentCUDAStream(0);
  c10::cuda::setCurrentCUDAStream(original);
  auto restored = c10::cuda::getCurrentCUDAStream(0);
  file << (pooled == current) << " ";
  file << (original != pooled) << " ";
  file << (restored == original) << " ";
  file << "\n";
  file.saveFile();
}

TEST_F(CUDATest2, PhiloxCudaStateConstructors) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "PhiloxCudaStateConstructors ";

  at::PhiloxCudaState default_state;
  at::PhiloxCudaState plain_state(12345, 67890);
  int64_t seed = 7;
  int64_t offset_extragraph = 9;
  at::PhiloxCudaState captured_state(&seed, &offset_extragraph, 11);

  file << default_state.captured_ << " ";
  file << plain_state.seed_.val << " ";
  file << plain_state.offset_.val << " ";
  file << captured_state.captured_ << " ";
  file << captured_state.offset_intragraph_ << " ";
  file << "\n";
  file.saveFile();
}

TEST_F(CUDATest2, CudaCheckFailurePath) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "CudaCheckFailurePath ";

  try {
    C10_CUDA_CHECK(cudaErrorInvalidValue);
    file << "ok ";
  } catch (const std::exception&) {
    file << "exception ";
  }
  file << "\n";
  file.saveFile();
}

}  // namespace test
}  // namespace at
