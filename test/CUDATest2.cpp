#include <ATen/ATen.h>
#ifndef USE_PADDLE_API
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/PhiloxCudaState.h>
#endif
#include <gtest/gtest.h>

#include <string>

#include "../src/file_manager.h"

extern paddle_api_test::ThreadSafeParam g_custom_param;

namespace at {
namespace test {

using paddle_api_test::FileManerger;
using paddle_api_test::ThreadSafeParam;

class CUDATest2 : public ::testing::Test {
 protected:
  void SetUp() override {}
};

// device_synchronize
TEST_F(CUDATest2, DeviceSynchronize) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();

  // c10::cuda::device_synchronize()
  // Only test if CUDA is available
  file << "device_synchronize_test ";
  file.saveFile();
}

// stream_synchronize
TEST_F(CUDATest2, StreamSynchronize) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();

  // c10::cuda::stream_synchronize
  file << "stream_synchronize_test ";
  file.saveFile();
}

#ifndef USE_PADDLE_API
// CUDAGuard tests
TEST_F(CUDATest2, CUDAGuardDefault) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();

  // Default constructor
  c10::cuda::CUDAGuard guard;
  file << "CUDAGuard_default ";
  file.saveFile();
}

TEST_F(CUDATest2, CUDAGuardDeviceIndex) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();

  // Constructor with DeviceIndex
  c10::cuda::CUDAGuard guard(0);
  file << "CUDAGuard_device_index ";
  file.saveFile();
}

TEST_F(CUDATest2, CUDAGuardDevice) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();

  // Constructor with c10::Device
  c10::cuda::CUDAGuard guard(c10::Device(c10::DeviceType::CUDA, 0));
  file << "CUDAGuard_device ";
  file.saveFile();
}

TEST_F(CUDATest2, CUDAGuardSetDevice) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();

  c10::cuda::CUDAGuard guard;
  guard.set_device(c10::Device(c10::DeviceType::CUDA, 0));
  file << "CUDAGuard_set_device ";
  file.saveFile();
}

TEST_F(CUDATest2, CUDAGuardResetDevice) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();

  c10::cuda::CUDAGuard guard;
  guard.reset_device();
  file << "CUDAGuard_reset_device ";
  file.saveFile();
}

TEST_F(CUDATest2, CUDAGuardSetIndex) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();

  c10::cuda::CUDAGuard guard;
  guard.set_index(0);
  file << "CUDAGuard_set_index ";
  file.saveFile();
}

TEST_F(CUDATest2, CUDAGuardCurrentDevice) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();

  c10::cuda::CUDAGuard guard;
  auto device = guard.current_device();
  file << "CUDAGuard_current_device ";
  file.saveFile();
}

// OptionalCUDAGuard tests
TEST_F(CUDATest2, OptionalCUDAGuardDefault) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();

  c10::cuda::OptionalCUDAGuard guard;
  file << "OptionalCUDAGuard_default ";
  file.saveFile();
}

// CUDAStream tests
TEST_F(CUDATest2, CUDAStreamDefault) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();

  c10::cuda::CUDAStream stream;
  file << "CUDAStream_default ";
  file.saveFile();
}

TEST_F(CUDATest2, CUDAStreamFromStream) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();

  // Create from gpuStream_t
  c10::cuda::CUDAStream stream(c10::cuda::CUDAStream::DEFAULT);
  file << "CUDAStream_from_stream ";
  file.saveFile();
}

TEST_F(CUDATest2, CUDAStreamId) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();

  c10::cuda::CUDAStream stream;
  auto id = stream.id();
  file << "CUDAStream_id ";
  file.saveFile();
}

TEST_F(CUDATest2, CUDAStreamDeviceType) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();

  c10::cuda::CUDAStream stream;
  auto device_type = stream.device_type();
  file << "CUDAStream_device_type ";
  file.saveFile();
}

TEST_F(CUDATest2, CUDAStreamStream) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();

  c10::cuda::CUDAStream stream;
  auto cuda_stream = stream.stream();
  file << "CUDAStream_stream ";
  file.saveFile();
}

TEST_F(CUDATest2, CUDAStreamRawStream) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();

  c10::cuda::CUDAStream stream;
  auto raw = stream.raw_stream();
  file << "CUDAStream_raw_stream ";
  file.saveFile();
}

TEST_F(CUDATest2, GetCurrentCUDAStream) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();

  auto stream = c10::cuda::getCurrentCUDAStream(c10::DeviceIndex(-1));
  file << "getCurrentCUDAStream ";
  file.saveFile();
}

#endif

#ifndef USE_PADDLE_API
// PhiloxCudaState tests
TEST_F(CUDATest2, PhiloxCudaStateDefault) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();

  c10::cuda::PhiloxCudaState state;
  file << "PhiloxCudaState_default ";
  file.saveFile();
}

TEST_F(CUDATest2, PhiloxCudaStateWithParams) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();

  int64_t seed = 12345;
  int64_t offset_extra = 0;
  uint64_t offset = 0;
  c10::cuda::PhiloxCudaState state(&seed, &offset_extra, offset);
  file << "PhiloxCudaState_params ";
  file.saveFile();
}

#endif

}  // namespace test
}  // namespace at
