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

// [DIFF] 文件级说明：该文件大部分用例依赖 c10::cuda::* 具体实现，
// 当前主要在 Torch 路径可稳定编译/运行，Paddle 路径能力面不对齐。

// device_synchronize
TEST_F(CUDATest2, DeviceSynchronize) {
  // [DIFF] 用例级差异：该用例仅写占位结果，未执行等价 API，
  // 反映两端在同步语义上的可比性不足。
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();
  file << "DeviceSynchronize ";

  // c10::cuda::device_synchronize()
  // Only test if CUDA is available
  file << "device_synchronize_test ";
  file << "\n";
  file.saveFile();
}

// stream_synchronize
TEST_F(CUDATest2, StreamSynchronize) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "StreamSynchronize ";

#ifndef USE_PADDLE_API
  if (!at::cuda::is_available()) {
    GTEST_SKIP() << "CUDA not available";
  }

  try {
    auto stream = c10::cuda::getCurrentCUDAStream();
    c10::cuda::stream_synchronize(stream.stream());
    file << "1 ";
  } catch (const std::exception& e) {
    file << "exception ";
  }
#else
  // Paddle 兼容头当前 stream
  // 类型定义与该测试编译单元存在依赖差异，先保留占位输出。
  file << "stream_sync_placeholder ";
#endif
  file << "\n";
  file.saveFile();
}

#ifndef USE_PADDLE_API
// [DIFF] 问题行：以下测试块仅在 !USE_PADDLE_API 下编译，
// 说明 CUDAGuard/CUDAStream/Philox 在 Paddle 侧接口不完整或行为不一致。
// CUDAGuard tests
TEST_F(CUDATest2, CUDAGuardDefault) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "CUDAGuardDefault ";

  // Default constructor
  c10::cuda::CUDAGuard guard;
  file << "CUDAGuard_default ";
  file << "\n";
  file.saveFile();
}

TEST_F(CUDATest2, CUDAGuardDeviceIndex) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "CUDAGuardDeviceIndex ";

  // Constructor with DeviceIndex
  c10::cuda::CUDAGuard guard(0);
  file << "CUDAGuard_device_index ";
  file << "\n";
  file.saveFile();
}

TEST_F(CUDATest2, CUDAGuardDevice) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "CUDAGuardDevice ";

  // Constructor with c10::Device
  c10::cuda::CUDAGuard guard(c10::Device(c10::DeviceType::CUDA, 0));
  file << "CUDAGuard_device ";
  file << "\n";
  file.saveFile();
}

TEST_F(CUDATest2, CUDAGuardSetDevice) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "CUDAGuardSetDevice ";

  c10::cuda::CUDAGuard guard;
  guard.set_device(c10::Device(c10::DeviceType::CUDA, 0));
  file << "CUDAGuard_set_device ";
  file << "\n";
  file.saveFile();
}

TEST_F(CUDATest2, CUDAGuardResetDevice) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "CUDAGuardResetDevice ";

  c10::cuda::CUDAGuard guard;
  guard.reset_device();
  file << "CUDAGuard_reset_device ";
  file << "\n";
  file.saveFile();
}

TEST_F(CUDATest2, CUDAGuardSetIndex) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "CUDAGuardSetIndex ";

  c10::cuda::CUDAGuard guard;
  guard.set_index(0);
  file << "CUDAGuard_set_index ";
  file << "\n";
  file.saveFile();
}

TEST_F(CUDATest2, CUDAGuardCurrentDevice) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "CUDAGuardCurrentDevice ";

  c10::cuda::CUDAGuard guard;
  auto device = guard.current_device();
  file << "CUDAGuard_current_device ";
  file << "\n";
  file.saveFile();
}

// OptionalCUDAGuard tests
TEST_F(CUDATest2, OptionalCUDAGuardDefault) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "OptionalCUDAGuardDefault ";

  c10::cuda::OptionalCUDAGuard guard;
  file << "OptionalCUDAGuard_default ";
  file << "\n";
  file.saveFile();
}

// CUDAStream tests
TEST_F(CUDATest2, CUDAStreamDefault) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "CUDAStreamDefault ";

  c10::cuda::CUDAStream stream;
  file << "CUDAStream_default ";
  file << "\n";
  file.saveFile();
}

TEST_F(CUDATest2, CUDAStreamFromStream) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "CUDAStreamFromStream ";

  // Create from gpuStream_t
  c10::cuda::CUDAStream stream(c10::cuda::CUDAStream::DEFAULT);
  file << "CUDAStream_from_stream ";
  file << "\n";
  file.saveFile();
}

TEST_F(CUDATest2, CUDAStreamId) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "CUDAStreamId ";

  c10::cuda::CUDAStream stream;
  auto id = stream.id();
  file << "CUDAStream_id ";
  file << "\n";
  file.saveFile();
}

TEST_F(CUDATest2, CUDAStreamDeviceType) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "CUDAStreamDeviceType ";

  c10::cuda::CUDAStream stream;
  auto device_type = stream.device_type();
  file << "CUDAStream_device_type ";
  file << "\n";
  file.saveFile();
}

TEST_F(CUDATest2, CUDAStreamStream) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "CUDAStreamStream ";

  c10::cuda::CUDAStream stream;
  auto cuda_stream = stream.stream();
  file << "CUDAStream_stream ";
  file << "\n";
  file.saveFile();
}

TEST_F(CUDATest2, CUDAStreamRawStream) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "CUDAStreamRawStream ";

  c10::cuda::CUDAStream stream;
  auto raw = stream.raw_stream();
  file << "CUDAStream_raw_stream ";
  file << "\n";
  file.saveFile();
}

TEST_F(CUDATest2, GetCurrentCUDAStream) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "GetCurrentCUDAStream ";

  auto stream = c10::cuda::getCurrentCUDAStream(c10::DeviceIndex(-1));
  file << "getCurrentCUDAStream ";
  file << "\n";
  file.saveFile();
}

#endif

#ifndef USE_PADDLE_API
// PhiloxCudaState tests
TEST_F(CUDATest2, PhiloxCudaStateDefault) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "PhiloxCudaStateDefault ";

  c10::cuda::PhiloxCudaState state;
  file << "PhiloxCudaState_default ";
  file << "\n";
  file.saveFile();
}

TEST_F(CUDATest2, PhiloxCudaStateWithParams) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "PhiloxCudaStateWithParams ";

  int64_t seed = 12345;
  int64_t offset_extra = 0;
  uint64_t offset = 0;
  c10::cuda::PhiloxCudaState state(&seed, &offset_extra, offset);
  file << "PhiloxCudaState_params ";
  file << "\n";
  file.saveFile();
}

#endif

}  // namespace test
}  // namespace at
