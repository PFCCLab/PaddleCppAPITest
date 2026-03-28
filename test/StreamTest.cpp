#include <ATen/ATen.h>
#include <c10/core/Stream.h>
#include <c10/cuda/CUDAStream.h>
#include <gtest/gtest.h>
#include <torch/all.h>

#include <cstdint>
#include <functional>
#include <optional>
#include <sstream>
#include <string>

#include "../src/file_manager.h"

extern paddle_api_test::ThreadSafeParam g_custom_param;

namespace at {
namespace test {

using paddle_api_test::FileManerger;
using paddle_api_test::ThreadSafeParam;

namespace {

static void write_stream_meta(FileManerger* file, const c10::Stream& stream) {
  *file << std::to_string(static_cast<int>(stream.device_type())) << " ";
  *file << std::to_string(stream.device_index()) << " ";
  *file << std::to_string(stream.id()) << " ";
}

}  // namespace

class StreamTest : public ::testing::Test {
 protected:
  void SetUp() override {
    cpu_stream_default.emplace(c10::Stream::DEFAULT,
                               c10::Device(c10::DeviceType::CPU, 0));
    cpu_stream_unsafe.emplace(c10::Stream::UNSAFE,
                              c10::Device(c10::DeviceType::CPU, 0),
                              static_cast<c10::StreamId>(17));
  }

  std::optional<c10::Stream> cpu_stream_default;
  std::optional<c10::Stream> cpu_stream_unsafe;
};

TEST_F(StreamTest, StreamIdAliasAndData3) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();
  file << "StreamIdAliasAndData3 ";

  c10::StreamId sid = static_cast<c10::StreamId>(123);
  c10::StreamData3 packed{
      sid, static_cast<c10::DeviceIndex>(2), c10::DeviceType::CUDA};

  file << std::to_string(sizeof(c10::StreamId)) << " ";
  file << std::to_string(static_cast<int64_t>(sid)) << " ";
  file << std::to_string(static_cast<int64_t>(packed.stream_id)) << " ";
  file << std::to_string(static_cast<int>(packed.device_index)) << " ";
  file << std::to_string(static_cast<int>(packed.device_type)) << " ";
  file << "\n";
  file.saveFile();
}

TEST_F(StreamTest, CtorAndAccessors) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "CtorAndAccessors ";

  write_stream_meta(&file, *cpu_stream_default);
  write_stream_meta(&file, *cpu_stream_unsafe);
  file << std::to_string(cpu_stream_default->device().index()) << " ";
  file << std::to_string(cpu_stream_unsafe->device().index()) << " ";
  file << "\n";
  file.saveFile();
}

TEST_F(StreamTest, CompareOperators) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "CompareOperators ";

  c10::Stream same_as_default(c10::Stream::DEFAULT,
                              c10::Device(c10::DeviceType::CPU, 0));
  c10::Stream different_id(c10::Stream::UNSAFE,
                           c10::Device(c10::DeviceType::CPU, 0),
                           static_cast<c10::StreamId>(99));

  file << std::to_string(*cpu_stream_default == same_as_default) << " ";
  file << std::to_string(*cpu_stream_default != same_as_default) << " ";
  file << std::to_string(*cpu_stream_default == different_id) << " ";
  file << std::to_string(*cpu_stream_default != different_id) << " ";
  file << "\n";
  file.saveFile();
}

TEST_F(StreamTest, PackAndUnpack3) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "PackAndUnpack3 ";

  c10::StreamData3 p = cpu_stream_unsafe->pack3();
  c10::Stream round_trip =
      c10::Stream::unpack3(p.stream_id, p.device_index, p.device_type);

  file << std::to_string(static_cast<int64_t>(p.stream_id)) << " ";
  file << std::to_string(static_cast<int>(p.device_index)) << " ";
  file << std::to_string(static_cast<int>(p.device_type)) << " ";
  file << std::to_string(round_trip == *cpu_stream_unsafe) << " ";
  write_stream_meta(&file, round_trip);
  file << "\n";
  file.saveFile();
}

TEST_F(StreamTest, HashAndStdHash) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "HashAndStdHash ";

  auto h1 = cpu_stream_default->hash();
  auto h2 = std::hash<c10::Stream>{}(*cpu_stream_default);

  file << std::to_string(static_cast<uint64_t>(h1)) << " ";
  file << std::to_string(static_cast<uint64_t>(h2)) << " ";
  file << std::to_string(h1 == h2) << " ";
  file << "\n";
  file.saveFile();
}

TEST_F(StreamTest, OstreamOperator) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "OstreamOperator ";

  std::ostringstream oss;
  oss << *cpu_stream_unsafe;

  file << oss.str() << " ";
  file << "\n";
  file.saveFile();
}

TEST_F(StreamTest, WaitTemplateAPI) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "WaitTemplateAPI ";

  struct DummyEvent {
    mutable bool called{false};
    mutable c10::StreamId last_id{0};
    void block(const c10::Stream& s) const {
      called = true;
      last_id = s.id();
    }
  } event;

  cpu_stream_unsafe->wait(event);

  file << std::to_string(event.called) << " ";
  file << std::to_string(static_cast<int64_t>(event.last_id)) << " ";
  file << "\n";
  file.saveFile();
}

TEST_F(StreamTest, QueryAndSynchronizeCPU) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "QueryAndSynchronizeCPU ";

  bool queried = false;
  bool sync_ok = true;
  try {
    queried = cpu_stream_default->query();
    cpu_stream_default->synchronize();
  } catch (const std::exception&) {
    sync_ok = false;
  }

  file << std::to_string(queried) << " ";
  file << std::to_string(sync_ok) << " ";
  file << "\n";
  file.saveFile();
}

TEST_F(StreamTest, NativeHandleCPU) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "NativeHandleCPU ";

#if USE_PADDLE_API
  try {
    (void)cpu_stream_default->native_handle();
    file << "ok ";
  } catch (const std::exception& e) {
    // Output the exception message for comparison with Torch
    std::string msg = e.what();
    // Check if it contains "not supported" to match Torch behavior
    if (msg.find("not supported") != std::string::npos ||
        msg.find("not_supported") != std::string::npos) {
      file << "not_supported ";
    } else {
      file << "exception: " << msg;
    }
  }
#else
  file << "not_supported ";
#endif

  file << "\n";
  file.saveFile();
}

TEST_F(StreamTest, CudaQuerySynchronizeAndNativeHandle) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "CudaQuerySynchronizeAndNativeHandle ";

  if (!torch::cuda::is_available()) {
    file << "no_cuda ";
    file << "\n";
    file.saveFile();
    return;
  }

  try {
    c10::Stream stream = c10::cuda::getCurrentCUDAStream(0);
#if USE_PADDLE_API
    (void)stream.native_handle();
#endif
    file << std::to_string(stream.query()) << " ";
    stream.synchronize();
    file << "sync_ok ";
    write_stream_meta(&file, stream);
  } catch (const std::exception&) {
    file << "exception ";
  }

  file << "\n";
  file.saveFile();
}

}  // namespace test
}  // namespace at
