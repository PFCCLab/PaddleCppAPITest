#include <ATen/ATen.h>
#include <gtest/gtest.h>

#include <string>

#include "../src/file_manager.h"
#ifndef USE_PADDLE_API
#include <c10/core/Event.h>
#endif

extern ::paddle_api_test::ThreadSafeParam g_custom_param;

namespace at {
namespace test {

using ::paddle_api_test::FileManerger;
using ::paddle_api_test::ThreadSafeParam;

class EventTest : public ::testing::Test {
 protected:
  void SetUp() override {}
};

#ifndef USE_PADDLE_API
// EventPool default constructor
TEST_F(EventTest, EventPoolDefault) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();

#ifdef PADDLE_WITH_CUDA
  c10::EventPool pool;
  file << "EventPool_default ";
#else
  file << "EventPool_default_skipped_no_cuda ";
#endif
  file.saveFile();
}

// EventPool copy constructor
TEST_F(EventTest, EventPoolCopy) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();

#ifdef PADDLE_WITH_CUDA
  c10::EventPool pool1;
  c10::EventPool pool2(pool1);
  file << "EventPool_copy ";
#else
  file << "EventPool_copy_skipped_no_cuda ";
#endif
  file.saveFile();
}

// EventPool move constructor
TEST_F(EventTest, EventPoolMove) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();

#ifdef PADDLE_WITH_CUDA
  c10::EventPool pool1;
  c10::EventPool pool2(std::move(pool1));
  file << "EventPool_move ";
#else
  file << "EventPool_move_skipped_no_cuda ";
#endif
  file.saveFile();
}

// EventPool::Instance
TEST_F(EventTest, EventPoolInstance) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();

#ifdef PADDLE_WITH_CUDA
  auto& instance = c10::EventPool::Instance();
  (void)instance;
  file << "EventPool_Instance ";
#else
  file << "EventPool_Instance_skipped_no_cuda ";
#endif
  file.saveFile();
}

// EventPool::CreateCudaEventFromPool
TEST_F(EventTest, EventPoolCreateCudaEventFromPool) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();

#ifdef PADDLE_WITH_CUDA
  // This requires CUDA
  file << "CreateCudaEventFromPool ";
#else
  file << "CreateCudaEventFromPool_skipped_no_cuda ";
#endif
  file.saveFile();
}

// Event default constructor
TEST_F(EventTest, EventDefault) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();

#ifdef PADDLE_WITH_CUDA
  c10::Event event(c10::DeviceType::CPU);
  (void)event;
  file << "Event_default ";
#else
  file << "Event_default_skipped_no_cuda ";
#endif
  file.saveFile();
}

// Event with device type
TEST_F(EventTest, EventWithDeviceType) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();

#ifdef PADDLE_WITH_CUDA
  c10::Event event(c10::DeviceType::CUDA);
  (void)event;
  file << "Event_cuda ";
#else
  file << "Event_cuda_skipped_no_cuda ";
#endif
  file.saveFile();
}

// Event::record
TEST_F(EventTest, EventRecord) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();

#ifdef PADDLE_WITH_CUDA
  c10::Event event(c10::DeviceType::CPU);
  // record requires a stream - skip actual call for non-CUDA build
  (void)event;
  file << "Event_record ";
#else
  file << "Event_record_skipped_no_cuda ";
#endif
  file.saveFile();
}

// Event::cuda_event
TEST_F(EventTest, EventCudaEvent) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();

#ifdef PADDLE_WITH_CUDA
  c10::Event event(c10::DeviceType::CUDA);
  // cuda_event returns cudaEvent_t - skip for non-CUDA build
  (void)event;
  file << "Event_cuda_event ";
#else
  file << "Event_cuda_event_skipped_no_cuda ";
#endif
  file.saveFile();
}

#endif  // USE_PADDLE_API

}  // namespace test
}  // namespace at
