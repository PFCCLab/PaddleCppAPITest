/*
 * =====================================================================================
 * @brief: c10::Event 跨库对齐回归测试
 *
 * 覆盖 c10::Event 的核心构造、属性访问与异常行为。
 * 原已删除的 unmatch_EventTest.cpp 中记录的历史差异（条件编译包裹、构造函数
 * 缺少 EventFlag、非 CUDA 构建下 Event 不可用）已通过对齐 Paddle compat 的
 * c10/core/Event.h 解决，本文件作为常规回归测试纳入 result_cmp。
 *
 * 说明：libtorch 的 CPU backend 不支持 events，因此 CPU 设备上调用 record()
 * 会抛出异常；本测试统一校验该异常行为。
 * =====================================================================================
 */
#include <ATen/ATen.h>
#include <c10/core/Event.h>
#include <gtest/gtest.h>

#include <string>

#include "src/file_manager.h"

extern ::paddle_api_test::ThreadSafeParam g_custom_param;

namespace at {
namespace test {

using ::paddle_api_test::FileManerger;
using ::paddle_api_test::ThreadSafeParam;

class EventCompatTest : public ::testing::Test {
 protected:
  void SetUp() override {}
};

template <typename Fn>
static bool throws_any(Fn&& fn) {
  try {
    fn();
    return false;
  } catch (...) {
    return true;
  }
}

// Event default constructor with single DeviceType argument
TEST_F(EventCompatTest, EventDefault) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();
  file << "EventDefault ";

  c10::Event event(c10::DeviceType::CPU);
  file << (event.device_type() == c10::DeviceType::CPU ? "1" : "0") << " ";
  file << event.device_index() << " ";
  file << (event.flag() == c10::EventFlag::PYTORCH_DEFAULT ? "1" : "0") << " ";
  file << (event.was_marked_for_recording() ? "1" : "0") << " ";
  file << (event.query() ? "1" : "0") << "\n";
  file.saveFile();
}

// Event constructor with DeviceType and EventFlag
TEST_F(EventCompatTest, EventWithFlag) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "EventWithFlag ";

  c10::Event event(c10::DeviceType::CPU, c10::EventFlag::BACKEND_DEFAULT);
  file << (event.device_type() == c10::DeviceType::CPU ? "1" : "0") << " ";
  file << (event.flag() == c10::EventFlag::BACKEND_DEFAULT ? "1" : "0") << "\n";
  file.saveFile();
}

// Event::record on CPU stream throws (CPU backend doesn't support events)
TEST_F(EventCompatTest, EventRecordThrows) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "EventRecordThrows ";

  c10::Event event(c10::DeviceType::CPU);
  c10::Stream stream(c10::Stream::DEFAULT, c10::Device(c10::kCPU));
  file << (throws_any([&]() { event.record(stream); }) ? "1" : "0") << "\n";
  file.saveFile();
}

// Event::recordOnce on CPU stream throws on first call
TEST_F(EventCompatTest, EventRecordOnceThrows) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "EventRecordOnceThrows ";

  c10::Event event(c10::DeviceType::CPU);
  c10::Stream stream(c10::Stream::DEFAULT, c10::Device(c10::kCPU));
  file << (throws_any([&]() { event.recordOnce(stream); }) ? "1" : "0") << "\n";
  file.saveFile();
}

// Event move semantics
TEST_F(EventCompatTest, EventMove) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "EventMove ";

  c10::Event event1(c10::DeviceType::CPU);
  c10::Event event2(std::move(event1));
  file << (event2.device_type() == c10::DeviceType::CPU ? "1" : "0") << " ";
  file << event2.device_index() << " ";
  file << (event2.query() ? "1" : "0") << "\n";
  file.saveFile();
}

// Event device() getter
TEST_F(EventCompatTest, EventDevice) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "EventDevice ";

  c10::Event event(c10::DeviceType::CPU);
  auto dev = event.device();
  file << (dev.type() == c10::DeviceType::CPU ? "1" : "0") << " ";
  file << dev.index() << "\n";
  file.saveFile();
}

}  // namespace test
}  // namespace at
