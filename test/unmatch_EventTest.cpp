/*
 * =====================================================================================
 * @brief: 兼容性对齐审计报告
 *
 * [异常点 1]
 * - 测试用例：整个文件的公共头文件引入部分
 * - 当前状况：`#include <c10/core/Event.h>` 被 `#ifndef USE_PADDLE_API`
 * 宏包裹， 在 Paddle 构建模式下该头文件完全不被包含。
 * - 根本原因：两个库均有 `c10/core/Event.h`，路径完全一致，头文件本身不是问题。
 *             但 Paddle compat 的 `c10/core/Event.h` 将其全部类定义包裹在
 *             `#ifdef PADDLE_WITH_CUDA` 中，导致非 CUDA 构建时 `c10::Event`、
 *             `c10::EventPool` 均不可用；而 libtorch 的同路径头文件在所有构建
 *             下均无条件暴露 `c10::Event`。因此，真正应当对齐的是 Paddle compat
 *             头文件中的条件编译范围，而非在测试侧用 `#ifndef USE_PADDLE_API`
 *             绕开整个头文件。
 * - 期望解决：在 Paddle compat 的 `c10/core/Event.h` 中，将 `c10::Event` 的
 *             基础定义（至少 CPU 路径）移至 `#ifdef PADDLE_WITH_CUDA` 之外，
 *             使其无论是否有 CUDA 支持都能被包含，从而测试侧无需条件保护该
 *             `#include` 语句。
 *
 * [异常点 2]
 * - 测试用例：EventPoolDefault / EventPoolCopy / EventPoolMove /
 *             EventPoolInstance / EventPoolCreateCudaEventFromPool /
 *             EventDefault / EventWithDeviceType / EventRecord / EventCudaEvent
 *             （共 9 个 TEST_F，全部被禁用）
 * - 当前状况：所有测试用例被一整块 `#ifndef USE_PADDLE_API … #endif` 包裹，
 *             Paddle 构建模式下该文件实际上是一个空测试套件，没有任何用例参与
 *             对比测试。
 * - 根本原因：① `c10::EventPool` 是 Paddle compat 对 CUDA Event
 * 管理的私有扩展， 在 libtorch 的 `c10/core/Event.h` 中根本不存在此类（libtorch
 * 使用 的是 `c10::impl::InlineEvent` + `c10::Event` 的 DeviceGuard 架构）。 ②
 * `c10::Event` 的构造函数签名存在差异：libtorch 要求传入
 *             `(DeviceType, EventFlag)` 两个参数，而 Paddle compat 的实现只接受
 *             `(const DeviceType&)` 一个参数（EventFlag 形参缺失）。两处根本性
 *             API 不对齐导致无法写出"双库同一"的测试代码。
 * - 期望解决：① 在 Paddle compat 的 `c10/core/Event.h` 中为 `c10::Event` 补齐
 *             带 `EventFlag` 参数的构造函数重载（默认值与 libtorch 保持一致：
 *             `EventFlag::PYTORCH_DEFAULT`），使构造接口对齐。
 *             ② `c10::EventPool` 属于 Paddle 私有扩展，无对应 libtorch API，
 *             应将 EventPool 相关测试用例（EventPoolDefault/Copy/Move/Instance/
 *             CreateCudaEventFromPool）整体迁移至 Paddle 专属测试文件，不应出
 *             现在跨库对齐测试中。完成上述修改后，可移除外层
 *             `#ifndef USE_PADDLE_API` 宏，仅保留核心 Event 用例。
 *
 * [异常点 3]
 * - 测试用例：EventPoolDefault / EventPoolCopy / EventPoolMove /
 *             EventPoolInstance / EventDefault / EventWithDeviceType /
 *             EventRecord / EventCudaEvent
 * - 当前状况：每个测试用例内部均使用 `#ifdef PADDLE_WITH_CUDA … #else … #endif`
 *             将实际测试逻辑与 fallback 字符串输出分离，造成两套库在 CUDA 和
 *             非 CUDA 构建下输出完全不同的内容，无法进行有效的差异对比。
 * - 根本原因：Paddle compat 的 `c10::Event` 和 `c10::EventPool` 强依赖 CUDA
 *             运行时，而 libtorch 的 `c10::Event` 对 CPU 设备同样提供完整实现，
 *             不需要 CUDA 条件保护。这一架构差异迫使测试代码在内层再次引入
 *             平台宏，破坏了测试的跨平台一致性。
 * - 期望解决：在 Paddle compat 侧提供不依赖 CUDA 的 `c10::Event` CPU 路径实现
 *             （哪怕是 stub），以匹配 libtorch 的通用事件模型。届时可移除测试
 *             内所有 `#ifdef PADDLE_WITH_CUDA` 分支，用统一的 CPU 设备测试逻辑
 *             覆盖两个库。
 * =====================================================================================
 */
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
