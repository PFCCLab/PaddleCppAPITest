#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <future>
#include <thread>

#include "src/file_manager.h"

extern paddle_api_test::ThreadSafeParam g_custom_param;

class PhiKernelStreamDeadlockTest : public ::testing::Test {};

namespace {

void CUDART_CB sleep_200ms_callback(void* user_data) {
  auto* flag = static_cast<std::atomic<bool>*>(user_data);
  std::this_thread::sleep_for(std::chrono::milliseconds(200));
  flag->store(true, std::memory_order_release);
}

}  // namespace

namespace {

// 构建事件链，将 |pool| 阻塞约 200ms。
// 返回 enq_stream，以便调用者清理。
cudaStream_t build_blocked_stream(c10::cuda::CUDAStream pool,
                                  cudaEvent_t* out_event_start,
                                  cudaEvent_t* out_event_end,
                                  std::atomic<bool>* out_callback_done) {
  cudaStream_t enq_stream = nullptr;
  cudaStreamCreateWithFlags(&enq_stream, cudaStreamNonBlocking);
  cudaEventCreateWithFlags(out_event_start, cudaEventDisableTiming);
  cudaEventCreateWithFlags(out_event_end, cudaEventDisableTiming);

  cudaEventRecord(*out_event_start, pool.stream());
  cudaStreamWaitEvent(enq_stream, *out_event_start, 0);
  cudaLaunchHostFunc(enq_stream, sleep_200ms_callback, out_callback_done);
  cudaEventRecord(*out_event_end, enq_stream);
  cudaStreamWaitEvent(pool.stream(), *out_event_end, 0);

  return enq_stream;
}

void wait_for_callback(std::atomic<bool>* done) {
  auto start = std::chrono::steady_clock::now();
  while (!done->load(std::memory_order_acquire)) {
    if (std::chrono::steady_clock::now() - start > std::chrono::seconds(5))
      break;
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
}

}  // namespace

// 对照测试 1：worker 只 sync 其 c10 current stream。
// 预期：两边都很快完成，因为 worker 的 current stream
// 不是被阻塞的 pool stream（c10 thread-local 隔离生效）。
TEST_F(PhiKernelStreamDeadlockTest, StreamSyncOnlyDoesNotDeadlock) {
  if (!at::cuda::is_available()) {
    return;
  }

  auto file_name = g_custom_param.get();
  paddle_api_test::FileManerger file(file_name);
  file.createFile();

  auto original = c10::cuda::getCurrentCUDAStream();
  auto pool = c10::cuda::getStreamFromPool(/*isHighPriority=*/false);
  c10::cuda::setCurrentCUDAStream(pool);

  cudaEvent_t event_start = nullptr;
  cudaEvent_t event_end = nullptr;
  std::atomic<bool> callback_done{false};
  cudaStream_t enq_stream =
      build_blocked_stream(pool, &event_start, &event_end, &callback_done);

  std::packaged_task<void()> task([]() {
    auto cur = c10::cuda::getCurrentCUDAStream();
    cudaStreamSynchronize(cur.stream());
  });
  auto future = task.get_future();
  std::thread worker(std::move(task));

  auto status = future.wait_for(std::chrono::milliseconds(50));
  file << std::to_string(status == std::future_status::ready ? 1 : 0) << "\n";

  wait_for_callback(&callback_done);
  future.wait();
  worker.join();

  cudaEventDestroy(event_end);
  cudaEventDestroy(event_start);
  cudaStreamDestroy(enq_stream);
  c10::cuda::setCurrentCUDAStream(original);

  file.saveFile();
}

// 对照测试 2：worker 在其 c10 current stream 上创建 tensor（内存分配）。
// 预期：两边都可能阻塞，因为 cudaMalloc 会同步所有 device streams。
TEST_F(PhiKernelStreamDeadlockTest, TensorAllocationOnCurrentStream) {
  if (!at::cuda::is_available()) {
    return;
  }

  auto file_name = g_custom_param.get();
  paddle_api_test::FileManerger file(file_name);
  file.openAppend();

  auto original = c10::cuda::getCurrentCUDAStream();
  auto pool = c10::cuda::getStreamFromPool(/*isHighPriority=*/false);
  c10::cuda::setCurrentCUDAStream(pool);

  cudaEvent_t event_start = nullptr;
  cudaEvent_t event_end = nullptr;
  std::atomic<bool> callback_done{false};
  cudaStream_t enq_stream =
      build_blocked_stream(pool, &event_start, &event_end, &callback_done);

  std::packaged_task<void()> task([]() {
    auto x =
        at::ones({4}, at::TensorOptions().dtype(at::kInt).device(at::kCUDA, 0));
    (void)x;
  });
  auto future = task.get_future();
  std::thread worker(std::move(task));

  auto status = future.wait_for(std::chrono::milliseconds(50));
  file << std::to_string(status == std::future_status::ready ? 1 : 0) << "\n";

  wait_for_callback(&callback_done);
  future.wait();
  worker.join();

  cudaEventDestroy(event_end);
  cudaEventDestroy(event_start);
  cudaStreamDestroy(enq_stream);
  c10::cuda::setCurrentCUDAStream(original);

  file.saveFile();
}

// 复现 temp_origin 死锁：主线程阻塞 current stream，
// worker 线程对预先分配的 at::Tensor 执行索引填充。
//
// Paddle 编译版本：at::Tensor::fill_() 底层调用 paddle::experimental::fill_()
// （phi kernel），phi kernel 使用 GPUContext 全局 stream。由于
// setCurrentCUDAStream(pool) 同时把 GPUContext stream 设为 pool，而 pool
// 被事件链阻塞，因此 worker 的 phi kernel 挂起。
//   => worker 在 50ms 内未完成 => 输出 0。
//
// Torch 编译版本：at::Tensor::fill_() 底层走 ATen kernel，使用 c10
// thread-local stream。worker 线程未显式设置 current stream，fallback 到
// default stream（未被阻塞）。
//   => worker 在 50ms 内完成 => 输出 1。
TEST_F(PhiKernelStreamDeadlockTest, WorkerTensorOpDeadlocksOnBlockedStream) {
  if (!at::cuda::is_available()) {
    return;
  }

  auto file_name = g_custom_param.get();
  paddle_api_test::FileManerger file(file_name);
  file.openAppend();

  auto original = c10::cuda::getCurrentCUDAStream();
  auto pool = c10::cuda::getStreamFromPool(/*isHighPriority=*/false);
  c10::cuda::setCurrentCUDAStream(pool);

  // 在主线程中预先分配 tensor，避免 worker 中 cudaMalloc 的同步干扰。
  auto x =
      at::ones({4}, at::TensorOptions().dtype(at::kInt).device(at::kCUDA, 0));

  cudaEvent_t event_start = nullptr;
  cudaEvent_t event_end = nullptr;
  std::atomic<bool> callback_done{false};
  cudaStream_t enq_stream =
      build_blocked_stream(pool, &event_start, &event_end, &callback_done);

  // Worker 线程：只做索引填充（不涉及内存分配）。
  std::packaged_task<void()> task([&x]() { x[0].fill_(0); });
  auto future = task.get_future();
  std::thread worker(std::move(task));

  // 检查 worker 是否在 50ms 内完成。
  auto status = future.wait_for(std::chrono::milliseconds(50));
  file << std::to_string(status == std::future_status::ready ? 1 : 0) << "\n";

  // 解锁：等待 callback 完成，使 stream 恢复。
  wait_for_callback(&callback_done);
  future.wait();
  worker.join();

  cudaEventDestroy(event_end);
  cudaEventDestroy(event_start);
  cudaStreamDestroy(enq_stream);

  c10::cuda::setCurrentCUDAStream(original);

  file.saveFile();
}
