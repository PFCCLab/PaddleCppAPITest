# Paddle C++ 兼容层架构图

本文档描述 Paddle 对 PyTorch C++ API（`c10::Stream`、`c10::cuda::CUDAStream`、`at::Tensor::record_stream`）的兼容层架构，包括各层的映射关系、Paddle 特有实现，以及与 PyTorch 的语义差异。

---

## 整体架构图

```mermaid
graph TD
    subgraph "调用方（用户代码）"
        U1["at::Tensor::record_stream(at::Stream s)"]
        U2["at::Tensor::record_stream(at::cuda::CUDAStream s)"]
        U3["c10::cuda::getCurrentCUDAStream()"]
        U4["c10::cuda::setCurrentCUDAStream(stream)"]
        U5["c10::cuda::getDefaultCUDAStream()"]
        U6["c10::cuda::getStreamFromPool(isHighPriority)"]
    end

    subgraph "兼容层核心类型"
        S["c10::Stream\n(device + StreamId)\nStream.h / Stream.cpp"]
        CS["c10::cuda::CUDAStream\n(CUDA 专用包装)\nCUDAStream.h"]
    end

    subgraph "Stream ID 编码"
        ENC["StreamId = reinterpret_cast&lt;intptr_t&gt;(cudaStream_t)\nid=0 ↔ cudaStreamDefault（null stream）\nid≠0 ↔ 实际 CUDA stream 句柄"]
    end

    subgraph "线程局部状态（TLS）"
        TLS["detail::TLSStreamState\n  cudaStream_t streams[kMaxDevices=64]\n  bool has_stream[kMaxDevices]\n线程本地，不影响其他线程"]
    end

    subgraph "流池（Stream Pool）"
        POOL["detail::StreamPoolState[kMaxDevices]\n  low_priority[32] / high_priority[32]\n  懒初始化（std::call_once）\n  round-robin 原子计数器分配"]
    end

    subgraph "Paddle phi 层"
        PHI["phi::backends::gpu::GetCurrentDeviceId()\nphi::GPUPlace(device_index)\nphi::CUDAStream / cudaStream_t"]
        MEM["paddle::memory::RecordStream(holder, stream)"]
    end

    U1 --> S
    U2 --> CS
    CS --> |"unwrap() → c10::Stream"| S
    S --> |"native_handle()\nreinterpret_cast"| PHI
    PHI --> MEM

    U3 --> TLS
    TLS --> |"has_stream=true: 返回 TLS 中的流"| CS
    TLS --> |"has_stream=false: 回退"| PHI

    U4 --> TLS

    U5 --> |"始终返回 id=0\n(cudaStreamDefault)"| CS

    U6 --> POOL
    POOL --> CS

    CS --> ENC
    S --> ENC
```

---

## 关键语义说明

### `getCurrentCUDAStream()` vs `getDefaultCUDAStream()`

| 函数 | 语义 | 返回值 |
|------|------|--------|
| `getCurrentCUDAStream(dev)` | per-thread per-device 当前流 | TLS 中的流，或 Paddle phi 默认流（若未设置） |
| `getDefaultCUDAStream(dev)` | 设备固定默认流 | 始终为 null stream（id=0，`cudaStreamDefault`） |

这与 PyTorch 语义完全一致：`getCurrentCUDAStream()` 可变（通过 `setCurrentCUDAStream()` 修改），`getDefaultCUDAStream()` 固定不变。

### Stream ID 编码方式

Paddle 兼容层将 `cudaStream_t`（一个指针）直接通过 `reinterpret_cast<intptr_t>` 存储在 `StreamId`（`int64_t`）中：

```
StreamId id = static_cast<StreamId>(reinterpret_cast<intptr_t>(cudaStream_t));
cudaStream_t handle = reinterpret_cast<cudaStream_t>(static_cast<intptr_t>(id));
```

因此：
- `id == 0` ↔ `cudaStreamDefault`（null stream）
- `id != 0` ↔ 实际分配的 CUDA stream 句柄

---

## Paddle 特有实现及必要性说明

以下实现在 PyTorch 中不存在，是 Paddle 特有的适配方案：

### 1. 通过 `phi` 层获取设备和流

**PyTorch 做法**：有独立的 CUDA 设备跟踪机制（`c10::cuda::current_device()` 内部维护自己的状态）。

**Paddle 做法**：通过 `phi::backends::gpu::GetCurrentDeviceId()` 和 `phi::GPUPlace` 获取当前设备，通过 `paddle::GetCurrentCUDAStream(phi::GPUPlace)` 获取 Paddle 侧管理的 phi 流。

**必要性**：Paddle 的设备管理和流管理由 `phi` 层统一负责，兼容层必须调用 `phi` 层接口才能与 Paddle 的执行引擎协同。直接访问底层 CUDA API 会绕过 Paddle 的流生命周期管理。

### 2. TLS 使用静态数组而非动态结构

**PyTorch 做法**：内部有完整的 `StreamGuard`/`CUDAGuard` 基础设施，stream 状态存储在更复杂的 per-thread 结构中。

**Paddle 做法**：使用固定大小静态数组（`kMaxDevices=64`）+ `has_stream[device_index]` 标志位，不引入额外依赖。

**必要性**：减少对 Paddle 内部基础设施的侵入性依赖，保持兼容层轻量独立。代价是设备数上限为 64（覆盖当前所有 CUDA 硬件）。

### 3. 流池按设备懒初始化

**PyTorch 做法**：有全局 `initCUDAStreamsOnce()` 一次性初始化所有设备的流池。

**Paddle 做法**：每个设备的流池通过 `std::call_once` 在首次使用时独立初始化。

**必要性**：Paddle 的多设备初始化是按需触发的，兼容层必须与此模型对齐，不能假设所有设备在程序启动时均已初始化。

---

## 向后兼容接口（待移除）

以下接口为过渡期兼容性保留，标有 `TODO` 注释，待下游（DeepEP/PaddleFleet）完成迁移后移除：

| 接口 | 位置 | 原因 |
|------|------|------|
| `CUDAStream::raw_stream()` | `CUDAStream.h` | DeepEP 旧接口，等价于 `stream()` |
| `Tensor::record_stream(cudaStream_t)` | `ATen/ops/record_stream.h` | DeepEP 旧接口，等价于接受 `at::Stream` 版本 |
| `Event::raw_event()` | `c10/core/Event.h` | 同上 |
