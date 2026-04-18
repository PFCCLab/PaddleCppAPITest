## CUDAStream.h 头文件 API 兼容性

对比文件：
- `/home/may/Paddle/paddle/phi/api/include/compat/c10/cuda/CUDAStream.h`
- `/home/may/pytorch/c10/cuda/CUDAStream.h`

状态说明：
- `✅` 已实现（接口存在且签名/语义基本一致）
- `🔧` 部分兼容（接口存在，但签名或实现语义有差异）
- `❌` 未实现（PyTorch 有，Paddle compat 头文件无）

---

### 常量与标签类型

| torch API | paddle API 兼容性 | 测试用例状态 | 优先级 | 备注 |
|-----------|------------------|------------|-------|------|
| `using StreamId = int64_t` | ✅ | - [ ] | P1 | 类型别名一致 |
| `max_compile_time_stream_priorities` | ✅ | - [ ] | P2 | 常量值同为 `4` |
| `CUDAStream::Unchecked` / `CUDAStream::UNCHECKED` | ✅ | - [x] | P0 | 已补齐无检查构造标签，`CUDATest2.CUDAStreamRoundTrip` 覆盖 |

---

### 构造、转换与比较

| torch API | paddle API 兼容性 | 测试用例状态 | 优先级 | 备注 |
|-----------|------------------|------------|-------|------|
| `CUDAStream(Stream)` | ✅ | - [x] | P0 | 已实现，构造时校验 `Stream` 的 `device_type()` 为 `CUDA` |
| `CUDAStream(Unchecked, Stream)` | ✅ | - [x] | P0 | 已实现，`CUDATest2.CUDAStreamRoundTrip` 覆盖 |
| `operator==(const CUDAStream&)` | ✅ | - [x] | P0 | 基于 `unwrap()` 比较，`CUDATest2.CUDAStreamRoundTrip` 覆盖 |
| `operator!=(const CUDAStream&)` | ✅ | - [x] | P0 | 已实现，`CUDATest2.CUDAStreamPoolAndCurrent` 覆盖 |
| `operator cudaStream_t()` | ✅ | - [x] | P0 | 已实现，`static_cast<cudaStream_t>(stream)` 与 `stream()` 一致 |
| `operator Stream()` | ✅ | - [x] | P0 | 已实现，语义与 PyTorch 一致 |

---

### 访问、同步与打包

| torch API | paddle API 兼容性 | 测试用例状态 | 优先级 | 备注 |
|-----------|------------------|------------|-------|------|
| `id()` | ✅ | - [x] | P0 | 已实现，`CUDATest2.CUDAStreamRoundTrip` 覆盖 |
| `device_type()` | ✅ | - [x] | P0 | 固定返回 `DeviceType::CUDA` |
| `device_index()` | ✅ | - [x] | P0 | 已实现，`CUDATest2.CUDAStreamRoundTrip` 覆盖 |
| `device()` | ✅ | - [ ] | P1 | 已实现，返回 `Device(DeviceType::CUDA, device_index())` |
| `stream()` | ✅ | - [x] | P0 | 已实现，通过 `StreamId` 反解 `cudaStream_t` |
| `unwrap()` | ✅ | - [x] | P0 | 已实现，直接返回底层 `c10::Stream` |
| `query()` | ✅ | - [x] | P1 | 已实现，委托 `c10::Stream::query()` |
| `synchronize()` | ✅ | - [x] | P1 | 已实现，委托 `c10::Stream::synchronize()` |
| `priority()` | ✅ | - [x] | P1 | 已实现，切换到 stream 所在设备后调用 `cudaStreamGetPriority` |
| `priority_range()` | 🔧 | - [x] | P1 | CUDA 路径与 PyTorch 一致；HIP 路径未像 PyTorch 那样将 `least_priority` 规范化为 `0` |
| `pack3()` | ✅ | - [x] | P1 | 已实现，直接复用 `c10::Stream::pack3()` |
| `unpack3(StreamId, DeviceIndex, DeviceType)` | ✅ | - [x] | P1 | 已实现，直接复用 `c10::Stream::unpack3()` |

---

### 全局辅助函数与标准库适配

| torch API | paddle API 兼容性 | 测试用例状态 | 优先级 | 备注 |
|-----------|------------------|------------|-------|------|
| `getStreamFromPool(const bool isHighPriority = false, DeviceIndex device_index = -1)` | ✅ | - [x] | P0 | 默认参数已对齐；`getStreamFromPool(true)` 不会再误绑到 `int` 重载，`c10_Stream_test` 覆盖 |
| `getStreamFromPool(const int priority, DeviceIndex device_index = -1)` | ✅ | - [x] | P1 | 已实现，使用 `std::clamp` 映射到最多 4 档优先级，与 PyTorch 一致 |
| `getStreamFromExternal(cudaStream_t, DeviceIndex)` | ✅ | - [x] | P1 | 已实现，通过 `make_cuda_stream()` 包装外部流 |
| `getDefaultCUDAStream(DeviceIndex device_index = -1)` | ✅ | - [x] | P0 | 已实现，返回默认 null stream（`id == 0`），`c10_Stream_test` 覆盖稳定性与不受 `setCurrentCUDAStream()` 影响 |
| `getCurrentCUDAStream(DeviceIndex device_index = -1)` | ✅ | - [x] | P0 | 已实现，保持 per-thread、per-device current stream 语义；TLS 未设置时回退到 default stream |
| `setCurrentCUDAStream(CUDAStream)` | ✅ | - [x] | P0 | 已实现，仅修改当前线程 TLS 中对应设备的 current stream |
| `operator<<(std::ostream&, const CUDAStream&)` | ✅ | - [x] | P2 | 已实现，委托到底层 `c10::Stream` 输出 |
| `std::hash<c10::cuda::CUDAStream>` | ✅ | - [x] | P2 | 已实现，委托 `std::hash<c10::Stream>` |

---

### 内部实现细节（非公开 API）

| torch 内部实现 | paddle 对应实现 | 说明 |
|---------------|----------------|------|
| `CUDAStreamForId(DeviceIndex, StreamId)` | `make_cuda_stream(cudaStream_t, DeviceIndex)` | PyTorch 内部辅助函数（位于 `anonymous namespace`），用于从 `stream_id` 构造 `CUDAStream`。Paddle 使用 `make_cuda_stream` 完成相同功能，**无需暴露为公开 API**。 |

---

### ROCm/HIP backward-compat 别名

| torch API | paddle API 兼容性 | 测试用例状态 | 优先级 | 备注 |
|-----------|------------------|------------|-------|------|
| `c10::hip::getStreamFromExternal(...)` | ❌ | - [ ] | P2 | PyTorch 在 `USE_ROCM` 下提供 using alias，Paddle 未提供 |
| `c10::hip::getStreamFromPool(...)` | ❌ | - [ ] | P2 | PyTorch 在 `USE_ROCM` 下提供对 bool/int 两个重载的 alias，Paddle 未提供 |
| `c10::hip::getDefaultHIPStream(DeviceIndex device_index = -1)` | ❌ | - [ ] | P2 | 缺失 |
| `c10::hip::getCurrentHIPStream(DeviceIndex device_index = -1)` | ❌ | - [ ] | P2 | 缺失 |
| `c10::hip::setCurrentHIPStream` | ❌ | - [ ] | P2 | 缺失 |
| `c10::hip::getStreamFromPoolMasqueradingAsCUDA(const bool isHighPriority = false, DeviceIndex device = -1)` | ❌ | - [ ] | P3 | 缺失 |
| `c10::hip::getStreamFromPoolMasqueradingAsCUDA(const int priority, DeviceIndex device = -1)` | ❌ | - [ ] | P3 | 缺失 |
| `c10::hip::getStreamFromExternalMasqueradingAsCUDA` | ❌ | - [ ] | P3 | 缺失 |
| `c10::hip::getDefaultHIPStreamMasqueradingAsCUDA(DeviceIndex device_index = -1)` | ❌ | - [ ] | P3 | 缺失 |
| `c10::hip::getCurrentHIPStreamMasqueradingAsCUDA(DeviceIndex device_index = -1)` | ❌ | - [ ] | P3 | 缺失 |
| `c10::hip::setCurrentHIPStreamMasqueradingAsCUDA` | ❌ | - [ ] | P3 | 缺失 |

---

### 兼容性统计

| 状态 | 数量 |
|---|---|
| ✅ 已实现 | 28 |
| 🔧 部分兼容 | 1 |
| ❌ 未实现 | 11 |

---

### 备注

1. **优先级说明**：
   - P0: 核心功能，必须支持
   - P1: 常用功能，高优先级
   - P2: 进阶功能，中优先级
   - P3: 边缘功能，低优先级

2. **对比范围说明**：
   - 本文档基于头文件声明与实现语义对比：
     - `paddle/phi/api/include/compat/c10/cuda/CUDAStream.h`
     - `/home/may/pytorch/c10/cuda/CUDAStream.h`
     - `/home/may/pytorch/c10/cuda/CUDAStream.cpp`
   - `getStreamFromPool(int, ...)` 的优先级分档语义需要结合 PyTorch `.cpp` 实现判断，不能只看声明。

3. **主要差异说明**：
   - `getStreamFromPool(int, ...)` 现已与 PyTorch 对齐，使用独立 stream pool 实现（低/高优先级各 32 条），通过 `std::call_once` 懒初始化。
   - `getCurrentCUDAStream()` 现已与 PyTorch 对齐，使用 thread-local `std::vector<cudaStream_t>` 实现 per-thread current stream 语义，不再直接依赖 phi 层。
   - `priority_range()` 在 CUDA 路径上可视为对齐；若构建为 HIP，PyTorch 会把 `least_priority` 规范化为 `0`，Paddle 当前未做该归一化。
   - PyTorch 在 `USE_ROCM` 下还暴露 `c10::hip` backward-compat alias；Paddle 当前 compat 头文件未覆盖这组入口。
   - `make_cuda_stream(cudaStream_t, DeviceIndex)`：Paddle 提供的辅助包装函数，功能上等价于 PyTorch 内部的 `CUDAStreamForId`，**非公开 API**。
   - `at::cuda` using alias：Paddle 在该头文件尾部直接导出了 `CUDAStream`、`getCurrentCUDAStream`、`getDefaultCUDAStream`、`getStreamFromExternal`、`getStreamFromPool`、`setCurrentCUDAStream`。

5. **测试现状**：
   - `test/c10/cuda/CUDATest2.cpp` 已覆盖 `UNCHECKED`、构造/比较、转换、`query()`、`synchronize()`、`priority()`、`priority_range()`、`pack3()`、`unpack3()`、`getCurrentCUDAStream()`、`getStreamFromPool()`、`getStreamFromExternal()`、`setCurrentCUDAStream()`、`operator<<`、`std::hash`。
   - `/home/may/Paddle/test/cpp/compat/c10_Stream_test.cc` 已覆盖 `getDefaultCUDAStream()` 的 null-stream/stable 语义、`getStreamFromPool(true)` 的 bool 重载分派，以及 `setCurrentCUDAStream()` 不影响 `getDefaultCUDAStream()` 的行为。

6. **内部实现说明**：
   - PyTorch 的 `CUDAStreamForId` 是 `anonymous namespace` 中的内部辅助函数，用于从 `stream_id` 构造 `CUDAStream`。Paddle 使用 `make_cuda_stream` 完成相同功能，**无需暴露为公开 API**。
   - PyTorch 使用编译时固定大小的 `std::array`（`C10_COMPILE_TIME_MAX_GPUS`）管理 stream pool；Paddle 使用运行时动态分配的 `std::vector<std::unique_ptr<DevicePools>>`，避免硬编码最大设备数限制。
