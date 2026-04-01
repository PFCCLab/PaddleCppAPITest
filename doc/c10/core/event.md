## Event.h 头文件 API 兼容性

> 2026-04-01 编制：Paddle compat 层为 CUDA/HIP 后端实现的事件机制。Paddle 版本提供直接的 CUDA 事件存储与管理，包括事件池优化；PyTorch 版本使用后端无关的 pimpl 模式（`impl::InlineEvent<impl::VirtualGuardImpl>`）。

对比文件：
- `/home/may/Paddle/paddle/phi/api/include/compat/c10/core/Event.h`
- `/home/may/torch/c10/core/Event.h`

状态说明：
- `✅` 已实现（接口存在且签名/语义基本一致）
- `🔧` 部分兼容（接口存在，但签名或实现语义有差异）
- `❌` 未实现（PyTorch 有，Paddle compat 头文件无）
- `🟦` 扩展（Paddle 特有功能）

---

## API 对比表

### 构造与析构

| torch API | paddle API 兼容性 | 测试用例状态 | 优先级 | 备注 |
|-----------|------------------|------------|-------|------|
| `Event()` (已删除) | ✅ | - [x] | P0 | 两者都禁用默认构造，要求手动传入 DeviceType |
| `Event(DeviceType, EventFlag)` | ✅ | - [x] | P0 | 签名与实现一致；Paddle 在 CUDA 下初始化事件池 |
| `Event(const Event&)` (删除) | ✅ | - [x] | P1 | 两者禁止拷贝构造 |
| `Event& operator=(const Event&)` (删除) | ✅ | - [x] | P1 | 两者禁止拷贝赋值 |
| `Event(Event&&)` | ✅ | - [x] | P1 | 移动语义一致 |
| `Event& operator=(Event&&)` | ✅ | - [x] | P1 | 移动语义一致 |
| `~Event()` | ✅ | - [x] | P1 | Paddle 版本依赖事件池清理，PyTorch 依赖 impl 析构 |

### 访问器（Getters）

| torch API | paddle API 兼容性 | 测试用例状态 | 优先级 | 备注 |
|-----------|------------------|------------|-------|------|
| `Device device() const noexcept` | ✅ | - [x] | P0 | 都返回 `Device(device_type_, device_index_)` |
| `DeviceType device_type() const noexcept` | ✅ | - [x] | P0 | 直接返回 `device_type_` |
| `DeviceIndex device_index() const noexcept` | ✅ | - [x] | P0 | Paddle 初始化为 -1，record() 时更新 |
| `EventFlag flag() const noexcept` | ✅ | - [x] | P1 | 都返回 `flag_` |
| `bool was_marked_for_recording() const noexcept` | ✅ | - [x] | P0 | 状态标志一致 |

### 事件记录与同步

| torch API | paddle API 兼容性 | 测试用例状态 | 优先级 | 备注 |
|-----------|------------------|------------|-------|------|
| `void recordOnce(const Stream&)` | ✅ | - [x] | P1 | 两者都调用 `record()` 当且仅当从未记录过 |
| `void record(const Stream&)` | 🔧 | - [ ] | P0 | **差异**：Paddle 在 CUDA 下使用 `static_cast<cudaStream_t>(stream.native_handle())`，但 `native_handle()` 返回 `void*`；PyTorch 通过 impl 动态分发。见关键差异 1。 |
| `void record(const c10::cuda::CUDAStream&)` | 🟦 | - [x] | P1 | Paddle 扩展：额外提供 CUDA-specific 重载，调用 `record(stream.unwrap())` |
| `void block(const Stream&) const` | 🔧 | - [ ] | P0 | **差异**：同 `record()`，Paddle 在 CUDA 下使用 `static_cast<cudaStream_t>(stream.native_handle())`；PyTorch 通过 impl 动态分发。见关键差异 1。 |
| `bool query() const` | ✅ | - [x] | P0 | 两者都查询最近版本的记录状态 |
| `void synchronize() const` | ✅ | - [x] | P1 | Paddle 在 CUDA 下调用 `cudaEventSynchronize()`；PyTorch 通过 impl 分发 |

### 时间与标识

| torch API | paddle API 兼容性 | 测试用例状态 | 优先级 | 备注 |
|-----------|------------------|------------|-------|------|
| `double elapsedTime(const Event&) const` | 🔧 | - [ ] | P3 | **差异**：Paddle 总是返回 `0.0`（未实现）；PyTorch 通过 impl 计算实际耗时 |
| `void* eventId() const` | ✅ | - [x] | P1 | Paddle 返回 `cuda_event_` 或 `nullptr`；PyTorch 通过 impl 返回后端标识 |

### Paddle 特有成员

| Paddle API | 类型 | 状态 | 备注 |
|-----------|------|------|------|
| `cudaEvent_t cuda_event() const` | 方法 | 🟦 | Paddle 专有，返回原始 CUDA 事件句柄；用于直接 CUDA 操作 |
| `EventPool` | 类 | 🟦 | Paddle 专有事件池实现，支持事件创建、查询、回收；singleton 模式 |

---

## 兼容性统计

| 状态 | 数量 |
|---|---|
| ✅ 已实现 | 16 |
| 🔧 部分兼容 | 3 |
| 🟦 扩展 | 3 |

---

## 关键差异说明

#### 1. Stream 参数处理

**问题**：Paddle 的 `record(const Stream&)` 与 `block(const Stream&)` 使用：
```cpp
static_cast<cudaStream_t>(stream.native_handle())
```

但 `stream.native_handle()` 返回 `void*` 指针。虽然该转换在大多数实现中有效，但存在隐藏的类型安全风险。

**PyTorch 方案**：通过 `impl::InlineEvent` 的动态分发，避免直接类型转换。

**建议**：如果 `Stream` 有 `stream_id()` 或类似方法，应优先使用；或添加显式的转换方法。

#### 2. elapsedTime() 未实现

**差异**：Paddle 总是返回 `0.0`（未实现），PyTorch 通过 impl 调用 CUDA 的 `cudaEventElapsedTime()` 等后端接口。若需支持时间计算，应调用底层后端 API。

#### 3. CUDA 事件池（Paddle 特有）

Paddle 实现了 `EventPool` 来优化事件创建/销毁，支持预分配、查询与复用。PyTorch 不提供此机制，性能优势为减少频繁 cudaEventCreate/Destroy 开销。

#### 4. CUDA Stream 重载（Paddle 扩展）

Paddle 提供额外重载 `void record(const c10::cuda::CUDAStream&)` 调用 `record(stream.unwrap())`，在 CUDA-only 场景中提供便利。

---

## 备注
1. **优先级说明**：
   - P0: 核心功能，必须支持
   - P1: 常用功能，高优先级
   - P2: 进阶功能，中优先级
   - P3: 边缘功能，低优先级

2. **对比范围说明**：
   - 本文档对比声明与实现（Paddle 大部分内联在头文件中，包括 EventPool）
   - PyTorch 声明在头文件，实现依赖 `impl::InlineEvent<impl::VirtualGuardImpl>`

3. **编译依赖**：
   - Paddle：需要 `PADDLE_WITH_CUDA` 宏；CUDA 操作受条件编译保护
   - PyTorch：无条件编译，后端由 impl 在运行时选择

4. **线程安全**：
   - Paddle Event 本身不保证线程安全；EventPool 使用 `std::mutex` 保护
   - PyTorch Event 记录中提到"not thread-safe"

5. **实现机制**：
   - Paddle：直接存储 `cudaEvent_t`，通过事件池管理
   - PyTorch：pimpl 模式，通过 `impl::InlineEvent` 与 `impl::VirtualGuardImpl` 支持多后端
