## Event.h 头文件 API 兼容性

> 2026-04-02 复核：Paddle compat `c10::Event` 已按 PyTorch 当前可观察语义改成首次 `record()` 时 lazy-create，并用 `EventFlag` 控制 timing。为兼容 DeepEP / PaddleFleet 现有调用，Paddle 暂时额外保留 `record(const cudaStream_t&)` 这类非上游扩展接口。

对比文件：
- `/home/may/Paddle/paddle/phi/api/include/compat/c10/core/Event.h`
- `/home/may/pytorch/c10/core/Event.h`

状态说明：
- `✅` 已实现（接口存在且签名/语义基本一致）
- `🔧` 部分兼容（接口存在，但签名或边界语义仍有差异）
- `❌` 未实现（PyTorch 有，Paddle compat 头文件无）
- `🟦` 扩展（Paddle 特有功能或为兼容下游暂时保留的接口）

---

## API 对比表

### 构造与析构

| torch API | paddle API 兼容性 | 测试用例状态 | 优先级 | 备注 |
|-----------|------------------|------------|-------|------|
| `Event()` (已删除) | ✅ | - [x] | P0 | 两者都禁用默认构造，必须显式传入 `DeviceType` |
| `Event(DeviceType, EventFlag)` | ✅ | - [x] | P0 | 两者都只保存 `device_type/flag`，真正的 backend event 在首次 `record()` 时创建 |
| `Event(const Event&)` (删除) | ✅ | - [x] | P1 | 两者都禁止拷贝构造 |
| `Event& operator=(const Event&)` (删除) | ✅ | - [x] | P1 | 两者都禁止拷贝赋值 |
| `Event(Event&&)` | ✅ | - [x] | P1 | 两者都支持移动语义 |
| `Event& operator=(Event&&)` | ✅ | - [x] | P1 | 两者都支持移动赋值 |
| `~Event()` | ✅ | - [x] | P1 | Paddle 直接析构持有的 `cudaEvent_t`；PyTorch 通过 impl 析构，外部可观察语义一致 |

### 访问器（Getters）

| torch API | paddle API 兼容性 | 测试用例状态 | 优先级 | 备注 |
|-----------|------------------|------------|-------|------|
| `Device device() const noexcept` | ✅ | - [x] | P0 | 两者都返回 `Device(device_type_, device_index_)` |
| `DeviceType device_type() const noexcept` | ✅ | - [x] | P0 | 直接返回构造时保存的 `device_type_` |
| `DeviceIndex device_index() const noexcept` | ✅ | - [x] | P0 | 初始为 `-1`，首次成功 `record()` 后绑定到目标 stream 的 device index |
| `EventFlag flag() const noexcept` | ✅ | - [x] | P1 | 直接返回 `flag_` |
| `bool was_marked_for_recording() const noexcept` | ✅ | - [x] | P0 | 首次成功 `record()` 后置 true |

### 事件记录与同步

| torch API | paddle API 兼容性 | 测试用例状态 | 优先级 | 备注 |
|-----------|------------------|------------|-------|------|
| `void recordOnce(const Stream&)` | ✅ | - [x] | P1 | 两者都仅在从未记录过时转调 `record()` |
| `void record(const Stream&)` | ✅ | - [x] | P0 | 两者都在首次 `record()` 时 lazy-create，并约束 `device_type/device_index` 一致性 |
| `void record(const c10::cuda::CUDAStream&)` | 🟦 | - [x] | P1 | Paddle 额外提供 CUDA-specific 便利重载，转调 `record(stream.unwrap())` |
| `void block(const Stream&) const` | ✅ | - [x] | P0 | 两者都对未记录事件直接返回；已记录事件要求与事件 device 对齐 |
| `bool query() const` | ✅ | - [x] | P0 | 两者都把“未记录”视为 ready，对已记录事件查询最近版本状态 |
| `void synchronize() const` | ✅ | - [x] | P1 | 两者都对未记录事件 no-op；对已记录事件等待完成 |

### 时间与标识

| torch API | paddle API 兼容性 | 测试用例状态 | 优先级 | 备注 |
|-----------|------------------|------------|-------|------|
| `double elapsedTime(const Event&) const` | ✅ | - [x] | P1 | 两者都要求 `enable_timing=True`（即 `EventFlag::BACKEND_DEFAULT`）、两端已 record 且已完成，然后返回实际耗时 |
| `void* eventId() const` | ✅ | - [x] | P1 | 两者都返回底层 backend event 标识；未创建时返回空值 |

### Paddle 特有成员

| Paddle API | 类型 | 状态 | 备注 |
|-----------|------|------|------|
| `void record(const cudaStream_t&)` | 方法 | 🟦 | 为兼容现有 DeepEP / PaddleFleet 调用暂时保留；首次调用时用当前 device 作为 stream device |
| `cudaEvent_t cuda_event() const` | 方法 | 🟦 | Paddle 专有调试/兼容接口，返回原始 CUDA 事件句柄 |

---

## 兼容性统计

| 状态 | 数量 |
|---|---|
| ✅ 已实现 | 18 |
| 🔧 部分兼容 | 0 |
| ❌ 未实现 | 0 |
| 🟦 扩展 | 3 |

---

## 关键说明

#### 1. 当前已对齐的 reviewer blocker

本轮补齐了之前 review 指出的两类核心语义偏差：

1. `cudaEvent_t` 不再在构造阶段绑定“当前 device”，而是在第一次 `record()` 时按目标 stream 的 device lazy-create。
2. `EventFlag` 现在真实参与 event 创建；`elapsedTime()` 不再静默返回 `0.0`，而是按 PyTorch 约束校验 timing/record/completion 前置条件后调用底层 CUDA elapsed-time 路径。

#### 2. 为什么仍保留 raw-stream 兼容接口

`record(const cudaStream_t&)` 不属于 PyTorch 上游 `c10::Event` API。Paddle 当前保留它，仅用于兼容仍依赖旧接口的下游代码；文档上应把它视为临时扩展，而不是上游对齐面的组成部分。

#### 3. 当前验证来源

当前 Event 相关验证分成两层：

1. `PaddleCppAPITest/test/c10/core/EventCompatTest.cpp`：继续覆盖构造、`EventFlag`、移动语义、属性访问和 CPU 路径异常主干。
2. Paddle 内部 `test/cpp/compat/c10_Event_test.cc` / `test/cpp/compat/ATen_record_stream_test.cc`：补充覆盖 lazy-create、device index 一致性、timing 语义和 legacy raw-stream 兼容路径；对应 `ctest -R c10` 与 `ctest -R ATen` 已通过。

#### 4. 实现形态差异

Paddle 当前仍是“直接持有 CUDA event 句柄”的实现；PyTorch 是后端无关的 pimpl/guard-impl 分发。两者内部结构不同，但在当前已支持的 CUDA 路径上，外部可观察语义已经基本对齐。
