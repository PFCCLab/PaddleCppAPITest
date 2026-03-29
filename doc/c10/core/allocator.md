## Allocator.h 头文件 API 兼容性

对比文件：
- `/home/may/Paddle/paddle/phi/api/include/compat/c10/core/Allocator.h`
- `/home/may/pytorch/c10/core/Allocator.h`

状态说明：
- `✅` 已实现（接口存在且签名/语义基本一致）
- `🔧` 部分兼容（接口存在，但签名或实现语义有差异）
- `❌` 未实现（PyTorch 有，Paddle compat 头文件无）

---

### 类型与基础定义

| torch API | paddle API 兼容性 | 测试用例状态 | 优先级 | 备注 |
|-----------|------------------|------------|-------|------|
| `DeleterFnPtr` | ✅ | - [x] | P0 | 已定义，`AllocatorCompatTest` 覆盖 |
| `CaptureId_t` | ✅ | - [x] | P2 | 已定义，`CaptureAndMempoolTypes` 覆盖 |
| `MempoolId_t` | ✅ | - [x] | P2 | 已定义，`CaptureAndMempoolTypes` 覆盖 |
| `MempoolIdHash` | ✅ | - [x] | P2 | 已定义，`CaptureAndMempoolTypes` 覆盖 |

---

### `DataPtr` 构造与所有权

| torch API | paddle API 兼容性 | 测试用例状态 | 优先级 | 备注 |
|-----------|------------------|------------|-------|------|
| `DataPtr()` | ✅ | - [x] | P0 | 已实现，`AllocatorCompatTest` 覆盖 |
| `DataPtr(void*, Device)` | ✅ | - [x] | P0 | 已实现，`AllocatorCompatTest` 覆盖 |
| `DataPtr(void*, void*, DeleterFnPtr, Device)` | ✅ | - [x] | P0 | 已实现，`AllocatorCompatTest` 覆盖 |
| copy ctor / copy assignment（move-only 语义） | ✅ | - [ ] | P1 | Paddle 显式 `= delete`，与 PyTorch 语义一致 |
| `DataPtr(DataPtr&&)` / `operator=(DataPtr&&)` | ✅ | - [x] | P0 | 已实现，`AllocatorCompatTest` 覆盖 |
| `mutable_get()` | ✅ | - [x] | P2 | 已实现，`MutableGet` 覆盖 |

---

### `DataPtr` 观察器与操作

| torch API | paddle API 兼容性 | 测试用例状态 | 优先级 | 备注 |
|-----------|------------------|------------|-------|------|
| `operator->()` | ✅ | - [x] | P1 | 已实现，`AllocatorCompatTest` 覆盖 |
| `unsafe_reset_data_and_ctx()` | ✅ | - [ ] | P1 | 已实现 |
| `clear()` | ✅ | - [x] | P1 | 已实现，`AllocatorCompatTest` 覆盖 |
| `get()` | ✅ | - [x] | P0 | 已实现，`AllocatorCompatTest` 覆盖 |
| `get_context()` | ✅ | - [x] | P1 | 已实现，`AllocatorCompatTest` 覆盖 |
| `release_context()` | ✅ | - [ ] | P1 | 已实现 |
| `move_context()` | ✅ | - [ ] | P2 | 已实现 |
| `operator bool()` | ✅ | - [x] | P0 | 已实现，`AllocatorCompatTest` 覆盖 |
| `cast_context<T>()` | ✅ | - [ ] | P2 | 已实现 |
| `get_deleter()` | ✅ | - [x] | P1 | 已实现，`AllocatorCompatTest` 覆盖 |
| `compare_exchange_deleter()` | ✅ | - [ ] | P2 | 已实现 |
| `device()` | ✅ | - [ ] | P0 | 已实现 |
| `unsafe_set_device()` | ✅ | - [ ] | P1 | 已实现 |
| `operator== (DataPtr, nullptr_t)` | ✅ | - [x] | P1 | 已实现，`AllocatorCompatTest` 覆盖 |
| `operator== (nullptr_t, DataPtr)` | ✅ | - [x] | P1 | 已实现，`AllocatorCompatTest` 覆盖 |
| `operator!= (DataPtr, nullptr_t)` | ✅ | - [x] | P1 | 已实现，`AllocatorCompatTest` 覆盖 |
| `operator!= (nullptr_t, DataPtr)` | ✅ | - [x] | P1 | 已实现，`AllocatorCompatTest` 覆盖 |

---

### `Allocator` 核心接口

| torch API | paddle API 兼容性 | 测试用例状态 | 优先级 | 备注 |
|-----------|------------------|------------|-------|------|
| `allocate(size_t)` | ✅ | - [ ] | P0 | 纯虚接口已声明 |
| `clone(const void*, size_t)` | ✅ | - [ ] | P1 | 已实现 |
| `is_simple_data_ptr(const DataPtr&)` | ✅ | - [x] | P1 | 语义已修正为 `get() == get_context()`，`IsSimpleDataPtrSemantics` 覆盖 |
| `raw_deleter()` | ✅ | - [ ] | P1 | 默认返回 `nullptr` |
| `raw_allocate(size_t)` | ✅ | - [ ] | P1 | 已实现 |
| `raw_deallocate(void*)` | ✅ | - [ ] | P1 | 已实现 |
| `copy_data(void*, const void*, size_t)` | ✅ | - [ ] | P0 | 纯虚接口已声明 |
| `default_copy_data(void*, const void*, size_t)` | ✅ | - [ ] | P2 | `protected` 辅助实现 |

---

### 全局注册与扩展接口

| torch API | paddle API 兼容性 | 测试用例状态 | 优先级 | 备注 |
|-----------|------------------|------------|-------|------|
| `SetAllocator(DeviceType, Allocator*, uint8_t)` | ✅ | - [x] | P2 | 已实现，`SetAndGetAllocatorPriority` 覆盖 |
| `GetAllocator(const DeviceType&)` | ✅ | - [x] | P2 | 已实现，`SetAndGetAllocatorPriority` 覆盖 |
| `AllocatorRegisterer` | ✅ | - [x] | P2 | 已实现，通过 `REGISTER_ALLOCATOR` 编译探针间接覆盖 |
| `REGISTER_ALLOCATOR` | ✅ | - [x] | P2 | 已实现，`RegisterAllocatorMacro` 作为编译探针覆盖 |
| `InefficientStdFunctionContext` | ✅ | - [x] | P2 | 已实现，`InefficientStdFunctionContextMakeDataPtr` 覆盖 |
| `InefficientStdFunctionContext::makeDataPtr()` | ✅ | - [x] | P2 | 已实现，`InefficientStdFunctionContextMakeDataPtr` 覆盖 |

---

### 内存分析相关接口

| torch API | paddle API 兼容性 | 测试用例状态 | 优先级 | 备注 |
|-----------|------------------|------------|-------|------|
| `MemoryReportingInfoBase` | ❌ | - [ ] | P3 | 未实现 |
| `memoryProfilingEnabled()` | ❌ | - [ ] | P3 | 未实现 |
| `reportMemoryUsageToProfiler()` | ❌ | - [ ] | P3 | 未实现 |
| `reportOutOfMemoryToProfiler()` | ❌ | - [ ] | P3 | 未实现 |
| `GatheredContext` | ❌ | - [ ] | P3 | 未实现 |
| `CachingAllocator::Stat/StatType/...` | ❌ | - [ ] | P3 | 未实现 |

---

### 命名空间别名

| torch API | paddle API 兼容性 | 测试用例状态 | 优先级 | 备注 |
|-----------|------------------|------------|-------|------|
| `at::DataPtr = c10::DataPtr` | ✅ | - [x] | P1 | 已实现，`AllocatorCompatTest` 覆盖 |

---

### 兼容性统计

| 状态 | 数量 |
|---|---|
| ✅ 已实现 | 42 |
| 🔧 部分兼容 | 0 |
| ❌ 未实现 | 6 |

---

### 备注

1. **优先级说明**：
   - P0: 核心功能，必须支持
   - P1: 常用功能，高优先级
   - P2: 进阶功能，中优先级
   - P3: 边缘功能，低优先级

2. **对比范围说明**：
   - 本文档基于头文件声明对比：
     - `paddle/phi/api/include/compat/c10/core/Allocator.h`
     - `/home/may/pytorch/c10/core/Allocator.h`

3. **主要差异说明**：
   - `DataPtr` / `Allocator` 直接接口已进一步对齐，`mutable_get()`、allocator 注册体系、`InefficientStdFunctionContext` 已补齐。
   - 当前剩余缺口主要集中在 profiler / caching allocator 辅助接口。
   - `is_simple_data_ptr()` 已修正为与 PyTorch 一致的 `get() == get_context()` 语义。

4. **测试现状**：
   - `test/c10/core/AllocatorCompatTest.cpp` 已覆盖 `DataPtr` 的构造、移动、`mutable_get()`、`InefficientStdFunctionContext`、allocator 注册与 `is_simple_data_ptr()` 语义。
   - profiling / caching allocator 相关缺失接口暂无直接测试。
