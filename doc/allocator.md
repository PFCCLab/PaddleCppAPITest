##### Allocator.h 头文件 API 兼容性

对比文件：
- `/home/may/Paddle/paddle/phi/api/include/compat/c10/core/Allocator.h`
- `/home/may/pytorch/c10/core/Allocator.h`

状态说明：
- `✅` 已实现（接口存在且签名兼容）
- `🔧` 部分兼容（接口存在，但行为/能力有差异）
- `❌` 未实现（PyTorch 有，Paddle compat 头文件无）

---

### 类型与基础定义

| torch API | paddle API 兼容性 | 备注 |
|---|---|---|
| `DeleterFnPtr` | ✅ | 已定义 |
| `CaptureId_t` | ❌ | 未定义 |
| `MempoolId_t` | ❌ | 未定义 |
| `MempoolIdHash` | ❌ | 未定义 |

---

### `DataPtr`

| torch API | paddle API 兼容性 | 备注 |
|---|---|---|
| `DataPtr()` | ✅ | 已实现 |
| `DataPtr(void*, Device)` | 🔧 | 已实现；设备映射仅覆盖 CPU/CUDA，其他类型归为 `UNDEFINED` |
| `DataPtr(void*, void*, DeleterFnPtr, Device)` | 🔧 | 已实现；设备语义同上 |
| `operator->()` | ✅ | 已实现 |
| `unsafe_reset_data_and_ctx()` | ✅ | 已实现 |
| `clear()` | ✅ | 已实现 |
| `get()` | ✅ | 已实现 |
| `mutable_get()` | ✅ | 已实现 |
| `get_context()` | ✅ | 已实现 |
| `release_context()` | ✅ | 已实现 |
| `move_context()` | ✅ | 已实现 |
| `operator bool()` | ✅ | 已实现 |
| `cast_context<T>()` | ✅ | 已实现 |
| `get_deleter()` | ✅ | 已实现 |
| `device()` | 🔧 | 已实现；通过 `phi::Place` 反向映射到 `c10::Device` |
| `compare_exchange_deleter()` | ❌ | 未实现 |
| `unsafe_set_device()` | ❌ | 未实现 |
| `operator==(DataPtr, nullptr_t)` | ✅ | 已实现 |
| `operator==(nullptr_t, DataPtr)` | ✅ | 已实现 |
| `operator!=(DataPtr, nullptr_t)` | ✅ | 已实现 |
| `operator!=(nullptr_t, DataPtr)` | ✅ | 已实现 |

---

### `Allocator`

| torch API | paddle API 兼容性 | 备注 |
|---|---|---|
| `allocate(size_t)` | ✅ | 纯虚接口已声明 |
| `clone(const void*, size_t)` | ✅ | 已实现 |
| `is_simple_data_ptr(const DataPtr&)` | ✅ | 已实现 |
| `raw_deleter()` | ✅ | 默认返回 `nullptr` |
| `raw_allocate(size_t)` | ✅ | 已实现 |
| `raw_deallocate(void*)` | ✅ | 已实现 |
| `copy_data(void*, const void*, size_t)` | ✅ | 纯虚接口已声明 |
| `default_copy_data(void*, const void*, size_t)` | ✅ | `protected` 辅助实现 |

---

### 全局注册与扩展接口

| torch API | paddle API 兼容性 | 备注 |
|---|---|---|
| `SetAllocator(DeviceType, Allocator*, uint8_t)` | ❌ | 未实现 |
| `GetAllocator(const DeviceType&)` | ❌ | 未实现 |
| `AllocatorRegisterer` | ❌ | 未实现 |
| `REGISTER_ALLOCATOR` | ❌ | 未实现 |
| `InefficientStdFunctionContext` | ❌ | 未实现 |
| `InefficientStdFunctionContext::makeDataPtr()` | ❌ | 未实现 |

---

### 内存分析相关接口

| torch API | paddle API 兼容性 | 备注 |
|---|---|---|
| `MemoryReportingInfoBase` | ❌ | 未实现 |
| `memoryProfilingEnabled()` | ❌ | 未实现 |
| `reportMemoryUsageToProfiler()` | ❌ | 未实现 |
| `reportOutOfMemoryToProfiler()` | ❌ | 未实现 |
| `GatheredContext` | ❌ | 未实现 |
| `CachingAllocator::Stat/StatType/...` | ❌ | 未实现 |

---

### 命名空间别名

| torch API | paddle API 兼容性 | 备注 |
|---|---|---|
| `at::DataPtr = c10::DataPtr` | ✅ | 已实现 |

---

### 兼容性统计（基于以上条目）

| 状态 | 数量 |
|---|---|
| ✅ 已实现 | 22 |
| 🔧 部分兼容 | 3 |
| ❌ 未实现 | 17 |

---

### 结论

- `DataPtr` 核心接口和 `Allocator` 主体接口在 compat 头文件中已基本具备。
- 与上游 PyTorch 的主要差距集中在：全局分配器注册机制、`InefficientStdFunctionContext`、内存分析与 `CachingAllocator` 相关接口。
- 设备语义是当前的主要“部分兼容”点：Paddle 侧实现基于 `phi::Place`，并非完全等价于 PyTorch 的 `c10::Device` 语义集合。
