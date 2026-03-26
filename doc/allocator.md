##### Allocator.h 头文件 API 兼容性

对比文件：
- `/home/may/Paddle/paddle/phi/api/include/compat/c10/core/Allocator.h`
- `/home/may/pytorch/c10/core/Allocator.h`

状态说明：
- `✅` 已实现（接口存在且签名/语义基本一致）
- `🔧` 部分兼容（接口存在，但签名或实现语义有差异）
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
| `DataPtr(void*, Device)` | ✅ | 已实现；完整保留 `Device`（含 device index），通过 `device._PD_GetInner()` 存入 `phi::Place` |
| `DataPtr(void*, void*, DeleterFnPtr, Device)` | ✅ | 已实现；device index 保留语义同上 |
| `DataPtr(DataPtr&&)` / `operator=(DataPtr&&)` | ✅ | 已实现；move-only（copy 构造与 copy 赋值显式 `= delete`，与 PyTorch 接口一致） |
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
| `device()` | ✅ | 已实现；返回 `Device(device_)`（利用 `phi::Place` 到 `c10::Device` 隐式转换），完整保留 device index |
| `compare_exchange_deleter()` | ✅ | 已实现 |
| `unsafe_set_device()` | ✅ | 已实现；通过 `device._PD_GetInner()` 更新内部 `phi::Place` |
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
| `copy_data(void*, const void*, size_t)` | ✅ | 纯虚接口已声明；`PaddleCUDAAllocatorAdapter` 重写此方法，使用 `cudaMemcpy(..., cudaMemcpyDeviceToDevice)` 实现 GPU-to-GPU 拷贝（支持 `clone()` 语义） |
| `default_copy_data(void*, const void*, size_t)` | ✅ | `protected` 辅助实现（CPU 路径使用 `std::memcpy`；CUDA allocator 重写 `copy_data` 以使用 D2D copy） |

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
| ✅ 已实现 | 27 |
| 🔧 部分兼容 | 0 |
| ❌ 未实现 | 16 |

---

### 结论

- `DataPtr` 核心接口和 `Allocator` 主体接口在 compat 头文件中已基本具备。
- 与上游 PyTorch 的主要差距集中在：全局分配器注册机制、`InefficientStdFunctionContext`、内存分析与 `CachingAllocator` 相关接口。
- **PR #78060 修复记录**：
  - `DataPtr::device()` 现在完整保留 GPU device index（多卡场景下 `device().index()` 返回正确值）；新增 `unsafe_set_device()` 实现。内部实现通过 `c10::Device::_PD_GetInner()` 直接存储完整 `phi::Place`。
  - **本轮修复** — `PaddleCUDAAllocatorAdapter::allocate(0)`：不再返回默认 CPU `DataPtr()`，改为返回 `DataPtr(nullptr, nullptr, nullptr, Device(CUDA/HIP, current_device_id))`，保留当前 CUDA 设备信息，使 `allocate(0).device().type() == DeviceType::CUDA`。
  - **本轮修复** — `PaddleCUDAAllocatorAdapter::copy_data()`：不再继承 `default_copy_data`（使用 `std::memcpy`），改为调用 `cudaMemcpy(dst, src, n, cudaMemcpyDeviceToDevice)`（HIP 路径使用 `hipMemcpy`），正确支持 GPU-to-GPU 内存拷贝，兼容 `c10::Allocator::clone()` 语义。
  - **本轮修复** — `DataPtr` 显式 move-only：新增 `DataPtr(const DataPtr&) = delete` 与 `DataPtr& operator=(const DataPtr&) = delete`，与 PyTorch 的 `c10::DataPtr` 接口严格一致。（原实现依赖 `UniqueVoidPtr` 内部 `unique_ptr` 隐式禁止拷贝，现显式声明使接口契约明确。）
  - **本轮修复** — `PaddleCUDAAllocatorAdapter::raw_deleter()` 返回 `nullptr`：`c10::Allocator` raw API 契约要求 `allocate(n)` 返回的 `DataPtr` 满足 `get() == get_context()`。本适配器中 `data` 为设备原始指针，`context` 为 `phi::Allocation*`，二者不等，因此不能宣称 raw API 可用，`raw_deleter()` 显式返回 `nullptr` 避免误用 `raw_allocate`/`raw_deallocate`。
