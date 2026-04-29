## CUDAGuard.h 头文件 API 兼容性

对比文件：
- `/home/may/Paddle/paddle/phi/api/include/compat/c10/cuda/CUDAGuard.h`
- `/home/may/pytorch/c10/cuda/CUDAGuard.h`

相关实现与测试：
- Paddle 底层 guard：`/home/may/Paddle/paddle/phi/core/platform/cuda_device_guard.h`
- Paddle 底层析构实现：`/home/may/Paddle/paddle/phi/core/platform/cuda_device_guard.cc`
- Paddle compat 测试：`/home/may/Paddle/test/cpp/compat/ATen_CUDAContext_test.cc`
- Paddle compat 测试：`/home/may/Paddle/test/cpp/compat/ATen_basic_test.cc`
- PaddleCppAPITest 测试：`/home/may/PaddleCppAPITest/test/c10/cuda/CUDATest2.cpp`

状态说明：
- `✅` 已实现（接口存在且签名/语义基本一致）
- `🔧` 部分兼容（接口存在，但实现路径或边界行为不同）
- `❌` 未实现（PyTorch 有，Paddle compat 头文件无）

---

### Device Guard 家族

| torch API | paddle API 兼容性 | 测试用例状态 | 优先级 | 备注 |
|---|---|---|---|---|
| `c10::cuda::CUDAGuard(DeviceIndex)` | ✅ | - [x] | P0 | 已实现；构造时快照 `original_device()`，随后切换到目标 CUDA 设备 |
| `c10::cuda::CUDAGuard(Device)` | ✅ | - [x] | P0 | 已实现；先校验 `device.is_cuda()`，无 index 时回退到当前 CUDA 设备 |
| `CUDAGuard::set_device(Device)` | ✅ | - [x] | P0 | 已实现；更新当前设备并同步 `current_device()` |
| `CUDAGuard::reset_device(Device)` | ✅ | - [x] | P1 | 已实现；语义上等价于 `set_device(Device)` |
| `CUDAGuard::set_index(DeviceIndex)` | ✅ | - [x] | P0 | 已实现；直接按设备号切换 |
| `CUDAGuard::original_device()` | ✅ | - [x] | P0 | 已实现；返回构造时快照的原始设备 |
| `CUDAGuard::current_device()` | ✅ | - [x] | P0 | 已实现；返回最近一次由该 guard 设置的设备 |

---

### Optional Device Guard 家族

| torch API | paddle API 兼容性 | 测试用例状态 | 优先级 | 备注 |
|---|---|---|---|---|
| `c10::cuda::OptionalCUDAGuard()` | ✅ | - [x] | P0 | 已实现；默认处于未初始化状态 |
| `c10::cuda::OptionalCUDAGuard(std::optional<Device>)` | ✅ | - [x] | P0 | 已实现；仅在 `device_opt.has_value()` 时初始化 |
| `c10::cuda::OptionalCUDAGuard(std::optional<DeviceIndex>)` | ✅ | - [ ] | P1 | 已实现；按可选设备号懒初始化 |
| `OptionalCUDAGuard::set_device(Device)` | ✅ | - [x] | P0 | 已实现；首次调用时建立底层 guard 并记录原设备 |
| `OptionalCUDAGuard::reset_device(Device)` | ✅ | - [ ] | P1 | 已实现；语义上等价于 `set_device(Device)` |
| `OptionalCUDAGuard::set_index(DeviceIndex)` | ✅ | - [ ] | P1 | 已实现；首次调用时建立底层 guard |
| `OptionalCUDAGuard::original_device()` | ✅ | - [x] | P0 | 已实现；未初始化时返回 `nullopt` |
| `OptionalCUDAGuard::current_device()` | ✅ | - [x] | P0 | 已实现；未初始化时返回 `nullopt` |
| `OptionalCUDAGuard::reset()` | ✅ | - [x] | P0 | 已实现；恢复原始设备并回到未初始化状态 |

---

### 命名空间别名

| torch API | paddle API 兼容性 | 测试用例状态 | 优先级 | 备注 |
|---|---|---|---|---|
| `at::cuda::CUDAGuard` | ✅ | - [ ] | P2 | 已通过 `using c10::cuda::CUDAGuard` 导出 |
| `at::cuda::OptionalCUDAGuard` | ✅ | - [ ] | P2 | 已通过 `using c10::cuda::OptionalCUDAGuard` 导出 |

---

### PyTorch 同头文件中 Paddle 尚未覆盖的 Stream Guard 家族

| torch API | paddle API 兼容性 | 测试用例状态 | 优先级 | 备注 |
|---|---|---|---|---|
| `c10::cuda::CUDAStreamGuard(Stream)` | ❌ | - [ ] | P1 | PyTorch 同头文件提供；Paddle compat 当前未定义 |
| `CUDAStreamGuard::reset_stream(Stream)` | ❌ | - [ ] | P1 | 缺失 |
| `CUDAStreamGuard::original_stream()` | ❌ | - [ ] | P1 | 缺失 |
| `CUDAStreamGuard::current_stream()` | ❌ | - [ ] | P1 | 缺失 |
| `CUDAStreamGuard::current_device()` | ❌ | - [ ] | P1 | 缺失 |
| `CUDAStreamGuard::original_device()` | ❌ | - [ ] | P1 | 缺失 |
| `c10::cuda::OptionalCUDAStreamGuard()` | ❌ | - [ ] | P1 | 缺失 |
| `OptionalCUDAStreamGuard(Stream)` | ❌ | - [ ] | P1 | 缺失 |
| `OptionalCUDAStreamGuard(std::optional<Stream>)` | ❌ | - [ ] | P1 | 缺失 |
| `OptionalCUDAStreamGuard::reset_stream(Stream)` | ❌ | - [ ] | P1 | 缺失 |
| `OptionalCUDAStreamGuard::original_stream()` | ❌ | - [ ] | P1 | 缺失 |
| `OptionalCUDAStreamGuard::current_stream()` | ❌ | - [ ] | P1 | 缺失 |
| `OptionalCUDAStreamGuard::reset()` | ❌ | - [ ] | P1 | 缺失 |
| `c10::cuda::CUDAMultiStreamGuard(ArrayRef<CUDAStream>)` | ❌ | - [ ] | P2 | 缺失 |

---

### 内部辅助实现（不计入统计）

| Paddle 内部实现 | 作用 | 备注 |
|---|---|---|
| `detail::current_cuda_device()` | 读取当前 CUDA 设备 | 用于在 guard 构造前快照原始设备 |
| `detail::normalize_cuda_device(Device)` | 规范化输入设备 | 对无 index 的 CUDA `Device` 回退到当前设备，并拒绝非 CUDA 设备 |
| `detail::restore_cuda_device(...)` | 显式恢复原始设备 | PyTorch 公开头文件里无同名 helper；Paddle 用它补齐与底层 `CUDADeviceGuard` 不同的恢复语义 |

---

### 兼容性统计

统计口径：
- 仅统计 PyTorch `c10/cuda/CUDAGuard.h` 中的公开类型和公开成员函数
- 不统计已删除的拷贝/移动特殊成员函数
- 不统计 Paddle 额外引入的 `detail::*` 内部 helper

| 状态 | 数量 |
|---|---|
| ✅ 已实现 | 18 |
| 🔧 部分兼容 | 0 |
| ❌ 未实现 | 14 |

---

### 备注

1. Paddle 当前文档对应的 compat 头文件只覆盖了 `CUDAGuard` 和 `OptionalCUDAGuard` 两个“设备 guard”类型；PyTorch 同头文件里的 `CUDAStreamGuard`、`OptionalCUDAStreamGuard`、`CUDAMultiStreamGuard` 尚未在 Paddle compat 层提供。

2. `CUDAGuard` / `OptionalCUDAGuard` 的实现路径与 PyTorch 不同。PyTorch 直接基于 `InlineDeviceGuard<CUDAGuardImpl>`；Paddle 则复用 `paddle::platform::CUDADeviceGuard`，并额外维护：
   - `original_device_`
   - `current_device_`
   - `detail::restore_cuda_device(...)`

3. 这层额外恢复逻辑是必要的。Paddle 底层 `CUDADeviceGuard` 的析构带有“首次 `prev_id == 0` 时不恢复”的优化，而 PyTorch `CUDAGuard` 语义要求作用域退出后回到构造时的原始设备。Paddle compat 因此把底层 guard 放进 `std::optional`，先 `reset()` 掉底层 guard，再显式恢复 `original_device_`，从而保证：
   - `torch::cuda::synchronize(device)` 不污染调用方当前设备
   - 多次 `set_index()` / `set_device()` 后仍能在析构时回到最初设备

4. `OptionalCUDAGuard` 也采用 `std::optional<paddle::platform::CUDADeviceGuard>`，除了和 `CUDAGuard` 一样控制析构顺序外，还承担“未初始化 / 已初始化”状态表达，与 PyTorch `OptionalDeviceGuard` 的 lazy-init 语义一致。

5. 当前测试覆盖已能验证核心 RAII 行为：
   - `CUDATest2.CUDAGuardDeviceCtor`
   - `CUDATest2.CUDAGuardLifecycle`
   - `CUDATest2.OptionalCUDAGuardLifecycle`
   - `CUDAFunctionsTest.TorchSynchronizePreservesCurrentDevice`
   - `CUDAFunctionsTest.CUDAGuardRestoresOriginalDeviceAfterMultipleSwitches`

6. 如果后续需要继续补齐 PyTorch 同头文件能力，优先级建议是：
   - 先补 `CUDAStreamGuard`
   - 再补 `OptionalCUDAStreamGuard`
   - 最后补 `CUDAMultiStreamGuard`
