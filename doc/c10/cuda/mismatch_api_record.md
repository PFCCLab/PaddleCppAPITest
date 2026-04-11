# CUDA 工具类（CUDAGuard / CUDAStream / PhiloxCudaState：接口已对齐）

> Paddle 头文件：`c10/cuda/CUDAGuard.h`、`c10/cuda/CUDAStream.h`、`ATen/cuda/PhiloxCudaState.h`
> 测试文件：`test/c10/cuda/CUDATest2.cpp`

## 2026-04-04 Stream Pool 实现与 PyTorch 对齐

### 本轮修改

按照 PyTorch 实现方式重写 Paddle Stream Pool：

| 修改项 | 原实现 | 新实现 | 状态 |
|--------|--------|--------|------|
| Stream Pool 数据结构 | `struct StreamPoolState` 含高/低两档数组 | 三维数组 `[priority][device][idx]`，最多 4 档 | ✅ 已对齐 |
| `getStreamFromPool(int, ...)` | 仅区分 `priority < 0` 高/`>= 0` 低 | `std::clamp(-priority, 0, 3)` 映射到多档 | ✅ 已对齐 |
| `raw_stream()` | legacy alias 保留 | **已删除**（PyTorch 无此接口） | ✅ 已清理 |

---

## 本轮对齐内容

- `c10::cuda::CUDAGuard` 补齐了 `original_device()`，并把 `current_device()` 语义改成“最近一次由 guard 设置的设备”。
- `c10::cuda::OptionalCUDAGuard` 补齐了 `original_device()` 和 `reset()`，生命周期与 PyTorch 对齐。
- `c10::cuda::CUDAStream` 补齐了 `UNCHECKED`、`query()`、`synchronize()`、`priority()`、`priority_range()`、`pack3()`、`unpack3()`、`getStreamFromExternal()`、`operator<<` 和 `std::hash`。
- `PhiloxCudaState` 保持与 PyTorch 一致的 canonical 路径：只从 `ATen/cuda/PhiloxCudaState.h` 暴露，不在 `c10/cuda` 下新增同名 shim。

---

## 测试侧修正

之前 `test/c10/cuda/CUDATest2.cpp` 使用了 `#ifndef USE_PADDLE_API`。但工程里 `USE_PADDLE_API` 在 Torch / Paddle 两个目标上都会被定义，只是值分别为 `0` 和 `1`，所以这批测试实际上在两边都被预处理排除了。

本轮改动将该文件改成两边共同编译、共同执行，并直接覆盖以下接口：

- `device_synchronize()`
- `stream_synchronize()`
- `CUDAGuard(DeviceIndex / Device)`
- `CUDAGuard::original_device()` / `current_device()` / `set_device()` / `reset_device()` / `set_index()`
- `OptionalCUDAGuard::original_device()` / `current_device()` / `reset()`
- `CUDAStream::UNCHECKED`
- `CUDAStream::query()` / `synchronize()` / `priority()` / `priority_range()`
- `CUDAStream::pack3()` / `unpack3()`
- `getCurrentCUDAStream()` / `getStreamFromPool()` / `setCurrentCUDAStream()` / `getStreamFromExternal()`
- `operator<<(ostream&, CUDAStream)` / `std::hash<CUDAStream>`
- `ATen/cuda/PhiloxCudaState.h` 下的 `PhiloxCudaState` 默认构造、普通构造和 graph-capture 构造

---

## 结论

这组差异的根因不是“Paddle 头文件完全缺失”，而是两部分叠加：

- 兼容层确实缺少一批 PyTorch 已有的 `CUDAGuard` / `CUDAStream` 成员与辅助接口；
- 测试文件的宏判断写错，导致这批 CUDA 工具类用例长期没有在任一构建目标里生效。

接口补齐后，`CUDATest2.cpp` 已经改为真实覆盖这些 API，不再依赖占位输出。
