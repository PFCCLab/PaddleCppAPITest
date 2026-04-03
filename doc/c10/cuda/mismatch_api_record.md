# CUDA 工具类（CUDAGuard / CUDAStream / PhiloxCudaState：接口已对齐）

> Paddle 头文件：`c10/cuda/CUDAGuard.h`、`c10/cuda/CUDAStream.h`、`ATen/cuda/PhiloxCudaState.h`
> 测试文件：`test/c10/cuda/CUDATest2.cpp`

## 2026-04-02 CUDAStream review blocker 收敛（Paddle 内部 ctest 已验证）

### 本轮复核

| 测试项 | 当前 Paddle | PyTorch | 结论 |
|--------|-------------|---------|------|
| `getStreamFromPool(true)` | bool 重载恢复 `device_index = -1` 默认参数，不再静默落到 `int priority` 重载 | 同签名、同语义 | ✅ 已对齐 |
| `CUDAStream::raw_stream()` legacy alias | 暂时保留，行为等价于 `stream()`，避免在本轮 misc apis 对齐里引入 breaking change | 上游无该旧入口 | ✅ compat surface 保持稳定 |

说明：

- reviewer 指出的两个 blocker 都落在 `paddle/phi/api/include/compat/c10/cuda/CUDAStream.h`：
  - `getStreamFromPool(const bool isHighPriority = false, DeviceIndex device_index = -1)` 的默认参数缺失会让 `getStreamFromPool(true)` 误绑到 `int` 重载，并错误返回低优先级 stream；
  - `raw_stream()` 的删除属于 breaking change，应从本轮 “misc apis” 对齐范围中剥离。
- Paddle 内部新增 `test/cpp/compat/c10_Stream_test.cc` 回归，直接覆盖 `getStreamFromPool(true)` 与 `raw_stream()`。
- 验证来自 `/home/may/Paddle/build` 下的 `ninja -j16`、`ctest -R c10 --output-on-failure`、`ctest -R ATen --output-on-failure`，当前均通过。

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
