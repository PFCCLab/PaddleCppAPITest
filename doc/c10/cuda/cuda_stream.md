## CUDAStream.h 头文件 API 兼容性

对比文件：
- `/home/may/Paddle/paddle/phi/api/include/compat/c10/cuda/CUDAStream.h`
- `/home/may/pytorch/c10/cuda/CUDAStream.h`

状态说明：
- `✅` 已实现（接口存在且签名/语义基本一致）
- `🔧` 部分兼容（接口存在，但签名或实现语义有差异）
- `❌` 未实现（PyTorch 有，Paddle compat 头文件无）

---

## 2026-04-02 review follow-up

本轮根据 reviewer comment 又补了两个收口点：

- `getStreamFromPool(const bool isHighPriority = false, DeviceIndex device_index = -1)` 恢复了 `device_index = -1` 默认参数，避免 `getStreamFromPool(true)` 静默绑定到 `int priority` 重载并错误返回低优先级 stream。
- `raw_stream()` 暂时保留为 compat legacy alias，当前行为仍等价于 `stream()`，避免在这组 “misc apis” 对齐改动里引入 breaking change。
- Paddle 内部新增 `test/cpp/compat/c10_Stream_test.cc` 回归，直接覆盖 `getStreamFromPool(true)` 与 `raw_stream()`。

---

## 当前结论

本轮对齐后，`c10/cuda/CUDAStream.h` 中 PyTorch 侧常用接口已经全部补齐，`CUDATest2.cpp` 也已经覆盖并通过了以下能力：

- `CUDAStream::UNCHECKED`
- `CUDAStream(Stream)` / `CUDAStream(Unchecked, Stream)`
- `operator==` / `operator!=`
- `operator cudaStream_t()` / `operator Stream()`
- `device_type()` / `device_index()` / `device()`
- `id()` / `stream()` / `raw_stream()` / `unwrap()`
- `query()` / `synchronize()`
- `priority()` / `priority_range()`
- `pack3()` / `unpack3()`
- `getCurrentCUDAStream()` / `getDefaultCUDAStream()`
- `getStreamFromPool(bool, DeviceIndex)` / `getStreamFromPool(int, DeviceIndex)`
- `getStreamFromExternal(cudaStream_t, DeviceIndex)`
- `setCurrentCUDAStream(CUDAStream)`
- `operator<<(ostream&, CUDAStream)`
- `std::hash<CUDAStream>`

---

## 兼容性表

| torch API | paddle API 兼容性 | 备注 |
|-----------|------------------|------|
| `CUDAStream::UNCHECKED` | ✅ | 已补齐无检查构造标签 |
| `CUDAStream(Unchecked, Stream)` | ✅ | 已补齐 |
| `query()` | ✅ | 通过 `c10::Stream::query()` 委托 |
| `synchronize()` | ✅ | 通过 `c10::Stream::synchronize()` 委托 |
| `priority()` | ✅ | 调用 `cudaStreamGetPriority` |
| `priority_range()` | ✅ | 调用 `cudaDeviceGetStreamPriorityRange` |
| `pack3()` / `unpack3()` | ✅ | 直接复用 `c10::Stream` 的 pack/unpack |
| `getStreamFromPool(bool, DeviceIndex)` | ✅ | `device_index` 默认值为 `-1`，`getStreamFromPool(true)` 不会再误绑到 `int` 重载 |
| `getStreamFromPool(int, DeviceIndex)` | ✅ | 负优先级走高优先级流池，非负走低优先级流池 |
| `getStreamFromExternal(cudaStream_t, DeviceIndex)` | ✅ | 通过 `make_cuda_stream` 包装外部流 |
| `operator<<(ostream&, CUDAStream)` | ✅ | 委托到底层 `c10::Stream` 输出 |
| `std::hash<CUDAStream>` | ✅ | 委托 `std::hash<c10::Stream>` |

---

## 验证状态

- `/home/may/Paddle/build` 下 `ctest -R c10 --output-on-failure`：14 / 14 通过
- `/home/may/Paddle/build` 下 `ctest -R ATen --output-on-failure`：44 / 44 通过
- `torch_CUDATest2 --gtest_list_tests`：8 个用例全部注册
- `paddle_CUDATest2 --gtest_list_tests`：8 个用例全部注册
- `torch_CUDATest2`：8 / 8 通过
- `paddle_CUDATest2`：8 / 8 通过
- `/tmp/paddle_cpp_api_test/torch_CUDATest2.txt` 与 `/tmp/paddle_cpp_api_test/paddle_CUDATest2.txt`：当前输出一致
