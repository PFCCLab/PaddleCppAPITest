# CUDA 工具类（CUDAGuard / CUDAStream / PhiloxCudaState：接口形态有差异）

> Paddle 头文件：`c10/cuda/CUDAGuard.h`、`c10/cuda/CUDAStream.h`、`c10/cuda/PhiloxCudaState.h`
> 测试文件：`test/c10/cuda/CUDATest2.cpp`

## 差异点列表

当前差异并非“文件缺失”，而是**接口形态与测试写法不一致**，导致对应测试被 `#ifndef USE_PADDLE_API` 整块保护跳过：

| 相关类/结构 | 头文件 | 现状 |
|-------------|--------|------|
| `c10::cuda::CUDAGuard` | `c10/cuda/CUDAGuard.h` | 可用，构造/成员形态与 Torch 有差异 |
| `c10::cuda::OptionalCUDAGuard` | `c10/cuda/CUDAGuard.h` | 可用 |
| `c10::cuda::CUDAStream` | `c10/cuda/CUDAStream.h` | 可用，默认构造与常量形态有差异 |
| `c10::cuda::getCurrentCUDAStream()` | `c10/cuda/CUDAStream.h` | 可用 |
| `PhiloxCudaState` | `c10/cuda/PhiloxCudaState.h` | 可用，但命名空间与 Torch 测试写法不同 |

---

## Diff 测试用例位置

测试文件：`test/c10/cuda/CUDATest2.cpp`

### 测试用例原文

```cpp
// Paddle 路径仅占位，Torch 路径执行真实同步
TEST_F(CUDATest2, StreamSynchronize) {
#ifndef USE_PADDLE_API
  auto stream = c10::cuda::getCurrentCUDAStream();
  c10::cuda::stream_synchronize(stream.stream());
#else
  file << "stream_sync_placeholder ";
#endif
}

// 大量 CUDA 工具类用例仅在 Torch 路径编译
#ifndef USE_PADDLE_API
TEST_F(CUDATest2, CUDAGuardDefault) { /* ... */ }
TEST_F(CUDATest2, CUDAStreamDefault) { /* ... */ }
TEST_F(CUDATest2, PhiloxCudaStateDefault) { /* ... */ }
#endif
```

---

## 输出对比

| 测试用例 | Paddle 输出 | Torch 输出 |
|---------|------------|------------|
| CUDAGuardDefault | 未执行（`#ifndef USE_PADDLE_API` 下被编译排除） | 已执行 |
| CUDAStreamDefault | 未执行（`#ifndef USE_PADDLE_API` 下被编译排除） | 已执行 |
| PhiloxCudaStateDefault | 未执行（`#ifndef USE_PADDLE_API` 下被编译排除） | 已执行 |

---

## 初步问题分析

Paddle compat 已提供上述 CUDA 相关头文件与主要类/函数。当前差异的根因是：
- `test/c10/cuda/CUDATest2.cpp` 中相关用例整体被 `#ifndef USE_PADDLE_API` 排除，Paddle 路径缺少同等覆盖；
- 兼容层接口形态与 Torch 侧测试写法不完全一致（例如构造方式、命名空间与成员能力）。

---
