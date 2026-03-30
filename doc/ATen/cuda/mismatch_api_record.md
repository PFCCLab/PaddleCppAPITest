# CUDAContext（历史差异：曾因 `unmatch` 路径未进入常规回归）

> Paddle 头文件：`ATen/cuda/CUDAContext.h`

## 当前状态

1. 原 `test/ATen/cuda/unmatch_CUDAContextTest.cpp` 已迁移为常规测试文件 `test/ATen/cuda/CUDAContextTest.cpp`，会参与默认构建与 `result_cmp` 比对。
2. `getDeviceProperties`、`getCurrentDeviceProperties`、`getCurrentCUDAStream` 三个接口已纳入常规回归。
3. 为避免地址值与运行时噪声导致假阳性，当前测试只比较稳定摘要字段：
   设备属性测试比较 `prop != nullptr`、`major`、`minor`、`multiProcessorCount`；
   stream 测试比较 `device_index`、`device_type`，以及与显式/默认 stream 的一致性。
4. 当 CUDA 运行时不可用时，两端统一输出 `cuda_runtime_unavailable`，不再依赖 `unmatch` 路径做条件分叉。

---

## Diff 测试用例位置

测试文件：`test/ATen/cuda/CUDAContextTest.cpp`

### 当前测试策略

```cpp
static bool HasCudaRuntime() {
  try {
    return at::cuda::is_available();
  } catch (const std::exception&) {
    return false;
  }
}

TEST_F(CUDAContextTest, GetCurrentCUDAStream) {
  if (!HasCudaRuntime()) {
    file << "cuda_runtime_unavailable ";
    ...
    return;
  }

  auto default_stream = at::cuda::getDefaultCUDAStream(0);
  at::cuda::setCurrentCUDAStream(default_stream);
  auto current_stream = at::cuda::getCurrentCUDAStream();
  auto explicit_stream = at::cuda::getCurrentCUDAStream(0);

  file << static_cast<int>(current_stream.device_index()) << " ";
  file << static_cast<int>(current_stream.device_type()) << " ";
  file << (current_stream == explicit_stream) << " ";
  file << (current_stream == default_stream) << " ";
  file.saveFile();
}
```

---

## 输出对比

| 测试用例 | Paddle 输出 | Torch 输出 |
|---------|------------|------------|
| GetDeviceProperties | 已进入常规 `result_cmp`；比较稳定设备属性摘要 | 已进入常规 `result_cmp`；比较同一组摘要 |
| GetCurrentDeviceProperties | 已进入常规 `result_cmp`；比较稳定设备属性摘要 | 已进入常规 `result_cmp`；比较同一组摘要 |
| GetCurrentCUDAStream | 已进入常规 `result_cmp`；比较稳定 stream 摘要 | 已进入常规 `result_cmp`；比较同一组摘要 |

---

## 结论

`ATen/cuda/CUDAContext.h` 已在 compat 提供。当前这组接口不再属于“未纳入常规回归”的问题，本文档保留该节仅作为历史记录，说明它已经转入常规回归路径。

---


---

# CUDADataTypeTest

## 差异点列表

1. **`ScalarTypeToCudaDataType(Bool)` 支持范围不同**：Paddle compat 不支持 `Bool` 转 `cudaDataType`，会抛出异常；Torch 侧接口支持范围更完整。当前测试已跳过 `Bool`。
2. **`empty_cuda` 结果依赖运行时/构建环境**：Torch CUDA 版通常可成功创建 CUDA Tensor；Paddle compat 在未编译 CUDA 或运行时不可用时会抛异常并进入不可用分支。该差异属于环境差异，不属于接口语义差异。

---

## Diff 测试用例位置

测试文件：`test/ATen/cuda/CUDADataTypeTest.cpp`

### 测试用例原文

```cpp
// 测试 ScalarTypeToCudaDataType 对 Bool 的支持
TEST_F(CUDADataTypeTest, GetCudaDataType) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();

  // 测试 Bool - [DIFF] Paddle 不支持，会抛出异常
  // file << std::to_string(
  //     at::cuda::ScalarTypeToCudaDataType(c10::ScalarType::Bool)) << " "; // [DIFF]

  file << "cuda_type_test ";
  file.saveFile();
}

// 测试 empty_cuda
TEST_F(CUDADataTypeTest, EmptyCUDA) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();

  try {
    at::Tensor t = at::empty_cuda({2, 3}, at::TensorOptions().dtype(at::kFloat));
    file << "cuda_empty ";
  } catch (const std::exception& e) {
    // Paddle 非 GPU 版或 CUDA 不可用时会抛异常
    file << "cuda_not_available ";
  }
  file.saveFile();
}
```

---

## 输出对比

| 测试用例 | Paddle 输出 | Torch 输出 |
|---------|------------|------------|
| GetCudaDataType | `cuda_type_test` | 正常输出（包含 Bool） |
| EmptyCUDA | `cuda_not_available` | `cuda_empty` |

---

## 初步问题分析

1. **ScalarTypeToCudaDataType(Bool)**：Paddle 未实现 Bool 到 cudaDataType 的转换，会抛出异常。

2. **empty_cuda**：属于运行时环境差异，取决于 Paddle 是否编译了 CUDA 支持。

---
