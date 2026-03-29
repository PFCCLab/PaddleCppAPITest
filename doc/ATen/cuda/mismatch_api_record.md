# CUDA Context（`at::cuda::getCurrentCUDAStream` 测试路径被跳过）

> Paddle 头文件：`ATen/cuda/CUDAContext.h`

## 差异点列表

1. **常规回归中该项未执行**：虽然 `unmatch_CUDAContextTest.cpp` 内已改为双端调用，但它不在常规构建集合，`result_cmp` 默认不会比较该项。

---

## Diff 测试用例位置

测试文件：`test/ATen/cuda/unmatch_CUDAContextTest.cpp`

### 测试用例原文

```cpp
// 测试 getCurrentCUDAStream
TEST_F(CUDAContextTest, GetCurrentCUDAStream) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();

try {
  auto stream = at::cuda::getCurrentCUDAStream();
  (void)stream;
  file << "stream_available ";
} catch (...) {
  file << "stream_not_available ";
}
  file.saveFile();
}
```

---

## 输出对比

| 测试用例 | Paddle 输出 | Torch 输出 |
|---------|------------|------------|
| GetCurrentCUDAStream | 不在常规 `result_cmp` 集合中 | 不在常规 `result_cmp` 集合中 |

---

## 初步问题分析

`ATen/cuda/CUDAContext.h` 已在 compat 提供。当前问题不是“接口缺失”，而是该节仍按 `unmatch` 管理，未纳入常规回归。

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
