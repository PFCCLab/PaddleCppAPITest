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

# CUDADataType（历史差异：旧文档曾误记 `Bool` 支持范围）

> Paddle 头文件：`ATen/cuda/CUDADataType.h`、`ATen/cuda/EmptyTensor.h`

## 当前状态

1. `ScalarTypeToCudaDataType` 针对当前 compat 暴露的 `Float`、`Double`、`Int`、`Long`、`Half`、`Byte`、`Char`、`Short`、`BFloat16`、`ComplexFloat`、`ComplexDouble` 已纳入常规回归，当前输出与 Torch 一致。
2. `Bool` 并不是 Torch 侧单独支持的类型；两端调用 `ScalarTypeToCudaDataType(c10::ScalarType::Bool)` 都会抛异常。当前测试已改为显式记录 `bool_unsupported`，不再把它记为 Paddle 单边差异。
3. `EmptyCUDA` / `EmptyCudaDifferentDtype` 的输出取决于当前 CUDA 运行时是否可用。在同一台机器上执行 `result_cmp` 时，两端会进入相同分支：有 CUDA 运行时时输出 `cuda_empty` / `cuda_empty_int`，无 CUDA 运行时时输出 `cuda_not_available`。当前环境下两端均输出 `cuda_not_available`。
4. 当前 compat `c10::ScalarType` 侧尚未暴露 `ComplexHalf` / `Float4_e2m1fn_x2`，因此这两项暂未纳入本测试覆盖；本节结论仅针对当前已暴露的 scalar type 子集。

---

## Diff 测试用例位置

测试文件：`test/ATen/cuda/CUDADataTypeTest.cpp`

### 当前测试策略

```cpp
TEST_F(CUDADataTypeTest, GetCudaDataType) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();

  try {
    (void)at::cuda::ScalarTypeToCudaDataType(c10::ScalarType::Bool);
    file << "bool_supported ";
  } catch (...) {
    file << "bool_unsupported ";
  }

  file.saveFile();
}

TEST_F(CUDADataTypeTest, EmptyCUDA) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();

  try {
    at::Tensor t = at::detail::empty_cuda({2, 3, 4},
                                          c10::ScalarType::Float,
                                          at::Device(at::kCUDA, 0),
                                          std::nullopt);
    (void)t;
    file << "cuda_empty ";
  } catch (...) {
    file << "cuda_not_available ";
  }
  file.saveFile();
}

TEST_F(CUDADataTypeTest, EmptyCudaDifferentDtype) {
  ...
  try {
    at::Tensor t = at::detail::empty_cuda(
        {2, 3}, c10::ScalarType::Int, at::Device(at::kCUDA, 0), std::nullopt);
    (void)t;
    file << "cuda_empty_int ";
  } catch (...) {
    file << "cuda_not_available ";
  }
  file.saveFile();
}
```

---

## 输出对比

| 测试用例 | Paddle 输出 | Torch 输出 |
|---------|------------|------------|
| GetCudaDataType | 已进入常规 `result_cmp`；当前输出包含 `bool_unsupported` 标记 | 已进入常规 `result_cmp`；当前输出包含 `bool_unsupported` 标记 |
| GetCudaDataTypeBFloat16 | `14` | `14` |
| GetCudaDataTypeComplex | `4 5` | `4 5` |
| EmptyCUDA | 同一运行环境下与 Torch 进入同一分支；当前环境为 `cuda_not_available` | 同左 |
| EmptyCudaDifferentDtype | 同一运行环境下与 Torch 进入同一分支；当前环境为 `cuda_not_available` | 同左 |

---

## 结论

`ATen/cuda/CUDADataType.h` 针对当前 compat 暴露的标量类型集合已与 Torch 保持一致回归，`CUDADataTypeTest` 不再是当前未对齐项。本文档保留该节仅作为历史记录，说明旧结论已回写。
