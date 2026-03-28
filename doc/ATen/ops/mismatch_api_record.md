# SparseTensor

> Paddle 头文件：`ATen/ops/sparse_coo_tensor.h`、`ATen/ops/sparse_csr_tensor.h`

## 差异点列表

1.  **sparse_coo_tensor 无 size 推断行为**：PyTorch 能根据 indices 内容正确推断完整 size（如 `2 2 2`）；Paddle 推断结果第一个维度为 0（如 `0 2 2`）

---

## Diff 测试用例位置

测试文件：`test/ATen/ops/SparseTensorTest.cpp`

### 测试用例原文

```cpp
// COO 带推断 size
TEST_F(SparseTensorTest, SparseCOOInferSize) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();

  // indices: [2, 3] -> values: [3]
  at::Tensor indices = at::tensor({{0, 1, 2}, {0, 1, 2}}, at::kLong);
  at::Tensor values = at::tensor({1.0, 2.0, 3.0}, at::kFloat);

  // 不指定 size，让框架推断
  at::Tensor sparse = at::sparse_coo_tensor(indices, values);

  file << std::to_string(sparse.size(0)) << " ";
  file << std::to_string(sparse.size(1)) << " ";
  file << std::to_string(sparse.size(2)) << " ";

  file.saveFile();
}
```

---

## 输出对比

| 测试用例 | Paddle 输出 | Torch 输出 |
|---------|------------|------------|
| SparseCOOInferSize | `0 2 2` | `2 2 2` |

---

## 初步问题分析

Paddle 在使用 sparse_coo_tensor(indices, values) 不指定 size 参数时，无法正确推断第一个维度的大小，会返回 0；而 PyTorch 能正确推断为 2。

---


---

# TensorFactoryTest

## 差异点列表

1. **ScalarType::Bool 枚举值不同**：Paddle 的 DataType::BOOL = 10，Torch 的 ScalarType::Bool = 11。

---

## Diff 测试用例位置

测试文件：`test/ATen/ops/TensorFactoryTest.cpp`

### 测试用例原文

```cpp
// 测试从 Bool 数组创建 Tensor
TEST_F(TensorFactoryTest, TensorFromBoolArrayRef) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();

  std::vector<bool> bool_data = {true, false, true};
  at::Tensor t = at::tensor(bool_data);

  // [DIFF] Paddle: scalar_type = 10 (DataType::BOOL)
  // Torch: scalar_type = 11 (ScalarType::Bool)
  // file << std::to_string(static_cast<int>(t.scalar_type())) << " "; // [DIFF]

  file << std::to_string(t.dim()) << " ";
  file << std::to_string(t.numel()) << " ";
  file.saveFile();
}
```

---

## 输出对比

| 测试用例 | Paddle 输出 | Torch 输出 |
|---------|------------|------------|
| TensorFromBoolArrayRef | `1 3` | `1 3` |

（注：scalar_type 字段已注释，仅对比其他字段）

---

## 初步问题分析

Paddle 与 PyTorch 的 ScalarType::Bool 枚举值不同：Paddle = 10，Torch = 11。这是两个框架设计上的差异。

---


---

# Equal

> Paddle 头文件：`ATen/ops/equal.h`

## 差异点列表

1. **数据类型不同时的比对行为**：Torch在比对类型不一致的Tensor时会静默返回false，不触发任何错误；而Paddle在尝试比对时会在底层抛出类型检查不匹配（例如要求int32但接收到了float32）的C++异常甚至崩溃。

---

## Diff 测试用例位置

测试文件：`test/ATen/ops/EqualTest.cpp`

### 测试用例原文

```cpp
// [DIFF] Test paddle equal exception when comparing tensors of different types
// Torch returns false without checking specific data types, whereas Paddle throws:
// "The type of data we are trying to retrieve (int32) does not match the type of data (float32)..."
TEST_F(EqualTest, NotEqualDtype) {
  /*
  at::Tensor t1 = at::zeros({4}, at::kFloat);
  at::Tensor t2 = at::zeros({4}, at::kInt);

  bool result = t1.equal(t2);

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  write_bool_result_to_file(&file, result);
  file.saveFile();
  */
}
```

---

## 输出对比

| 测试用例 | Paddle 输出 | Torch 输出 |
|---------|------------|------------|
| NotEqualDtype | 历史观测：抛出异常 | 历史观测：`false` |

---

## 初步问题分析

该差异是历史实测结论；当前 case 已禁用，暂不参与常规回归对比。

---

# Select

> Paddle 头文件：`ATen/ops/select.h`

## 差异点列表

1. **支持负数维度的表现**：Torch支持传入负数维（如-1代表最后一维）进行选取；而Paddle在使用 -1 时可能会引发底层的 double free or corruption (out) 崩溃引发SIGABRT。

---

## Diff 测试用例位置

测试文件：`test/ATen/ops/SelectTest.cpp`

### 测试用例原文

```cpp
// [DIFF] Paddle select with negative dim causes double free or corruption SIGABRT
TEST_F(SelectTest, SelectNegativeDim) {
  /*
  at::Tensor t1 = at::zeros({3, 3}, at::kFloat);
  float* data = t1.data_ptr<float>();
  for (int i = 0; i < 9; ++i) {
    data[i] = static_cast<float>(i);
  }

  at::Tensor result = t1.select(-1, 0);

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  write_result_to_file(&file, result);
  file.saveFile();
  */
}
```

---

## 输出对比

| 测试用例 | Paddle 输出 | Torch 输出 |
|---------|------------|------------|
| SelectNegativeDim | 历史观测：崩溃 (SIGABRT) | 历史观测：正常返回 Tensor |

---

## 初步问题分析

该差异是历史复现结论；当前为保持回归稳定性已禁用该 case。

---


---

# Empty（`at::empty` CUDA 场景）

> Paddle 头文件：`ATen/ops/empty.h`

## 差异点列表

1. **CUDA 结果受运行环境影响**：当前环境中 Paddle compat 可创建 CUDA Tensor，而 Torch 侧进入 `cuda_not_available` 分支。

---

## Diff 测试用例位置

测试文件：`test/ATen/ops/EmptyOpsTest.cpp`

### 测试用例原文

```cpp
TEST_F(EmptyOpsTest, EmptyCUDA) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();

  // Try to create empty CUDA tensor
  try {
    at::Tensor t = at::empty({2, 3}, at::TensorOptions().device(at::kCUDA));
    // [DIFF] 当前环境下 Paddle compat 可成功创建 CUDA Tensor，而 Torch 侧进入不可用分支。
    // [DIFF] 该差异受运行时/构建环境影响，不属于 empty 接口的稳定语义差异，因此不参与结果序列化。
    (void)t;
  } catch (...) {
  }
  file.saveFile();
}
```

---

## 输出对比

| 测试用例 | Paddle 输出 | Torch 输出 |
|---------|------------|------------|
| EmptyCUDA（原始） | `cuda_tensor` | `cuda_not_available` |

---

## 初步问题分析

该差异与二进制构建方式和运行环境（CUDA 可用性）强相关，不属于 `at::empty` 的稳定接口语义差异。测试已保留调用路径，但不再序列化该字段以避免伪 diff。

---
