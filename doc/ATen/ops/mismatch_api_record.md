# SparseTensor

> Paddle 头文件：`ATen/ops/sparse_coo_tensor.h`、`ATen/ops/sparse_csr_tensor.h`
>
> 2026-03-28 复核：`SparseCOOInferSize` 已对齐。当前 `Paddle` 与 `Torch` 输出均为 `2 3 3`。

## 历史差异

1. **sparse_coo_tensor 无 size 推断行为**：历史上 `Paddle` compat 在不显式传入 `size` 时，曾把首个维度错误推断为 `0`。

---

## 当前回归用例位置

测试文件：`test/ATen/ops/SparseTensorTest.cpp`

### 当前测试用例原文

```cpp
TEST_F(SparseTensorTest, SparseCOOInferSize) {
  auto idx_data = create_tensor_from_list({0L, 1L, 2L, 1L, 2L, 0L});
  at::Tensor indices = idx_data.reshape({2, 3});
  at::Tensor values = create_tensor_from_float_list({5.0f, 6.0f, 7.0f});
  at::Tensor sparse = at::sparse_coo_tensor(indices, values);

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "SparseCOOInferSize ";
  write_sparse_info_to_file(&file, sparse);
  file << "\n";
  file.saveFile();
}
```

---

## 输出对比

| 测试用例 | Paddle 输出 | Torch 输出 |
|---------|------------|------------|
| SparseCOOInferSize（2026-03-28 复核） | `2 3 3` | `2 3 3` |
| SparseCOOInferSize（历史） | `0 2 2` | `2 2 2` |

---

## 复核结论

当前 compat 中 `ATen/ops/sparse_coo_tensor.h` 已能按 indices 正确推断 shape。为避免该行为再次回退，`/home/may/Paddle/test/cpp/compat/c10_layout_test.cc` 已补充 infer-size 的 shape 断言。

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
>
> 2026-03-28 复核：`NotEqualDtype` 已对齐。当前 `Paddle` 与 `Torch` 输出均为 `1`。

## 历史差异

1. **数据类型不同时的比对行为**：历史上 `Paddle` compat 的某些路径曾在异 dtype 比较时抛出类型检查异常。

---

## 当前回归用例位置

测试文件：`test/ATen/ops/EqualTest.cpp`

### 当前测试用例原文

```cpp
TEST_F(EqualTest, NotEqualDtype) {
  at::Tensor t1 = at::zeros({4}, at::kFloat);
  at::Tensor t2 = at::zeros({4}, at::kInt);

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "NotEqualDtype ";
  try {
    bool result = t1.equal(t2);
    write_bool_result_to_file(&file, result);
  } catch (const std::exception& e) {
    file << "exception: " << e.what();
  }
  file << "\n";
  file.saveFile();
}
```

---

## 输出对比

| 测试用例 | Paddle 输出 | Torch 输出 |
|---------|------------|------------|
| NotEqualDtype（2026-03-28 复核） | `1` | `1` |
| NotEqualDtype（历史） | 历史观测：抛出异常 | 历史观测：`false` |

---

## 复核结论

当前 `EqualTest.NotEqualDtype` 在两侧都返回 `true`，该 case 不再是稳定差异。当前残留差异仍主要集中在异常路径的 stack trace 完整性，而不是异 dtype 的比较结果本身。

---

# Select

> Paddle 头文件：`ATen/ops/select.h`
>
> 2026-03-28 复核：`SelectNegativeDim` 已对齐。当前 `Paddle` 与 `Torch` 输出均为 `1 3 0.000000 1.000000 2.000000`。

## 历史差异

1. **支持负数维度的表现**：历史上 `Paddle` compat 在 `select(-1, ...)` 路径上曾出现崩溃。

---

## 当前回归用例位置

测试文件：`test/ATen/ops/SelectTest.cpp`

### 当前测试用例原文

```cpp
TEST_F(SelectTest, SelectNegativeDim) {
  at::Tensor t1 = at::zeros({3, 3}, at::kFloat);
  float* data = t1.data_ptr<float>();
  for (int i = 0; i < 9; ++i) {
    data[i] = static_cast<float>(i);
  }

  at::Tensor result = t1.select(-1, 0);

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "SelectNegativeDim ";
  write_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}
```

---

## 输出对比

| 测试用例 | Paddle 输出 | Torch 输出 |
|---------|------------|------------|
| SelectNegativeDim（2026-03-28 复核） | `1 3 0.000000 1.000000 2.000000` | `1 3 0.000000 1.000000 2.000000` |
| SelectNegativeDim（历史） | 历史观测：崩溃 (SIGABRT) | 历史观测：正常返回 Tensor |

---

## 复核结论

当前负维 `select` 路径已经正常工作。`SelectTest` 里仍可观察到的差异主要是 `SelectException` 异常输出缺少 Torch 侧的 C++ stack trace，而不是负维行为本身。

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
