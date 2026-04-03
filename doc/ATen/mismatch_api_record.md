# Utils（历史差异：原 `unmatch` 直接调用 `at::detail::*`）

> Paddle 头文件：`ATen/Utils.h`、`ATen/ops/tensor.h`
> 状态：已纳入常规回归（2026-03-30）

当前 `ATen/Utils` 相关构造入口已经纳入默认构建与 `result_cmp` 比对：

1. 原 `test/ATen/unmatch_UtilsTest.cpp` 已迁移为常规测试文件 `test/ATen/UtilsTest.cpp`。
2. 常规回归不再直接比较 `at::detail::tensor_cpu`、`tensor_backend`、`tensor_complex_*` 这些内部 helper 模板符号，而是改为通过公开入口 `at::tensor(ArrayRef<T>, TensorOptions)` 与 `at::tensor(ArrayRef<c10::complex<double>>, TensorOptions)` 覆盖 `ATen/Utils.cpp` 的实现路径。
3. `TensorBackend` 与 `TensorComplexBackend` 现在使用显式 `device(CUDA)` 选项直接尝试 backend 分支；若当前运行时路径不可用，则两端统一回写 `cuda_runtime_unavailable`。
4. 当前环境下 `/tmp/paddle_cpp_api_test/paddle_UtilsTest.txt` 与 `/tmp/paddle_cpp_api_test/torch_UtilsTest.txt` 输出一致。

验证位置：

- `test/ATen/UtilsTest.cpp`

### 当前测试片段

```cpp
TEST(UtilsTest, TensorCPU) {
  std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
  at::ArrayRef<float> arr(data);
  at::Tensor t1 = at::tensor(arr, at::TensorOptions(at::kFloat));
  WriteTensorResult(&file, t1);
}

TEST(UtilsTest, TensorComplexCPU) {
  std::vector<c10::complex<double>> data = {{1.0, 2.0}, {3.0, 4.0}};
  at::ArrayRef<c10::complex<double>> arr(data);
  at::Tensor t1 = at::tensor(arr, at::TensorOptions(at::kComplexDouble));
  WriteComplexTensorResult(&file, t1);
}
```

### 输出对比

| 测试用例 | Paddle 输出 | Torch 输出 |
|---------|------------|------------|
| `TensorCPU` | `1 4 1 2 3 4 1 0` | `1 4 1 2 3 4 1 0` |
| `TensorBackend` | `cuda_runtime_unavailable` | `cuda_runtime_unavailable` |
| `TensorComplexCPU` | `1 2 1 2 3 4` | `1 2 1 2 3 4` |
| `TensorComplexBackend` | `cuda_runtime_unavailable` | `cuda_runtime_unavailable` |

备注：

- 原 `unmatch_UtilsTest.cpp` 中直接调用 `at::detail::*` 的写法不再作为外部回归目标；当前回归聚焦公开 `at::tensor` 构造入口。
- `TensorCPU` 结果中尾部的 `1 0` 对应空 `ArrayRef<float>` 构造结果：`dim=1`、`numel=0`。

---

# at::indexing（Slice / TensorIndex，已对齐）

> Paddle 头文件：`ATen/TensorIndexing.h`
> 状态：已对齐（2026-03-30）

当前 compat 与 PyTorch C++ API 的索引入口已经一致：

1. 头文件入口统一为 `ATen/TensorIndexing.h`。
2. `Tensor::index(std::initializer_list<TensorIndex>)` 可直接用于单维和多维 `Slice` 索引。
3. `TensorIndex` 的 `None / Ellipsis / Integer / Boolean / Slice / Tensor` 构造与判定接口可直接复用 PyTorch 写法。
4. `TensorIndex::integer()` 当前两端都走 `c10::SymInt` 语义；测试中通过统一辅助函数回写数值。
5. `Tensor::operator[](Slice)` 不是当前 PyTorch C++ API 的标准入口；两端统一建议使用 `index({Slice(...)})`。

验证位置：

- `test/ATen/IndexingTest.cpp`
- `test/ATen/ops/IndexTest.cpp`

### 当前测试片段

```cpp
TEST_F(IndexingTest, TensorIndexing) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "TensorIndexing ";

  at::Tensor t = at::arange(12, at::kInt).view({3, 4});
  at::Tensor result = t.index({at::indexing::Slice()});

  file << std::to_string(result.dim()) << " ";
  file << std::to_string(result.numel()) << " ";
  file << "\n";
  file.saveFile();
}

TEST_F(IndexingTest, SliceIndexing) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "SliceIndexing ";

  at::Tensor t = at::arange(12, at::kInt).view({3, 4});
  at::Tensor result =
      t.index({at::indexing::Slice(0, 2), at::indexing::Slice(1, 3)});

  file << std::to_string(result.dim()) << " ";
  file << std::to_string(result.size(0)) << " ";
  file << std::to_string(result.size(1)) << " ";
  file << "\n";
  file.saveFile();
}
```

### 输出对比

| 测试用例 | Paddle 输出 | Torch 输出 |
|---------|------------|------------|
| `TensorIndexing` | `2 12` | `2 12` |
| `SliceIndexing` | `2 2 2` | `2 2 2` |

备注：

- 旧文档中关于 `ATen/indexing.h`、`std::vector<Slice>` 专用入口以及 `operator[](Slice)` 差异的结论已过时。
- 当前 `/tmp/paddle_cpp_api_test/paddle_IndexingTest.txt` 与 `/tmp/paddle_cpp_api_test/torch_IndexingTest.txt` 输出一致。

---

# DeviceGuard（`device_of`，已对齐）

> Paddle 头文件：`ATen/DeviceGuard.h`
> 状态：已对齐（2026-03-30）

当前 compat 的 `device_of` 行为与 PyTorch 已保持一致：

1. `device_of(const Tensor&)` 对已定义 CPU Tensor 返回 `cpu:-1`。
2. `device_of(const std::optional<Tensor>&)` 对 `nullopt` 返回 `nullopt`，对有值 CPU Tensor 返回 `cpu:-1`。
3. 当前测试已直接比较 `type + index`，不再需要“只比较 type”的回避策略。

验证位置：

- `test/ATen/DeviceGuardTest.cpp`

### 当前测试片段

```cpp
static void write_device_result_to_file(FileManerger* file,
                                        const std::optional<at::Device>& dev) {
  if (dev.has_value()) {
    *file << dev->type() << " " << static_cast<int>(dev->index()) << " ";
  } else {
    *file << "nullopt ";
  }
}
```

### 输出对比

| 测试用例 | Paddle 输出 | Torch 输出 |
|---------|------------|------------|
| `DeviceOfTensor` | `cpu -1 cpu -1` | `cpu -1 cpu -1` |
| `DeviceOfOptionalTensor` | `nullopt cpu -1` | `nullopt cpu -1` |

备注：

- 旧文档中关于 `device_of(tensor)` 会被规范化为 `cpu:0` 的结论已失效。
- 当前 `/tmp/paddle_cpp_api_test/paddle_DeviceGuardTest.txt` 与 `/tmp/paddle_cpp_api_test/torch_DeviceGuardTest.txt` 输出一致。

---
