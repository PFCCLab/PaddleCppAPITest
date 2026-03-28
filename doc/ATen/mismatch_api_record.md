# at::indexing（Slice / EllipsisIndexType）

> Paddle 头文件：`ATen/TensorIndexing.h`

## 差异点列表

- [x] **头文件路径不同**：PyTorch 为 `ATen/TensorIndexing.h`；Paddle compat 为 `ATen/indexing.h`
- [ ] **`Tensor::operator[](Slice)` 不支持**：PyTorch 的 `Tensor::operator[]` 接受 `at::indexing::Slice`；Paddle compat 的 `operator[]` 仅重载 `int64_t`，传入 `Slice` 会编译报错
- [x] **多维 Slice 索引写法不同**：
    - PyTorch：`t.index({Slice(0,2), Slice(1,3)})` —— 接受 `std::initializer_list<TensorIndex>`
    - Paddle：`t.index(std::vector<at::indexing::Slice>{Slice(0,2), Slice(1,3)})` —— 仅重载 `std::vector<Slice>`
4.  **`TensorIndex` 能力对齐状态**：Paddle compat 已提供 `TensorIndex`，但 `Tensor::operator[](Slice)` 仍不支持，实际使用仍需通过 `index(...)` 路径。

---

## Diff 测试用例位置

测试文件：`test/ATen/IndexingTest.cpp`

### 测试用例原文

```cpp
// Test using indexing with tensors
TEST_F(IndexingTest, TensorIndexing) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();

  at::Tensor t = at::arange(12, at::kInt).view({3, 4});

  // 【API 差异】Paddle compat 的 Tensor::operator[] 仅重载 int64_t，不支持传入
  // Slice；须改用 index(std::vector<at::indexing::Slice>) 接口。PyTorch 支持
  // operator[](Slice) 及 index({Slice, ...}) 两种写法。
#if USE_PADDLE_API
  at::Tensor result = t.index(std::vector<at::indexing::Slice>{
      at::indexing::Slice(), at::indexing::Slice()});
#else
  at::Tensor result = t.index({at::indexing::Slice(), at::indexing::Slice()});
#endif
  file << std::to_string(result.dim()) << " ";
  file << std::to_string(result.numel()) << " ";
  file.saveFile();
}

// Test Slice indexing
TEST_F(IndexingTest, SliceIndexing) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();

  at::Tensor t = at::arange(12, at::kInt).view({3, 4});

  // 【API 差异】同上：Paddle 不支持链式 operator[](Slice)，
  // 多维 Slice 须放入同一个 std::vector<Slice> 传给 index()；
  // PyTorch 可用 index({Slice(0,2), Slice(1,3)}) 的 initializer_list 写法。
#if USE_PADDLE_API
  at::Tensor result = t.index(std::vector<at::indexing::Slice>{
      at::indexing::Slice(0, 2), at::indexing::Slice(1, 3)});
#else
  at::Tensor result =
      t.index({at::indexing::Slice(0, 2), at::indexing::Slice(1, 3)});
#endif
  file << std::to_string(result.dim()) << " ";
  file << std::to_string(result.size(0)) << " ";
  file << std::to_string(result.size(1)) << " ";
  file.saveFile();
}
```

---

## 输出对比

| 测试用例 | Paddle 输出 | Torch 输出 |
|---------|------------|------------|
| TensorIndexing | `2 12` | `2 12` |
| SliceIndexing | `2 2 3` | `2 2 3` |

（注：输出相同，但调用方式不同）

---

## 初步问题分析

1. **头文件路径**：PyTorch 使用 `ATen/TensorIndexing.h`，Paddle 使用 `ATen/indexing.h`。

2. **operator[] 不支持 Slice**：Paddle 的 Tensor::operator[] 仅重载了 int64_t，不支持传入 Slice，需要使用 index() 方法。

3. **多维 Slice 写法**：PyTorch 支持 `t.index({Slice, Slice})` 写法（initializer_list），Paddle 只能使用 `t.index(std::vector<Slice>{})`。

4. **`TensorIndex` 已具备，但索引入口仍有差异**：`TensorIndex` 已在 compat 中实现；主要差异仍是 `operator[](Slice)` 与部分索引入口写法。

---


---

# DeviceGuard

> Paddle 头文件：`ATen/DeviceGuard.h`

## 差异点列表

1.  **`device_of(tensor)` 的索引语义存在历史差异**：PyTorch 默认 CPU 设备常表现为 `index=-1`、`has_index=false`，Paddle 常规范化到 `cpu:0`。当前测试已避免直接序列化 `index/has_index` 字段。

---

## Diff 测试用例位置

测试文件：`test/ATen/DeviceGuardTest.cpp`

### 测试用例原文

```cpp
static void write_device_result_to_file(FileManerger* file,
                                        const std::optional<at::Device>& dev) {
  if (dev.has_value()) {
    // [DIFF] index/has_index 在两端语义不同，当前仅比较 type
    *file << dev->type() << " ";
  } else {
    *file << "nullopt ";
  }
}
```

---

## 输出对比

| 测试用例 | Paddle 输出 | Torch 输出 |
|---------|------------|------------|
| DeviceOfTensor | 仅比较 `type` 字段 | 仅比较 `type` 字段 |

---

## 初步问题分析

历史上该差异存在于 `index/has_index` 字段；当前测试策略已切换为只比较 `device.type()`，避免将设备表示差异放大为回归失败。

---
