# IValue（已对齐）

> Paddle 头文件：`ATen/core/ivalue.h`
> 状态：已对齐（2026-03-28）

当前 compat `IValue` 已补齐以下 PyTorch 风格接口：

1. `c10::IValue` 入口，可直接按 PyTorch 方式引用。
2. camelCase 方法：`isNone()`、`isBool()`、`isInt()`、`isDouble()`、`isString()`、`isList()`、`isTensor()`、`isCustomClass()`、`isTuple()`。
3. 提取方法：`toBool()`、`toInt()`、`toDouble()`、`toStringRef()`、`toStringView()`、`toTensor()`、`toScalarType()`。
4. 调试接口：`tagKind()`。

验证位置：

- `test/ATen/core/IValueTest.cpp`
- `test/torch/LibraryTest.cpp`

说明：

- `torch::IValue` 兼容入口仍保留，便于已有调用方平滑迁移。
- 该节原先记录的命名空间、camelCase 命名、`tagKind()`、`toStringRef()` 差异已不再是当前阻塞项。


---

# Tensor::resize_（Paddle 不支持）

> Paddle 头文件：`ATen/core/Tensor.h`

## Diff 测试用例位置

测试文件：`test/ATen/core/TensorTest.cpp`

### 测试用例原文

```cpp
// 测试 resize_ - Paddle不支持，会抛出异常
TEST_F(TensorTest, Resize) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();
  try {
    tensor.resize_({4, 5});
    file << "0 ";
  } catch (const std::exception& e) {
    file << "1 ";
  }
  file.saveFile();
}
```

---

## 输出对比

| 测试用例 | Paddle 输出 | Torch 输出 |
|---------|------------|------------|
| Resize | `1`（抛出异常） | `0`（成功） |

---

## 初步问题分析

Paddle 不支持 Tensor::resize_() 方法，调用时会抛出异常；PyTorch 完整支持原地调整 tensor 形状。

---

# Tensor::pin_memory / is_pinned（已对齐）

> Paddle 相关头文件：`ATen/core/TensorBody.h`
> 状态：已按 PyTorch 语义对齐（2026-03-28）

当前行为：

1. `pin_memory()` 仅接受 CPU Tensor，非 CPU Tensor 直接报错。
2. `is_pinned()` 仅对 pinned host tensor 返回 true。
3. `device` 形参保留兼容入口，但按 PyTorch 语义视为 deprecated。

验证位置：

- `test/ATen/core/TensorTest.cpp`

备注：

- 该节原先记录的是历史差异，当前实现与文档旧结论不一致，排查时请以现有实现和测试结果为准。

---


---

# Tensor 指针 API（`const_data_ptr<T>` / `mutable_data_ptr<T>`，已对齐）

> Paddle 头文件：`ATen/core/TensorBody.h`
> 状态：已对齐（2026-03-28）

当前状态：

1. 模板版本 `const_data_ptr<T>()` / `mutable_data_ptr<T>()` 已可正常链接和调用。
2. `test/ATen/ops/TensorPtrTest.cpp` 已恢复直接验证 `float*` / `const float*` 路径。

验证位置：

- `test/ATen/ops/TensorPtrTest.cpp`

备注：

- 本节保留作为历史记录，旧的 `undefined reference` 结论不再适用。

---
