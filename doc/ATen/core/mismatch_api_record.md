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

# Tensor::resize_（已对齐）

> Paddle 相关头文件：`ATen/core/TensorBody.h`、`ATen/ops/resize.h`
> 状态：基础 `resize_` 语义已对齐（2026-03-30）

## Diff 测试用例位置

测试文件：`test/ATen/core/TensorTest.cpp`

### 测试用例原文

```cpp
// 测试 resize_ - 缩小元素数时应成功并保留前缀数据
TEST_F(TensorTest, Resize) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "Resize ";
  tensor.resize_({4, 5});
  file << std::to_string(tensor.sizes()[0]) << " ";
  file << std::to_string(tensor.sizes()[1]) << " ";
  file << std::to_string(tensor.numel()) << " ";
  file << std::to_string(tensor.data_ptr<float>()[0]) << " ";
  file << std::to_string(tensor.data_ptr<float>()[19]) << " ";
  file << "\n";
  file.saveFile();
}
```

---

## 输出对比

| 测试用例 | Paddle 输出 | Torch 输出 |
|---------|------------|------------|
| Resize | `4 5 20 1.000000 1.000000` | `4 5 20 1.000000 1.000000` |

---

## 当前行为

当前 compat `resize_` 已改为混合实现：元素总数不变时走 `reshape`，元素总数变化时走 Paddle 原生 `set_` 路径，因此覆盖了当前 diff 用例中 `2x3x4 -> 4x5` 的缩容场景，也不会破坏连续多次 `resize_()` 的稳定性。

当前范围：

1. 支持元素总数变化的 `resize_()` 调用，不再退化为只能 `reshape`。
2. 现有对比用例验证了缩容后 shape、`numel()` 和前缀数据保留行为。
3. `memory_format` 目前仅覆盖 `nullopt` / `Contiguous` 路径，这与当前仓里的使用方式一致。

备注：

- 本节原先“不支持、会抛异常”的结论已失效，排查时请以现有实现和测试结果为准。

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
