# BFloat16

> Paddle 头文件：`c10\util\BFloat16.h`

## 差异点列表

1.  **BFloat16 ScalarType 枚举值**：PyTorch 为 **11**，Paddle 为 **15**
2.  **ComplexFloat ScalarType 枚举值**：PyTorch 为 **8**，Paddle 为 **9**

---

## Diff 测试用例位置

测试文件：`test/c10/util/HalfBFloat16Test.cpp`

### 测试用例原文

```cpp
// ScalarType 对应关系
// [DIFF] PyTorch输出: 5 11, PaddlePaddle输出: 5 15 (BFloat16枚举值不同)
TEST_F(HalfBFloat16Test, ScalarTypeMapping) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << std::to_string(static_cast<int>(at::kHalf)) << " ";
  // file << std::to_string(static_cast<int>(at::kBFloat16)) << " "; // [DIFF]
  file.saveFile();
}
```

---

## 输出对比

| 字段 | Paddle 输出 | Torch 输出 |
|------|------------|------------|
| kHalf | 5 | 5 |
| kBFloat16 | 15 | 11 |

---

## 初步问题分析

Paddle 与 PyTorch 的 ScalarType 枚举值定义不同：BFloat16 在 PyTorch=11，Paddle=15；ComplexFloat 在 PyTorch=8，Paddle=9。这是两个框架设计上的差异，需要在兼容层进行映射对齐。

---


---

# OptionalArrayRef

> Paddle 头文件：`c10\util\OptionalArrayRef.h`

## 差异点列表

1.  **运行时内存地址值**：两框架输出的内存地址不同（属正常运行时差异，不影响功能）
2.  **内部对象标识符**：两框架内部唯一标识符数值不同（属正常实现差异，不影响功能）
3.  **FromOptionalArrayRef 临时对象悬空引用**：
    `std::optional<c10::ArrayRef<int64_t>>(std::vector<int64_t>{...})`
    让 `ArrayRef` 指向临时 vector，`front()` 输出随机内存值，Torch/Paddle diff。
    已按测试规范在 `OptionalArrayRefTest.cpp` 添加 `DIFF` 标注并注释该不稳定输出字段，仅保留 `has_value/size`。

> 注：OptionalArrayRef 核心功能（has_value、size、元素访问、reset、swap、emplace、slice 等）在两个框架中完全兼容，仅运行时地址和标识符存在差异。

---

## Diff 测试用例位置

测试文件：`test/c10/util/OptionalArrayRefTest.cpp`

### 测试用例原文

```cpp
// DIFF: std::vector<int64_t>{1, 2, 3, 4, 5} 是临时对象，传入 OptionalArrayRef
// 在语句结束后被销毁， OptionalArrayRef 内部 ArrayRef
// 指向的内存已释放，继续访问会导致未定义行为
TEST_F(OptionalArrayRefTest, InPlaceConstruction) {
  c10::OptionalArrayRef<int64_t> arr(std::vector<int64_t>{1, 2, 3, 4, 5});

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();

  // [DIFF] 此处访问可能导致随机值或崩溃
  // file << std::to_string(arr.front()) << " ";  // 已注释
  file << std::to_string(arr.has_value()) << " ";
  file << std::to_string(arr->size()) << " ";

  file.saveFile();
}
```

---

## 输出对比

| 测试用例 | Paddle 输出 | Torch 输出 |
|---------|------------|------------|
| InPlaceConstruction | `1 5`（稳定字段） | `1 5`（稳定字段） |
| front() 值 | 不稳定（随机值） | 不稳定（随机值） |

---

## 初步问题分析

OptionalArrayRef 核心功能在两个框架中完全兼容。差异仅在于：
1. 运行时内存地址值不同（正常差异）
2. 内部对象标识符不同（正常差异）
3. 临时对象悬空引用问题：使用 std::vector 临时对象构造 OptionalArrayRef 时，ArrayRef 会指向已释放的内存，导致未定义行为。

---


---

# Exception 宏（`TORCH_CHECK_EQ / TORCH_CHECK_NE` 已对齐）

> Paddle 头文件：`c10/util/Exception.h`
>
> 2026-03-29 复核：本节原先把 `TORCH_CHECK_OP` 派生宏的失败路径记成“Torch abort / Paddle throw”，这与当前 PyTorch 头文件实现和本仓库运行结果都不一致。当前 `TORCH_CHECK_EQ / NE / LE / LT / GE / GT` 已通过共享 `TORCH_CHECK_OP` 对齐到统一的失败消息前缀，并由测试直接校验异常文本。

## 当前状态

当前 compat 对齐点如下：

1. `TORCH_CHECK_EQ / NE / LE / LT / GE / GT` 统一复用 `TORCH_CHECK_OP`
2. 失败消息前缀对齐为 PyTorch 口径：`Check failed: <lhs> <op> <rhs> (<lhs_value> vs. <rhs_value>). `
3. `ExceptionTest` 不再通过 `#if USE_PADDLE_API` 分叉 `EXPECT_DEATH` / `try-catch` 两套逻辑，而是统一捕获异常并校验 `what()` 中的共享前缀

需要特别说明的是：PyTorch 当前 `TORCH_CHECK_OP` 宏走的是 `NON_FATAL_IF(..., exit_on_fatal=false)` 路径，失败时会抛出异常，而不是本节旧文档中记录的 `abort()`。

---

## 当前测试覆盖

测试文件：`test/c10/util/ExceptionTest.cpp`

### 关键测试项

1. `TorchCheckEqFailure`：校验 `TORCH_CHECK_EQ(3, 4)` 抛异常，且异常文本包含 `Check failed: 3 == 4 (3 vs. 4). `
2. `TorchCheckNe`：校验 `TORCH_CHECK_NE(3, 4)` 成功；`TORCH_CHECK_NE(3, 3)` 抛异常，且异常文本包含 `Check failed: 3 != 3 (3 vs. 3). `
3. `TorchCheckComparisons`：保留 `LT / LE / GT / GE` 成功路径覆盖，确认共享宏未破坏正向行为

### 当前测试代码

```cpp
template <typename Fn>
bool ThrowsMessageContaining(Fn&& fn, const char* expected_substr) {
  try {
    fn();
  } catch (const std::exception& e) {
    return std::string(e.what()).find(expected_substr) != std::string::npos;
  } catch (...) {
    return false;
  }
  return false;
}

TEST_F(ExceptionTest, TorchCheckEqFailure) {
  ...
  bool caught = ThrowsMessageContaining(
      [] { TORCH_CHECK_EQ(3, 4); }, "Check failed: 3 == 4 (3 vs. 4). ");
  file << std::to_string(caught ? 1 : 0) << " ";
  ...
}

TEST_F(ExceptionTest, TorchCheckNe) {
  ...
  bool caught = ThrowsMessageContaining(
      [] { TORCH_CHECK_NE(3, 3); }, "Check failed: 3 != 3 (3 vs. 3). ");
  file << std::to_string(caught ? 1 : 0) << " ";
  ...
}
```

---

## 当前对齐结果

| 测试用例 | Paddle/Torch 当前输出 |
|---------|----------------------|
| `TorchCheckEqFailure` | `1` |
| `TorchCheckNe` | `1 1` |
| `TorchCheckComparisons` | `1 1 1 1` |

补充说明：

- `1` 表示异常被成功捕获，且 `what()` 中包含与 PyTorch 对齐后的共享前缀。
- 当前 `bash test/result_cmp.sh ./build/` 中，`paddle_ExceptionTest` 与 `torch_ExceptionTest` 的结果文件应保持一致。

---

## 历史背景

本节原先记录的是两类历史问题：

1. 文档把 PyTorch `TORCH_CHECK_OP` 失败路径误记成了 death test / `abort()`
2. Paddle compat 的 `TORCH_CHECK_OP` 报错文案与 PyTorch 前缀不一致，测试也因此长期通过条件编译分叉来规避直接比对

当前这两项都已修正，旧结论仅保留作为回溯背景。

---
