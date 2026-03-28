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

# Exception 宏（TORCH_CHECK_EQ / TORCH_CHECK_NE 失败语义差异）

> Paddle 头文件：`c10/util/Exception.h`

## 差异点列表

1. **`TORCH_CHECK_EQ` 失败行为**：PyTorch 调用 `abort()` 终止进程（测试用 `EXPECT_DEATH` 捕获）；Paddle 抛出 C++ 异常（测试用 try-catch 捕获）。
2. **`TORCH_CHECK_NE` 失败行为**：同上，两者失败行为不一致。

当前代码通过 `#if USE_PADDLE_API` 分叉两套检测逻辑以绕过差异，但这导致两个平台实际走不同测试路径，无法真正对比行为。

---

## Diff 测试用例位置

测试文件：`test/c10/util/ExceptionTest.cpp`

### 测试用例原文

```cpp
// 测试 TORCH_CHECK_EQ 失败行为
TEST_F(ExceptionTest, TorchCheckEqFailure) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();

#if USE_PADDLE_API
  // Paddle: 抛出异常
  try {
    TORCH_CHECK_EQ(1, 2, "Values should be equal");
    file << "no_exception ";
  } catch (const c10::Error& e) {
    file << "c10::Error ";
  }
#else
  // PyTorch: 调用 abort()，使用 EXPECT_DEATH 捕获
  // 在非 death test 中直接跳过
  file << "skipped ";
#endif
  file.saveFile();
}

// 测试 TORCH_CHECK_NE 失败行为
TEST_F(ExceptionTest, TorchCheckNe) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();

#if USE_PADDLE_API
  // Paddle: 抛出异常
  try {
    TORCH_CHECK_NE(1, 1, "Values should not be equal");
    file << "no_exception ";
  } catch (const c10::Error& e) {
    file << "c10::Error ";
  }
#else
  // PyTorch: 调用 abort()
  file << "skipped ";
#endif
  file.saveFile();
}
```

---

## 输出对比

| 测试用例 | Paddle 输出 | Torch 输出 |
|---------|------------|------------|
| TorchCheckEqFailure | `c10::Error` | `skipped`（需用 EXPECT_DEATH） |
| TorchCheckNe | `c10::Error` | `skipped`（需用 EXPECT_DEATH） |

---

## 初步问题分析

1. **TORCH_CHECK_EQ 失败行为**：PyTorch 调用 abort() 终止进程，Paddle 抛出 C++ 异常。
2. **TORCH_CHECK_NE 失败行为**：同上。

当前通过条件编译分叉两套测试逻辑，导致无法真正对比两框架的行为差异。

---
