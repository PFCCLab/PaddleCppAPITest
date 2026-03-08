##### 记录PaddleCPPAPITest仓库检测出来的接口不一致情况

# Allocator 类与 torch 存在差异

## 差异点列表

1.  **构造函数参数默认值**
2.  **拷贝语义**
3.  **`get_deleter()` 在默认构造后的返回值**
4.  **`clear()` 后 `get_deleter()` 的行为**
5.  **Device 类型和方法**
6.  **`allocation()` 方法**

---

涉及到的 PR：https://github.com/PFCCLab/PaddleCppAPITest/pull/42/changes#diff

---

# Device 类与 torch 存在差异

> Paddle 头文件：`c10\core\Device.h`

## 差异点列表

1.  **未指定 Index 时的默认行为**：PyTorch index = -1，has_index() = false；Paddle 强制默认为 0，has_index() = true
2.  **纯字符串解析行为**：PyTorch 保持无索引状态（如 `cpu`、`cuda`）；Paddle 自动补全为 0 号设备（如 `cpu:0`、`gpu:0`）
3.  **GPU/CUDA 字符串表示**：PyTorch 严格输出 `cuda` 或 `cuda:0`；Paddle 底层映射为 GPU，输出 `gpu:0` 或 `gpu:1`
4.  **底层类型枚举值（Enum ID）**：PyTorch CPU=0，CUDA=1；Paddle CPU=1，CUDA/GPU=2
5.  **默认 Tensor 所在设备**：PyTorch 处于无明确索引的 cpu 状态；Paddle 明确挂载在 cpu:0 设备上

---

# BFloat16 类与 torch 存在差异

> Paddle 头文件：`c10\util\BFloat16.h`

## 差异点列表

1.  **BFloat16 ScalarType 枚举值**：PyTorch 为 **11**，Paddle 为 **15**
2.  **ComplexFloat ScalarType 枚举值**：PyTorch 为 **8**，Paddle 为 **9**

---

# DefaultDtype 类与 torch 存在差异

> Paddle 头文件：`c10\core\DefaultDtype.h`

## 差异点列表

1.  **BFloat16 枚举值**：PyTorch 为 **11**，Paddle 为 **15**
2.  **ComplexFloat（复数类型）枚举值**：PyTorch 为 **8**，Paddle 为 **9**

---

# IValue 类与 torch 存在差异

> Paddle 头文件：`ATen/core/ivalue.h`

## 差异点列表

1.  **命名空间**：PyTorch 为 `c10::IValue`；Paddle 为 `torch::IValue`（c10 命名空间中不存在 IValue）
2.  **方法命名风格**：PyTorch 使用 camelCase（如 `isNone()`、`toBool()`）；Paddle 使用 snake_case（如 `is_none()`、`to_bool()`）
3.  **`tagKind()` 方法**：PyTorch 存在；Paddle 中**不存在**
4.  **字符串提取方法**：PyTorch 为 `toStringRef()`；Paddle 为 `to_string()`
5.  **Optional 支持命名空间**：PyTorch 使用 `c10::optional`；Paddle 使用 `paddle::optional`

---

# SparseTensor 类与 torch 存在差异

> Paddle 头文件：`ATen/ops/sparse_coo_tensor.h`、`ATen/ops/sparse_csr_tensor.h`

## 差异点列表

1.  **sparse_coo_tensor 无 size 推断行为**：PyTorch 能根据 indices 内容正确推断完整 size（如 `2 2 2`）；Paddle 推断结果第一个维度为 0（如 `0 2 2`）

---

# OptionalArrayRef 类与 torch 存在差异

> Paddle 头文件：`c10\util\OptionalArrayRef.h`

## 差异点列表

1.  **运行时内存地址值**：两框架输出的内存地址不同（属正常运行时差异，不影响功能）
2.  **内部对象标识符**：两框架内部唯一标识符数值不同（属正常实现差异，不影响功能）

> 注：OptionalArrayRef 核心功能（has_value、size、元素访问、reset、swap、emplace、slice 等）在两个框架中完全兼容，仅运行时地址和标识符存在差异。

---

# at::indexing（Slice / EllipsisIndexType）类与 torch 存在差异

> Paddle 头文件：`ATen/indexing.h`（位于 `paddle/phi/api/include/compat/ATen/indexing.h`）
> PyTorch 头文件：`ATen/TensorIndexing.h`

## 差异点列表

1.  **头文件路径不同**：PyTorch 为 `ATen/TensorIndexing.h`；Paddle compat 为 `ATen/indexing.h`
2.  **`Tensor::operator[](Slice)` 不支持**：PyTorch 的 `Tensor::operator[]` 接受 `at::indexing::Slice`；Paddle compat 的 `operator[]` 仅重载 `int64_t`，传入 `Slice` 会编译报错
3.  **多维 Slice 索引写法不同**：
    - PyTorch：`t.index({Slice(0,2), Slice(1,3)})` —— 接受 `std::initializer_list<TensorIndex>`
    - Paddle：`t.index(std::vector<at::indexing::Slice>{Slice(0,2), Slice(1,3)})` —— 仅重载 `std::vector<Slice>`
4.  **`TensorIndex` 类不存在**：Paddle compat 的 `indexing.h` 未定义 `TensorIndex` 类，注释掉了 `index(ArrayRef<TensorIndex>)` 重载，仅保留 `index(const std::vector<Slice>&)`

## 修复方式

在Paddle源文件修改前先使用 `USE_PADDLE_API` 宏在编译期分支，分别调用各自支持的接口：

```cpp
#if USE_PADDLE_API
  at::Tensor result = t.index(std::vector<at::indexing::Slice>{at::indexing::Slice(0, 2), at::indexing::Slice(1, 3)});
#else
  at::Tensor result = t.index({at::indexing::Slice(0, 2), at::indexing::Slice(1, 3)});
#endif
```

---
