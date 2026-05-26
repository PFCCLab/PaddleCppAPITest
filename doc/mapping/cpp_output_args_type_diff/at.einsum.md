# at::einsum 与 paddle::experimental::einsum 差异对比

## 差异概述

| 维度 | PyTorch | Paddle | 差异 |
|------|---------|--------|------|
| 返回类型 | `Tensor` | `std::tuple<Tensor, std::vector<Tensor>, std::vector<Tensor>>` | **返回结构不同** |
| 参数 | `string_view equation, TensorList operands` | `const std::vector<Tensor>& x, const std::string& equation` | 参数类型等价 |
| 功能 | 爱因斯坦求和约定 | 爱因斯坦求和约定 | 核心功能一致 |

## PyTorch 签名

```cpp
// aten/src/ATen/native/LinearAlgebra.cpp
Tensor einsum(c10::string_view equation, TensorList operands)
```

- 返回单个 `Tensor`
- 支持 `TensorList`（张量列表）作为输入

## Paddle 签名

```cpp
// paddle/phi/api/include/api.h:1336
std::tuple<Tensor, std::vector<Tensor>, std::vector<Tensor>>
    einsum(const std::vector<Tensor>& x, const std::string& equation);
```

- 返回三元组 `(result_tensor, input_tensors, output_tensors)`
- `input_tensors` 和 `output_tensors` 是内部使用的中间结果

## 核心差异

Paddle 的 `einsum` 返回三元组是为了支持**反向传播**和**中间结果复用**。compat 层封装时只需取三元组的第一个元素：

```cpp
// compat 层建议实现
at::Tensor einsum(c10::string_view equation, at::TensorList operands) {
    auto paddle_result = paddle::experimental::einsum(
        std::vector<paddle::Tensor>(operands.begin(), operands.end()),
        std::string(equation)
    );
    return std::get<0>(paddle_result);  // 取第一个元素
}
```

## 审核结论

- **风险：低**
- 核心数学语义完全一致（爱因斯坦求和约定）
- 差异仅在于返回类型（Paddle 多了中间结果）
- compat 层只需取 tuple 的第一个元素
