## [paddle 参数更多]at::argmin

### PyTorch C++ API
```cpp
at::argmin(self, dim=::std::nullopt, keepdim=false)
```

### Paddle C++ API
```cpp
paddle::experimental::argmin(x, axis, keepdims=false, flatten=false, dtype=DataType::INT64)
```

两者功能一致，Paddle 相比 PyTorch 支持更多参数，具体如下：

> 注：参数映射表按 PyTorch 签名顺序排列。

### 参数映射

| PyTorch C++ | Paddle C++ | 备注 |
| ----------- | ---------- | ---- |
| self | x | 仅参数名不一致，`self` 对应 `x`。 |
| dim | axis | 仅参数名不一致，`dim` 对应 `axis`。 默认值不同：PyTorch 默认 `dim=::std::nullopt`，Paddle 无默认值。 |
| keepdim | - | Paddle 无此参数，PyTorch 有 `keepdim`。 |
| - | keepdims | PyTorch 无此参数，Paddle 有 `keepdims`。 |
| - | flatten | PyTorch 无此参数，Paddle 有 `flatten`。 |
| - | dtype | PyTorch 无此参数，Paddle 有 `dtype`。 |
