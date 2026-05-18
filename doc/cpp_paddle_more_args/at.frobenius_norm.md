## [paddle 参数更多]at::frobenius_norm

### PyTorch C++ API
```cpp
at::frobenius_norm(self, dim, keepdim=false)
```

### Paddle C++ API
```cpp
paddle::experimental::frobenius_norm(x, axis, keep_dim, reduce_all)
```

两者功能一致，Paddle 相比 PyTorch 支持更多参数，具体如下：

> 注：参数映射表按 PyTorch 签名顺序排列。

### 参数映射

| PyTorch C++ | Paddle C++ | 备注 |
| ----------- | ---------- | ---- |
| self | x | 仅参数名不一致，`self` 对应 `x`。 |
| dim | axis | 仅参数名不一致，`dim` 对应 `axis`。 |
| keepdim | - | Paddle 无此参数，PyTorch 有 `keepdim`。 |
| - | keep_dim | PyTorch 无此参数，Paddle 有 `keep_dim`。 |
| - | reduce_all | PyTorch 无此参数，Paddle 有 `reduce_all`。 |
