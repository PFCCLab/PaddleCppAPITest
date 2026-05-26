## [输入参数类型不一致]at::renorm

### PyTorch C++ API
```cpp
at::renorm(self, p, dim, maxnorm)
```

### Paddle C++ API
```cpp
paddle::experimental::renorm(x, p, axis, max_norm)
```

两者功能一致，但输入参数类型不一致，具体如下：

### 参数映射

| PyTorch C++ | Paddle C++ | 备注 |
| ----------- | ---------- | ---- |
| self | x | 仅参数名不一致，`self` 对应 `x`。 |
| p | p | 参数类型不一致，PyTorch 为 `Scalar`，Paddle 为 `float`。 |
| dim | axis | 参数名与类型均不一致，PyTorch `dim` (`int64_t`) 对应 Paddle `axis` (`int`)。 |
| maxnorm | max_norm | 参数名与类型均不一致，PyTorch `maxnorm` (`Scalar`) 对应 Paddle `max_norm` (`float`)。 |
