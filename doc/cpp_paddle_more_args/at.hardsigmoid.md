## [paddle 参数更多]at::hardsigmoid

### PyTorch C++ API
```cpp
at::hardsigmoid(self)
```

### Paddle C++ API
```cpp
paddle::experimental::hardsigmoid(x, slope=0.2, offset=0.5)
```

两者功能一致，Paddle 相比 PyTorch 支持更多参数，具体如下：

> 注：参数映射表按 PyTorch 签名顺序排列。

### 参数映射

| PyTorch C++ | Paddle C++ | 备注 |
| ----------- | ---------- | ---- |
| self | x | 仅参数名不一致，`self` 对应 `x`。 |
| - | slope | PyTorch 无此参数，Paddle 有 `slope`。 |
| - | offset | PyTorch 无此参数，Paddle 有 `offset`。 |
