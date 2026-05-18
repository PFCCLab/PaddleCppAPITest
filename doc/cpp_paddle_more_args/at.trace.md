## [paddle 参数更多]at::trace

### PyTorch C++ API
```cpp
at::trace(self)
```

### Paddle C++ API
```cpp
paddle::experimental::trace(x, offset=0, axis1=0, axis2=1)
```

两者功能一致，Paddle 相比 PyTorch 支持更多参数，具体如下：

> 注：参数映射表按 PyTorch 签名顺序排列。

### 参数映射

| PyTorch C++ | Paddle C++ | 备注 |
| ----------- | ---------- | ---- |
| self | x | 仅参数名不一致，`self` 对应 `x`。 |
| - | offset | PyTorch 无此参数，Paddle 有 `offset`。 |
| - | axis1 | PyTorch 无此参数，Paddle 有 `axis1`。 |
| - | axis2 | PyTorch 无此参数，Paddle 有 `axis2`。 |
