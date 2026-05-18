## [paddle 参数更多]at::bitwise_left_shift

### PyTorch C++ API
```cpp
at::bitwise_left_shift(self, other)
```

### Paddle C++ API
```cpp
paddle::experimental::bitwise_left_shift(x, y, is_arithmetic=true)
```

两者功能一致，Paddle 相比 PyTorch 支持更多参数，具体如下：

> 注：参数映射表按 PyTorch 签名顺序排列。

### 参数映射

| PyTorch C++ | Paddle C++ | 备注 |
| ----------- | ---------- | ---- |
| self | x | 仅参数名不一致，`self` 对应 `x`。 |
| other | y | 仅参数名不一致，`other` 对应 `y`。 |
| - | is_arithmetic | PyTorch 无此参数，Paddle 有 `is_arithmetic`。 |
