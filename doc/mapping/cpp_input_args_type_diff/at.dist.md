## [输入参数类型不一致]at::dist

### PyTorch C++ API
```cpp
at::dist(self, other, p=2)
```

### Paddle C++ API
```cpp
paddle::experimental::dist(x, y, p=2.0)
```

两者功能一致，但输入参数类型不一致，具体如下：

### 参数映射

| PyTorch C++ | Paddle C++ | 备注 |
| ----------- | ---------- | ---- |
| self | x | 仅参数名不一致，`self` 对应 `x`。 |
| other | y | 仅参数名不一致，`other` 对应 `y`。 |
| p | p | 参数类型不一致，PyTorch 为 `Scalar`，Paddle 为 `float`。 |
