## [输入参数类型不一致]at::bitwise_or

### PyTorch C++ API
```cpp
at::bitwise_or(self, other)
```

### Paddle C++ API
```cpp
paddle::experimental::bitwise_or(x, y)
```

两者功能一致，但输入参数类型不一致，具体如下：

### 参数映射

| PyTorch C++ | Paddle C++ | 备注 |
| ----------- | ---------- | ---- |
| self | x | 仅参数名不一致，`self` 对应 `x`。 |
| other | y | 参数名与类型均不一致，PyTorch `other` (`Scalar`) 对应 Paddle `y` (`Tensor`)。 |
