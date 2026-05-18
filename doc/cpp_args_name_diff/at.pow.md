## [仅参数名不一致]at::pow

### PyTorch C++ API
```cpp
at::pow(self, exponent)
```

### Paddle C++ API
```cpp
paddle::experimental::pow(x, y=1.0f)
```

两者功能一致且参数用法一致，仅参数名不一致，具体如下：

### 参数映射

| PyTorch C++ | Paddle C++ | 备注 |
| ----------- | ---------- | ---- |
| self | x | 仅参数名不一致，`self` 对应 `x`。 |
| exponent | y | 仅参数名不一致，`exponent` 对应 `y`。 |
