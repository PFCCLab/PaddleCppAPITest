## [仅参数名不一致]at::atan2

### PyTorch C++ API
```cpp
at::atan2(self, other)
```

### Paddle C++ API
```cpp
paddle::experimental::atan2(x, y)
```

两者功能一致且参数用法一致，仅参数名不一致，具体如下：

### 参数映射

| PyTorch C++ | Paddle C++ | 备注 |
| ----------- | ---------- | ---- |
| self | x | 仅参数名不一致，`self` 对应 `x`。 |
| other | y | 仅参数名不一致，`other` 对应 `y`。 |
