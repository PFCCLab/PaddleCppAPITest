## [仅参数名不一致]at::heaviside

### PyTorch C++ API
```cpp
at::heaviside(self, values)
```

### Paddle C++ API
```cpp
paddle::experimental::heaviside(x, y)
```

两者功能一致且参数用法一致，仅参数名不一致，具体如下：

### 参数映射

| PyTorch C++ | Paddle C++ | 备注 |
| ----------- | ---------- | ---- |
| self | x | 仅参数名不一致，`self` 对应 `x`。 |
| values | y | 仅参数名不一致，`values` 对应 `y`。 |
