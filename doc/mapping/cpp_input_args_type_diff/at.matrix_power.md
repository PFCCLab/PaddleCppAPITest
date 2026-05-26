## [输入参数类型不一致]at::matrix_power

### PyTorch C++ API
```cpp
at::matrix_power(self, n)
```

### Paddle C++ API
```cpp
paddle::experimental::matrix_power(x, n)
```

两者功能一致，但输入参数类型不一致，具体如下：

### 参数映射

| PyTorch C++ | Paddle C++ | 备注 |
| ----------- | ---------- | ---- |
| self | x | 仅参数名不一致，`self` 对应 `x`。 |
| n | n | 参数类型不一致，PyTorch 为 `int64_t`，Paddle 为 `int`。 |
