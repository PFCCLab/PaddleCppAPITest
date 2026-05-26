## [输入参数类型不一致]at::leaky_relu

### PyTorch C++ API
```cpp
at::leaky_relu(self, negative_slope=0.01)
```

### Paddle C++ API
```cpp
paddle::experimental::leaky_relu(x, negative_slope=0.02)
```

两者功能一致，但输入参数类型不一致，具体如下：

### 参数映射

| PyTorch C++ | Paddle C++ | 备注 |
| ----------- | ---------- | ---- |
| self | x | 仅参数名不一致，`self` 对应 `x`。 |
| negative_slope | negative_slope | 参数类型不一致，PyTorch 为 `Scalar`，Paddle 为 `double`。 |
