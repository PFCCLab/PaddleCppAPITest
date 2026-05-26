## [输入参数类型不一致]at::masked_fill

### PyTorch C++ API
```cpp
at::masked_fill(self, mask, value)
```

### Paddle C++ API
```cpp
paddle::experimental::masked_fill(x, mask, value)
```

两者功能一致，但输入参数类型不一致，具体如下：

### 参数映射

| PyTorch C++ | Paddle C++ | 备注 |
| ----------- | ---------- | ---- |
| self | x | 仅参数名不一致，`self` 对应 `x`。 |
| mask | mask | 参数名与类型均一致。 |
| value | value | 参数类型不一致，PyTorch 为 `Scalar`，Paddle 为 `Tensor`。 |
