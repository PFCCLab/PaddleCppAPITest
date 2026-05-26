## [输入参数类型不一致]at::lerp

### PyTorch C++ API
```cpp
at::lerp(self, end, weight)
```

### Paddle C++ API
```cpp
paddle::experimental::lerp(x, y, weight)
```

两者功能一致，但输入参数类型不一致，具体如下：

### 参数映射

| PyTorch C++ | Paddle C++ | 备注 |
| ----------- | ---------- | ---- |
| self | x | 仅参数名不一致，`self` 对应 `x`。 |
| end | y | 仅参数名不一致，`end` 对应 `y`。 |
| weight | weight | 参数类型不一致，PyTorch 为 `Scalar`，Paddle 为 `Tensor`。 |
