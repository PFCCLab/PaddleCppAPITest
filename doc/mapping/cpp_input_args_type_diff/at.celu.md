## [输入参数类型不一致]at::celu

### PyTorch C++ API
```cpp
at::celu(self, alpha=1.0)
```

### Paddle C++ API
```cpp
paddle::experimental::celu(x, alpha=1.0)
```

两者功能一致，但输入参数类型不一致，具体如下：

### 参数映射

| PyTorch C++ | Paddle C++ | 备注 |
| ----------- | ---------- | ---- |
| self | x | 仅参数名不一致，`self` 对应 `x`。 |
| alpha | alpha | 参数类型不一致，PyTorch 为 `Scalar`，Paddle 为 `float`。 |
