## [输入参数类型不一致]at::isclose

### PyTorch C++ API
```cpp
at::isclose(self, other, rtol=1e-05, atol=1e-08, equal_nan=false)
```

### Paddle C++ API
```cpp
paddle::experimental::isclose(x, y, rtol=1e-5, atol=1e-8, equal_nan=false)
```

两者功能一致，但输入参数类型不一致，具体如下：

### 参数映射

| PyTorch C++ | Paddle C++ | 备注 |
| ----------- | ---------- | ---- |
| self | x | 仅参数名不一致，`self` 对应 `x`。 |
| other | y | 仅参数名不一致，`other` 对应 `y`。 |
| rtol | rtol | 参数类型不一致，PyTorch 为 `double`，Paddle 为 `Scalar`。 |
| atol | atol | 参数类型不一致，PyTorch 为 `double`，Paddle 为 `Scalar`。 |
| equal_nan | equal_nan | 参数名与类型均一致。 |
