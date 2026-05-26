## [输入参数类型不一致]at::addmm

### PyTorch C++ API
```cpp
at::addmm(self, mat1, mat2, beta=1, alpha=1)
```

### Paddle C++ API
```cpp
paddle::experimental::addmm(input, x, y, beta=1.0, alpha=1.0)
```

两者功能一致，但输入参数类型不一致，具体如下：

### 参数映射

| PyTorch C++ | Paddle C++ | 备注 |
| ----------- | ---------- | ---- |
| self | input | 仅参数名不一致，`self` 对应 `input`。 |
| mat1 | x | 仅参数名不一致，`mat1` 对应 `x`。 |
| mat2 | y | 仅参数名不一致，`mat2` 对应 `y`。 |
| beta | beta | 参数类型不一致，PyTorch 为 `Scalar`，Paddle 为 `float`。 |
| alpha | alpha | 参数类型不一致，PyTorch 为 `Scalar`，Paddle 为 `float`。 |
