## [输入参数类型不一致]at::softplus

### PyTorch C++ API
```cpp
at::softplus(self, beta=1, threshold=20)
```

### Paddle C++ API
```cpp
paddle::experimental::softplus(x, beta=1.0, threshold=20.0)
```

两者功能一致，但输入参数类型不一致，具体如下：

### 参数映射

| PyTorch C++ | Paddle C++ | 备注 |
| ----------- | ---------- | ---- |
| self | x | 仅参数名不一致，`self` 对应 `x`。 |
| beta | beta | 参数类型不一致，PyTorch 为 `Scalar`，Paddle 为 `double`。 |
| threshold | threshold | 参数类型不一致，PyTorch 为 `Scalar`，Paddle 为 `double`。 |
