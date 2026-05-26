## [输入参数类型不一致]at::hardshrink

### PyTorch C++ API
```cpp
at::hardshrink(self, lambd=0.5)
```

### Paddle C++ API
```cpp
paddle::experimental::hardshrink(x, threshold=0.5)
```

两者功能一致，但输入参数类型不一致，具体如下：

### 参数映射

| PyTorch C++ | Paddle C++ | 备注 |
| ----------- | ---------- | ---- |
| self | x | 仅参数名不一致，`self` 对应 `x`。 |
| lambd | threshold | 参数名与类型均不一致，PyTorch `lambd` (`Scalar`) 对应 Paddle `threshold` (`float`)。 |
