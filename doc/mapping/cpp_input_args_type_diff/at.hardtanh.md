## [输入参数类型不一致]at::hardtanh

### PyTorch C++ API
```cpp
at::hardtanh(self, min_val=-1, max_val=1)
```

### Paddle C++ API
```cpp
paddle::experimental::hardtanh(x, t_min=0, t_max=24)
```

两者功能一致，但输入参数类型不一致，具体如下：

### 参数映射

| PyTorch C++ | Paddle C++ | 备注 |
| ----------- | ---------- | ---- |
| self | x | 仅参数名不一致，`self` 对应 `x`。 |
| min_val | t_min | 参数名与类型均不一致，PyTorch `min_val` (`Scalar`) 对应 Paddle `t_min` (`float`)。 |
| max_val | t_max | 参数名与类型均不一致，PyTorch `max_val` (`Scalar`) 对应 Paddle `t_max` (`float`)。 |
