## [输入参数类型不一致]at::diagonal

### PyTorch C++ API
```cpp
at::diagonal(self, offset=0, dim1=0, dim2=1)
```

### Paddle C++ API
```cpp
paddle::experimental::diagonal(x, offset=0, axis1=0, axis2=1)
```

两者功能一致，但输入参数类型不一致，具体如下：

### 参数映射

| PyTorch C++ | Paddle C++ | 备注 |
| ----------- | ---------- | ---- |
| self | x | 仅参数名不一致，`self` 对应 `x`。 |
| offset | offset | 参数类型不一致，PyTorch 为 `int64_t`，Paddle 为 `int`。 |
| dim1 | axis1 | 参数名与类型均不一致，PyTorch `dim1` (`int64_t`) 对应 Paddle `axis1` (`int`)。 |
| dim2 | axis2 | 参数名与类型均不一致，PyTorch `dim2` (`int64_t`) 对应 Paddle `axis2` (`int`)。 |
