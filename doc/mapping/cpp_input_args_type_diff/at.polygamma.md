## [输入参数类型不一致]at::polygamma

### PyTorch C++ API
```cpp
at::polygamma(n, self)
```

### Paddle C++ API
```cpp
paddle::experimental::polygamma(x, n)
```

两者功能一致，但输入参数类型不一致，具体如下：

### 参数映射

| PyTorch C++ | Paddle C++ | 备注 |
| ----------- | ---------- | ---- |
| n | x | 参数名与类型均不一致，PyTorch `n` (`int64_t`) 对应 Paddle `x` (`Tensor`)。 |
| self | n | 参数名与类型均不一致，PyTorch `self` (`Tensor`) 对应 Paddle `n` (`int`)。 |
