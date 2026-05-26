## [输入参数类型不一致]at::_stack

### PyTorch C++ API
```cpp
at::_stack(tensors, dim=0)
```

### Paddle C++ API
```cpp
paddle::experimental::stack(x, axis=0)
```

两者功能一致，但输入参数类型不一致，具体如下：

### 参数映射

| PyTorch C++ | Paddle C++ | 备注 |
| ----------- | ---------- | ---- |
| tensors | x | 参数名与类型均不一致，PyTorch `tensors` (`TensorList`) 对应 Paddle `x` (`std::vector<Tensor>`)。 |
| dim | axis | 参数名与类型均不一致，PyTorch `dim` (`int64_t`) 对应 Paddle `axis` (`int`)。 |
