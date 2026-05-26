## [返回参数类型不一致]at::batch_norm

### PyTorch C++ API
```cpp
at::batch_norm(input, weight, bias, running_mean, running_var, training, momentum, eps, cudnn_enabled) -> Tensor
```

### Paddle C++ API
```cpp
paddle::experimental::batch_norm(x, mean, variance, scale, bias, is_test, momentum, epsilon, data_format, use_global_stats, trainable_statistics) -> tuple<Tensor, Tensor, Tensor, Tensor, Tensor, Tensor>
```

两者功能一致，但返回类型不一致，具体如下：

### 返回类型映射

| PyTorch C++ | Paddle C++ | 备注 |
| ----------- | ---------- | ---- |
| Tensor | tuple<Tensor, Tensor, Tensor, Tensor, Tensor, Tensor> | 返回类型不一致。 |
