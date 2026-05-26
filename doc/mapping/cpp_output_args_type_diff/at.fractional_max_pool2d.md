## [返回参数类型不一致]at::fractional_max_pool2d

### PyTorch C++ API
```cpp
at::fractional_max_pool2d(self, kernel_size, output_size, random_samples) -> ::tuple<Tensor,Tensor>
```

### Paddle C++ API
```cpp
paddle::experimental::fractional_max_pool2d(x, output_size, kernel_size={0, 0}, random_u=0.0, return_mask=true) -> tuple<Tensor, Tensor>
```

两者功能一致，但返回类型不一致，具体如下：

### 返回类型映射

| PyTorch C++ | Paddle C++ | 备注 |
| ----------- | ---------- | ---- |
| ::tuple<Tensor,Tensor> | tuple<Tensor, Tensor> | 返回类型不一致。 |
