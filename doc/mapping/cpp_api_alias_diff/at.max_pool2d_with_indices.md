## [API 别名]at::max_pool2d_with_indices

### PyTorch C++ API
```cpp
at::max_pool2d_with_indices(self, kernel_size, stride={}, padding=0, dilation=1, ceil_mode=false)
```

### Paddle C++ API
```cpp
paddle::experimental::max_pool2d_with_index(x, kernel_size, strides={1, 1}, paddings={0, 0}, dilations={1, 1}, global_pooling=false, adaptive=false, ceil_mode=false)
```

两者功能一致，但 API 名称不同，PyTorch 为 `max_pool2d_with_indices`，Paddle 为 `max_pool2d_with_index`。参数映射具体如下：

### 参数映射

| PyTorch C++ | Paddle C++ | 备注 |
| ----------- | ---------- | ---- |
| self | x | 仅参数名不一致，`self` 对应 `x`。 |
| kernel_size | kernel_size | 参数名一致。 |
| stride | - | Paddle 无此参数，PyTorch 有 `stride`。 |
| padding | - | Paddle 无此参数，PyTorch 有 `padding`。 |
| dilation | - | Paddle 无此参数，PyTorch 有 `dilation`。 |
| ceil_mode | ceil_mode | 参数名一致。 |
| - | strides | PyTorch 无此参数，Paddle 有 `strides`。 |
| - | paddings | PyTorch 无此参数，Paddle 有 `paddings`。 |
| - | dilations | PyTorch 无此参数，Paddle 有 `dilations`。 |
| - | global_pooling | PyTorch 无此参数，Paddle 有 `global_pooling`。 |
| - | adaptive | PyTorch 无此参数，Paddle 有 `adaptive`。 |
