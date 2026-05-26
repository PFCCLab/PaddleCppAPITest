## [输入参数类型不一致]at::group_norm

### PyTorch C++ API
```cpp
at::group_norm(input, num_groups, weight={}, bias={}, eps=1e-05, cudnn_enabled=true)
```

### Paddle C++ API
```cpp
paddle::experimental::group_norm(x, scale, bias, epsilon=1e-5, groups=-1, data_format="NCHW")
```

两者功能一致，但输入参数类型不一致，具体如下：

### 参数映射

| PyTorch C++ | Paddle C++ | 备注 |
| ----------- | ---------- | ---- |
| input | x | 仅参数名不一致，`input` 对应 `x`。 |
| num_groups | scale | 参数名与类型均不一致，PyTorch `num_groups` (`int64_t`) 对应 Paddle `scale` (`optional<Tensor>`)。 |
| weight | bias | 参数名与类型均不一致，PyTorch `weight` (`::optional<Tensor>`) 对应 Paddle `bias` (`optional<Tensor>`)。 |
| bias | epsilon | 参数名与类型均不一致，PyTorch `bias` (`::optional<Tensor>`) 对应 Paddle `epsilon` (`double`)。 |
| eps | groups | 参数名与类型均不一致，PyTorch `eps` (`double`) 对应 Paddle `groups` (`int`)。 |
| cudnn_enabled | data_format | 参数名与类型均不一致，PyTorch `cudnn_enabled` (`bool`) 对应 Paddle `data_format` (`string`)。 |
