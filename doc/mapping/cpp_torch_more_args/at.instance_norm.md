## [torch 参数更多]at::instance_norm

### PyTorch C++ API
```cpp
at::instance_norm(input, weight, bias, running_mean, running_var, use_input_stats, momentum, eps, cudnn_enabled)
```

### Paddle C++ API
```cpp
paddle::experimental::instance_norm(x, scale, bias, epsilon=1e-5)
```

PyTorch 相比 Paddle 支持更多参数，具体如下：

> 注：参数映射表按 PyTorch 签名顺序排列。

### 参数映射

| PyTorch C++ | Paddle C++ | 备注 |
| ----------- | ---------- | ---- |
| input | x | 仅参数名不一致，`input` 对应 `x`。 |
| weight | scale | 仅参数名不一致，`weight` 对应 `scale`。 |
| bias | bias | 参数名一致。 |
| running_mean | - | Paddle 无此参数，PyTorch 有 `running_mean`。 |
| running_var | - | Paddle 无此参数，PyTorch 有 `running_var`。 |
| use_input_stats | - | Paddle 无此参数，PyTorch 有 `use_input_stats`。 |
| momentum | - | Paddle 无此参数，PyTorch 有 `momentum`。 |
| eps | epsilon | 仅参数名不一致，`eps` 对应 `epsilon`。 默认值不同：PyTorch 无默认值，Paddle 默认 `epsilon=1e-5`。 |
| cudnn_enabled | - | Paddle 无此参数，PyTorch 有 `cudnn_enabled`。 |
