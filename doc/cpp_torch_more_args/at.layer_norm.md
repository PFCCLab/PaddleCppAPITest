## [torch 参数更多]at::layer_norm

### PyTorch C++ API
```cpp
at::layer_norm(input, normalized_shape, weight={}, bias={}, eps=1e-05, cudnn_enable=true)
```

### Paddle C++ API
```cpp
paddle::experimental::layer_norm(x, scale, bias, epsilon=1e-5, begin_norm_axis=1)
```

PyTorch 相比 Paddle 支持更多参数，具体如下：

> 注：参数映射表按 PyTorch 签名顺序排列。

### 参数映射

| PyTorch C++ | Paddle C++ | 备注 |
| ----------- | ---------- | ---- |
| input | x | 仅参数名不一致，`input` 对应 `x`。 |
| normalized_shape | - | Paddle 无此参数，PyTorch 有 `normalized_shape`。 |
| weight | scale | 仅参数名不一致，`weight` 对应 `scale`。 默认值不同：PyTorch 默认 `weight={}`，Paddle 无默认值。 |
| bias | bias | 参数名一致。 默认值不同：PyTorch 默认 `bias={}`，Paddle 无默认值。 |
| eps | epsilon | 仅参数名不一致，`eps` 对应 `epsilon`。 |
| cudnn_enable | - | Paddle 无此参数，PyTorch 有 `cudnn_enable`。 |
| - | begin_norm_axis | PyTorch 无此参数，Paddle 有 `begin_norm_axis`。 |
