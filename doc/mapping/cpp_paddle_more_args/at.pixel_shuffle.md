## [paddle 参数更多]at::pixel_shuffle

### PyTorch C++ API
```cpp
at::pixel_shuffle(self, upscale_factor)
```

### Paddle C++ API
```cpp
paddle::experimental::pixel_shuffle(x, upscale_factor=1, data_format="NCHW")
```

两者功能一致，Paddle 相比 PyTorch 支持更多参数，具体如下：

> 注：参数映射表按 PyTorch 签名顺序排列。

### 参数映射

| PyTorch C++ | Paddle C++ | 备注 |
| ----------- | ---------- | ---- |
| self | x | 仅参数名不一致，`self` 对应 `x`。 |
| upscale_factor | upscale_factor | 参数名一致。 默认值不同：PyTorch 无默认值，Paddle 默认 `upscale_factor=1`。 |
| - | data_format | PyTorch 无此参数，Paddle 有 `data_format`。 |
