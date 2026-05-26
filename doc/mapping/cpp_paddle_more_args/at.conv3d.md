## [paddle 参数更多]at::conv3d

### PyTorch C++ API
```cpp
at::conv3d(input, weight, bias={}, stride=1, padding=0, dilation=1, groups=1)
```

### Paddle C++ API
```cpp
paddle::experimental::conv3d(input, filter, strides={1, 1, 1}, paddings={0, 0, 0}, padding_algorithm="EXPLICIT", groups=1, dilations={1, 1, 1}, data_format="NCDHW")
```

两者功能一致，Paddle 相比 PyTorch 支持更多参数，具体如下：

> 注：参数映射表按 PyTorch 签名顺序排列。

### 参数映射

| PyTorch C++ | Paddle C++ | 备注 |
| ----------- | ---------- | ---- |
| input | input | 参数名一致。 |
| weight | filter | 仅参数名不一致，`weight` 对应 `filter`。 |
| bias | - | Paddle 无此参数，PyTorch 有 `bias`。 |
| stride | - | Paddle 无此参数，PyTorch 有 `stride`。 |
| padding | - | Paddle 无此参数，PyTorch 有 `padding`。 |
| dilation | - | Paddle 无此参数，PyTorch 有 `dilation`。 |
| groups | groups | 参数名一致。 |
| - | strides | PyTorch 无此参数，Paddle 有 `strides`。 |
| - | paddings | PyTorch 无此参数，Paddle 有 `paddings`。 |
| - | padding_algorithm | PyTorch 无此参数，Paddle 有 `padding_algorithm`。 |
| - | dilations | PyTorch 无此参数，Paddle 有 `dilations`。 |
| - | data_format | PyTorch 无此参数，Paddle 有 `data_format`。 |
