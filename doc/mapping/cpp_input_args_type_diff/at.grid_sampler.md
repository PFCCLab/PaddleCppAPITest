## [输入参数类型不一致]at::grid_sampler

### PyTorch C++ API
```cpp
at::grid_sampler(input, grid, interpolation_mode, padding_mode, align_corners)
```

### Paddle C++ API
```cpp
paddle::experimental::grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=true)
```

两者功能一致，但输入参数类型不一致，具体如下：

### 参数映射

| PyTorch C++ | Paddle C++ | 备注 |
| ----------- | ---------- | ---- |
| input | x | 仅参数名不一致，`input` 对应 `x`。 |
| grid | grid | 参数名与类型均一致。 |
| interpolation_mode | mode | 参数名与类型均不一致，PyTorch `interpolation_mode` (`int64_t`) 对应 Paddle `mode` (`string`)。 |
| padding_mode | padding_mode | 参数类型不一致，PyTorch 为 `int64_t`，Paddle 为 `string`。 |
| align_corners | align_corners | 参数名与类型均一致。 |
