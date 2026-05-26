## [API 别名]at::grid_sampler

### PyTorch C++ API
```cpp
at::grid_sampler(input, grid, interpolation_mode, padding_mode, align_corners)
```

### Paddle C++ API
```cpp
paddle::experimental::grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=true)
```

两者功能一致，但 API 名称不同，PyTorch 为 `grid_sampler`，Paddle 为 `grid_sample`。参数映射具体如下：

### 参数映射

| PyTorch C++ | Paddle C++ | 备注 |
| ----------- | ---------- | ---- |
| input | x | 仅参数名不一致，`input` 对应 `x`。 |
| grid | grid | 参数名一致。 |
| interpolation_mode | - | Paddle 无此参数，PyTorch 有 `interpolation_mode`。 |
| padding_mode | padding_mode | 参数名一致。 |
| align_corners | align_corners | 参数名一致。 |
| - | mode | PyTorch 无此参数，Paddle 有 `mode`。 |
