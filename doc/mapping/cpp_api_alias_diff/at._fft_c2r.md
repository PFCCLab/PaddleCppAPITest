## [API 别名]at::_fft_c2r

### PyTorch C++ API
```cpp
at::_fft_c2r(self, dim, normalization, last_dim_size)
```

### Paddle C++ API
```cpp
paddle::experimental::fft_c2r(x, axes, normalization, forward, last_dim_size=0L)
```

两者功能一致，但 API 名称不同，PyTorch 为 `_fft_c2r`，Paddle 为 `fft_c2r`。参数映射具体如下：

### 参数映射

| PyTorch C++ | Paddle C++ | 备注 |
| ----------- | ---------- | ---- |
| self | x | 仅参数名不一致，`self` 对应 `x`。 |
| dim | - | Paddle 无此参数，PyTorch 有 `dim`。 |
| normalization | normalization | 参数名一致。 |
| last_dim_size | last_dim_size | 参数名一致。 |
| - | axes | PyTorch 无此参数，Paddle 有 `axes`。 |
| - | forward | PyTorch 无此参数，Paddle 有 `forward`。 |
