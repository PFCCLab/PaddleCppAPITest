## [API 别名]at::_fft_r2c

### PyTorch C++ API
```cpp
at::_fft_r2c(self, dim, normalization, onesided)
```

### Paddle C++ API
```cpp
paddle::experimental::fft_r2c(x, axes, normalization, forward, onesided)
```

两者功能一致，但 API 名称不同，PyTorch 为 `_fft_r2c`，Paddle 为 `fft_r2c`。参数映射具体如下：

### 参数映射

| PyTorch C++ | Paddle C++ | 备注 |
| ----------- | ---------- | ---- |
| self | x | 仅参数名不一致，`self` 对应 `x`。 |
| dim | - | Paddle 无此参数，PyTorch 有 `dim`。 |
| normalization | normalization | 参数名一致。 |
| onesided | onesided | 参数名一致。 |
| - | axes | PyTorch 无此参数，Paddle 有 `axes`。 |
| - | forward | PyTorch 无此参数，Paddle 有 `forward`。 |
