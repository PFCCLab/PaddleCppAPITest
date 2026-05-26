## [输入参数类型不一致]at::_fft_c2c

### PyTorch C++ API
```cpp
at::_fft_c2c(self, dim, normalization, forward)
```

### Paddle C++ API
```cpp
paddle::experimental::fft_c2c(x, axes, normalization, forward)
```

两者功能一致，但输入参数类型不一致，具体如下：

### 参数映射

| PyTorch C++ | Paddle C++ | 备注 |
| ----------- | ---------- | ---- |
| self | x | 仅参数名不一致，`self` 对应 `x`。 |
| dim | axes | 仅参数名不一致，`dim` 对应 `axes`。 |
| normalization | normalization | 参数类型不一致，PyTorch 为 `int64_t`，Paddle 为 `string`。 |
| forward | forward | 参数名与类型均一致。 |
