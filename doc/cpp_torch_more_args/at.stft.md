## [torch 参数更多]at::stft

### PyTorch C++ API
```cpp
at::stft(self, n_fft, hop_length, win_length, window, normalized, onesided=::std::nullopt, return_complex=::std::nullopt, align_to_window=::std::nullopt)
```

### Paddle C++ API
```cpp
paddle::experimental::stft(x, window, n_fft, hop_length, normalized, onesided)
```

PyTorch 相比 Paddle 支持更多参数，具体如下：

> 注：参数映射表按 PyTorch 签名顺序排列。

### 参数映射

| PyTorch C++ | Paddle C++ | 备注 |
| ----------- | ---------- | ---- |
| self | x | 仅参数名不一致，`self` 对应 `x`。 |
| n_fft | n_fft | 参数名一致。 |
| hop_length | hop_length | 参数名一致。 |
| win_length | - | Paddle 无此参数，PyTorch 有 `win_length`。 |
| window | window | 参数名一致。 |
| normalized | normalized | 参数名一致。 |
| onesided | onesided | 参数名一致。 默认值不同：PyTorch 默认 `onesided=::std::nullopt`，Paddle 无默认值。 |
| return_complex | - | Paddle 无此参数，PyTorch 有 `return_complex`。 |
| align_to_window | - | Paddle 无此参数，PyTorch 有 `align_to_window`。 |
