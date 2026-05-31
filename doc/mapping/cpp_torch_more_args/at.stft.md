## [torch 参数更多]at::stft

> 状态： compat 层已实现 (2026-05-31)
> 位置：`paddle/phi/api/include/compat/ATen/ops/stft.h`
> 测试：`test/cpp/compat/ATen_stft_test.cc`、`test/ATen/ops/StftTest.cpp`

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
| hop_length | hop_length | 参数名一致。默认 `n_fft/4`。 |
| win_length | - | Paddle 无此参数，compat 层忽略该参数。 |
| window | window | 参数名一致。PyTorch 可选，compat 层未提供时创建矩形窗。 |
| normalized | normalized | 参数名一致。 |
| onesided | onesided | 参数名一致。默认 `true`。 |
| return_complex | - | Paddle 无此参数，compat 层忽略（Paddle 始终返回 complex）。 |
| align_to_window | - | Paddle 无此参数，compat 层忽略。 |

### 已实现行为

- `Tensor::stft()` 方法已注入 `at::Tensor` 类声明（`ATen/core/TensorBody.h`）。
- `at::stft` 命名空间函数已实现，映射到 `paddle::experimental::stft`。
- 1D 输入自动 unsqueeze 为 2D（Paddle stft 仅支持 2D）。
- `hop_length` 默认值为 `n_fft / 4`（与 PyTorch 一致）。
- `onesided` 默认值为 `true`（与 PyTorch 一致）。
- `window` 未提供时自动创建全 1 矩形窗（大小 `n_fft`）。
- `return_complex=true` 需显式传入以兼容 PyTorch 当前版本要求。

### 已知差异

1. **center padding**：PyTorch 默认对输入做 `n_fft/2` 的 reflect  padding，Paddle C++ API 不支持 reflect padding，compat 层未实现 center padding，因此边缘帧的数值与 PyTorch 存在差异。shape/dtype 一致。
2. **onesided=false**：Paddle stft kernel 在 `onesided=false` 时存在中间内存分配 dtype 不匹配问题，compat 层可调用但可能抛异常。
3. **FFT 数值精度**：Paddle 与 PyTorch 使用不同 FFT 后端，极个别频点可能存在微小数值差异。

### 测试覆盖

- **Paddle compat 测试**：`test/cpp/compat/ATen_stft_test.cc`
  - 基础功能、自定义 hop、带 window、onesided false、1D 输入、float64 dtype
- **PCAT 回归**：`test/ATen/ops/StftTest.cpp`
  - Shape 覆盖：小 shape、大 shape、1D 输入
  - Dtype 覆盖：kFloat、kDouble
  - API 变体：带 window、normalized、不同 hop_length
