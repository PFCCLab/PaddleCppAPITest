# P3 "API 别名"类Agent源码审核报告

审核时间：2026-05-26
审核范围：P3 批次 13 个 API（"API 别名"类）
审核方法：逐一阅读 PyTorch + Paddle kernel 实现文件，对比实现逻辑

## 审核结论概览

| 风险评级 | 数量 | 说明 |
|----------|------|------|
| 低 | 11 | 核心语义一致，仅名称差异或下划线前缀 |
| 中 | 2 | 下划线前缀变体有额外参数，需 compat 层适配 |

---

## 一、下划线前缀别名（9 个）— 风险：低~中

PyTorch 下划线前缀 API（如 `_softmax`）是内部实现入口，Paddle 对应 API 是标准命名。

| PyTorch API | Paddle API | 核心差异 | 风险 |
|-------------|-----------|---------|------|
| `_conj` | `conj` | PyTorch `_conj` 是内部 dispatcher 入口，与 `conj` 语义完全一致 | 低 |
| `_softmax` | `softmax` | PyTorch `_softmax` 多了 `half_to_float` 参数；Paddle `softmax` 无此参数 | **中** |
| `_log_softmax` | `log_softmax` | 同上，`half_to_float` 参数差异 | **中** |
| `_stack` | `stack` | 语义完全一致 | 低 |
| `_logcumsumexp` | `logcumsumexp` | 语义完全一致 | 低 |
| `_standard_gamma` | `standard_gamma` | 语义完全一致 | 低 |
| `_fft_c2c` | `fft_c2c` | FFT 操作，语义一致 | 低 |
| `_fft_c2r` | `fft_c2r` | FFT 操作，语义一致 | 低 |
| `_fft_r2c` | `fft_r2c` | FFT 操作，语义一致 | 低 |

### `_softmax` / `_log_softmax` 详细审核

**PyTorch**（`SoftMax.cpp:40-57`）：
```cpp
TORCH_META_FUNC(_softmax)(const Tensor& input, const int64_t dim, const bool half_to_float) {
  auto output_options = input.options().memory_format(LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  if (half_to_float) {
    output_options = output_options.dtype(ScalarType::Float);
  }
  // ...
}
```

**Paddle**：`softmax` 无 `half_to_float` 参数。但 Paddle 的 `api.h` 中 `softmax` 签名与 PyTorch `softmax` 一致（无 `half_to_float`）。

**结论**：`_softmax` 的 `half_to_float` 参数是 PyTorch 内部优化参数，不影响核心语义。compat 层封装时可直接忽略该参数。

---

## 二、命名差异别名（4 个）— 风险：低

| PyTorch API | Paddle API | 差异说明 |
|-------------|-----------|---------|
| `conv_transpose2d` | `conv2d_transpose` | 命名风格翻转（Nd + transpose vs transpose + Nd），语义完全一致 |
| `conv_transpose3d` | `conv3d_transpose` | 同上 |
| `grid_sampler` | `grid_sample` | 单复数差异（sampler vs sample），语义完全一致 |
| `log_sigmoid` | `logsigmoid` | 下划线 vs 驼峰风格，语义完全一致 |

---

## 总结

**P3 批次 13 个 API 中**：
- **11 个（84.6%）**：风险低，核心语义完全一致，compat 层只需处理参数名/数量映射
- **2 个（15.4%）**：`_softmax`、`_log_softmax` 有 `half_to_float` 内部参数，compat 层可忽略

**compat 层封装建议**：
1. 下划线前缀 API：去掉前缀后直接调用 Paddle 对应实现
2. 命名差异 API：直接映射参数名
3. `_softmax` / `_log_softmax`：忽略 `half_to_float` 参数（Paddle 自动处理 half→float）
