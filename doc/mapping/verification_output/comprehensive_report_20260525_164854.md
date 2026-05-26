# API 映射综合验证报告

生成时间: 2026-05-25 16:48:54

## 总体概览

- **总验证 API 数**: 163
- **alias_candidate**: 15 (9.2%)
- **kernel_only**: 1 (0.6%)
- **truly_missing**: 4 (2.5%)
- **verified_api_h_only**: 77 (47.2%)
- **verified_compat**: 66 (40.5%)

## 各批次验证结果

| 批次 | 总数 | 状态分布 |
|------|------|----------|
| P0_exact_match | 66 | verified_compat: 66 |
| P1_name_diff | 79 | alias_candidate: 2, verified_api_h_only: 77 |
| P3_alias | 18 | alias_candidate: 13, kernel_only: 1, truly_missing: 4 |

## 别名映射候选汇总

| PyTorch API | Paddle API | 规则 | 置信度 |
|-------------|-----------|------|--------|
| `at::_conj` | `paddle::experimental::conj` | strip_underscore_prefix | high |
| `at::_fft_c2c` | `paddle::experimental::fft_c2c` | strip_underscore_prefix | high |
| `at::_fft_c2r` | `paddle::experimental::fft_c2r` | strip_underscore_prefix | high |
| `at::_fft_r2c` | `paddle::experimental::fft_r2c` | strip_underscore_prefix | high |
| `at::_log_softmax` | `paddle::experimental::log_softmax` | strip_underscore_prefix | high |
| `at::_logcumsumexp` | `paddle::experimental::logcumsumexp` | strip_underscore_prefix | high |
| `at::_softmax` | `paddle::experimental::softmax` | strip_underscore_prefix | high |
| `at::_stack` | `paddle::experimental::stack` | strip_underscore_prefix | high |
| `at::_standard_gamma` | `paddle::experimental::standard_gamma` | strip_underscore_prefix | high |
| `at::conv_transpose2d` | `paddle::experimental::conv2d_transpose` | conv_transpose_alias | high |
| `at::conv_transpose3d` | `paddle::experimental::conv3d_transpose` | conv_transpose_alias | high |
| `at::grid_sampler` | `paddle::experimental::grid_sample` | common_naming_differences | medium |
| `at::hardswish` | `paddle::experimental::hardswish` | common_naming_differences | medium |
| `at::hardtanh` | `paddle::experimental::hardtanh` | common_naming_differences | medium |
| `at::log_sigmoid` | `paddle::experimental::logsigmoid` | common_naming_differences | medium |
| `at::pow` | `paddle::experimental::elementwise_pow` | paddle_elementwise_alias | high |
| `at::range` | `paddle::experimental::arange` | common_naming_differences | medium |
| `at::range` | `paddle::experimental::range_v2` | paddle_v2_suffix | low |
| `at::relu` | `paddle::experimental::relu` | common_naming_differences | medium |

**总计发现 19 个别名候选**

## Kernel 已注册但未暴露到 api.h

| PyTorch API | Paddle Kernel 文件 | CPU | GPU |
|-------------|-------------------|-----|-----|
| `at::range` |  | 是 | 是 |

**总计发现 1 个 kernel_only 候选**

## 详细批次结果

### P0_exact_match

**总数**: 66

无需关注

### P1_name_diff

**总数**: 79

**需关注的 API**:

| API | 当前状态 | 验证状态 | 备注 | 别名候选 |
|-----|---------|---------|------|----------|
| `at::hardswish` | - | verified_api_h_only |  | hardswish(medium) |
| `at::hardtanh` | - | verified_api_h_only |  | hardtanh(medium) |
| `at::pow` | - | verified_api_h_only |  | elementwise_pow(high) |
| `at::relu` | - | verified_api_h_only |  | relu(medium) |
| `at::_conj` | - | alias_candidate |  | conj(high) |
| `at::log_sigmoid` | - | alias_candidate |  | logsigmoid(medium) |

### P3_alias

**总数**: 18

**需关注的 API**:

| API | 当前状态 | 验证状态 | 备注 | 别名候选 |
|-----|---------|---------|------|----------|
| `at::_conj` | - | alias_candidate |  | conj(high) |
| `at::_fft_c2c` | - | alias_candidate |  | fft_c2c(high) |
| `at::_fft_c2r` | - | alias_candidate |  | fft_c2r(high) |
| `at::_fft_r2c` | - | alias_candidate |  | fft_r2c(high) |
| `at::_log_softmax` | - | alias_candidate |  | log_softmax(high) |
| `at::_logcumsumexp` | - | alias_candidate |  | logcumsumexp(high) |
| `at::_softmax` | - | alias_candidate |  | softmax(high) |
| `at::_stack` | - | alias_candidate |  | stack(high) |
| `at::_standard_gamma` | - | alias_candidate |  | standard_gamma(high) |
| `at::conv_transpose2d` | - | alias_candidate |  | conv2d_transpose(high) |
| `at::conv_transpose3d` | - | alias_candidate |  | conv3d_transpose(high) |
| `at::grid_sampler` | - | alias_candidate |  | grid_sample(medium) |
| `at::log_sigmoid` | - | alias_candidate |  | logsigmoid(medium) |
| `at::range` | - | kernel_only |  | arange(medium), range_v2(low) |
