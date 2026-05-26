# API 映射验证报告

生成时间: 2026-05-26 13:28:22

## 执行摘要

| 批次 | 总数 | 已验证 | 待审核 | 发现别名 | 真正缺失 |
|------|------|--------|--------|----------|----------|
| P3_alias | 13 | 0 | 13 | 13 | 0 |
| **总计** | **13** | **0** | **13** | **13** | **0** |

## 发现的别名映射候选

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
| `at::log_sigmoid` | `paddle::experimental::logsigmoid` | common_naming_differences | medium |

## 详细验证结果

### P3_alias

- alias_candidate: 13

#### 需关注的 API

| API | 状态 | 备注 |
|-----|------|------|
| `at::_conj` | alias_candidate |  |
| `at::_fft_c2c` | alias_candidate |  |
| `at::_fft_c2r` | alias_candidate |  |
| `at::_fft_r2c` | alias_candidate |  |
| `at::_log_softmax` | alias_candidate |  |
| `at::_logcumsumexp` | alias_candidate |  |
| `at::_softmax` | alias_candidate |  |
| `at::_stack` | alias_candidate |  |
| `at::_standard_gamma` | alias_candidate |  |
| `at::conv_transpose2d` | alias_candidate |  |
| `at::conv_transpose3d` | alias_candidate |  |
| `at::grid_sampler` | alias_candidate |  |
| `at::log_sigmoid` | alias_candidate |  |
