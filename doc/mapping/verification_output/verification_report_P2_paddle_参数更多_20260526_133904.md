# API 映射验证报告

生成时间: 2026-05-26 13:39:04

## 执行摘要

| 批次 | 总数 | 已验证 | 待审核 | 发现别名 | 真正缺失 |
|------|------|--------|--------|----------|----------|
| P2_paddle_参数更多 | 44 | 39 | 5 | 5 | 0 |
| **总计** | **44** | **39** | **5** | **5** | **0** |

## 发现的别名映射候选

| PyTorch API | Paddle API | 规则 | 置信度 |
|-------------|-----------|------|--------|
| `at::dropout` | `paddle::experimental::dropout` | common_naming_differences | medium |
| `at::selu` | `paddle::experimental::selu` | common_naming_differences | medium |
| `at::_fft_c2r` | `paddle::experimental::fft_c2r` | strip_underscore_prefix | high |
| `at::_fft_r2c` | `paddle::experimental::fft_r2c` | strip_underscore_prefix | high |
| `at::_logcumsumexp` | `paddle::experimental::logcumsumexp` | strip_underscore_prefix | high |
| `at::conv_transpose2d` | `paddle::experimental::conv2d_transpose` | conv_transpose_alias | high |
| `at::conv_transpose3d` | `paddle::experimental::conv3d_transpose` | conv_transpose_alias | high |

## 详细验证结果

### P2_paddle_参数更多

- alias_candidate: 5
- verified_api_h_only: 39

#### 需关注的 API

| API | 状态 | 备注 |
|-----|------|------|
| `at::_fft_c2r` | alias_candidate |  |
| `at::_fft_r2c` | alias_candidate |  |
| `at::_logcumsumexp` | alias_candidate |  |
| `at::conv_transpose2d` | alias_candidate |  |
| `at::conv_transpose3d` | alias_candidate |  |
