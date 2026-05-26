# API 映射验证报告

生成时间: 2026-05-26 13:40:07

## 执行摘要

| 批次 | 总数 | 已验证 | 待审核 | 发现别名 | 真正缺失 |
|------|------|--------|--------|----------|----------|
| P2_torch_参数更多 | 22 | 19 | 3 | 3 | 0 |
| **总计** | **22** | **19** | **3** | **3** | **0** |

## 发现的别名映射候选

| PyTorch API | Paddle API | 规则 | 置信度 |
|-------------|-----------|------|--------|
| `at::elu` | `paddle::experimental::elu` | common_naming_differences | medium |
| `at::huber_loss` | `paddle::experimental::huber_loss` | common_naming_differences | medium |
| `at::instance_norm` | `paddle::experimental::instance_norm` | common_naming_differences | medium |
| `at::layer_norm` | `paddle::experimental::layer_norm` | common_naming_differences | medium |
| `at::log_softmax` | `paddle::experimental::log_softmax` | common_naming_differences | medium |
| `at::softmax` | `paddle::experimental::softmax` | common_naming_differences | medium |
| `at::_log_softmax` | `paddle::experimental::log_softmax` | strip_underscore_prefix | high |
| `at::_softmax` | `paddle::experimental::softmax` | strip_underscore_prefix | high |
| `at::_standard_gamma` | `paddle::experimental::standard_gamma` | strip_underscore_prefix | high |

## 详细验证结果

### P2_torch_参数更多

- alias_candidate: 3
- verified_api_h_only: 19

#### 需关注的 API

| API | 状态 | 备注 |
|-----|------|------|
| `at::_log_softmax` | alias_candidate |  |
| `at::_softmax` | alias_candidate |  |
| `at::_standard_gamma` | alias_candidate |  |
