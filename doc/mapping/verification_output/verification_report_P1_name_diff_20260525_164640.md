# API 映射验证报告

生成时间: 2026-05-25 16:46:40

## 执行摘要

| 批次 | 总数 | 已验证 | 待审核 | 发现别名 | 真正缺失 |
|------|------|--------|--------|----------|----------|
| P1_name_diff | 79 | 77 | 2 | 2 | 0 |
| **总计** | **79** | **77** | **2** | **2** | **0** |

## 发现的别名映射候选

| PyTorch API | Paddle API | 规则 | 置信度 |
|-------------|-----------|------|--------|
| `at::hardswish` | `paddle::experimental::hardswish` | common_naming_differences | medium |
| `at::hardtanh` | `paddle::experimental::hardtanh` | common_naming_differences | medium |
| `at::pow` | `paddle::experimental::elementwise_pow` | paddle_elementwise_alias | high |
| `at::relu` | `paddle::experimental::relu` | common_naming_differences | medium |
| `at::_conj` | `paddle::experimental::conj` | strip_underscore_prefix | high |
| `at::log_sigmoid` | `paddle::experimental::logsigmoid` | common_naming_differences | medium |

## 详细验证结果

### P1_name_diff

- alias_candidate: 2
- verified_api_h_only: 77

#### 需关注的 API

| API | 状态 | 备注 |
|-----|------|------|
| `at::_conj` | alias_candidate |  |
| `at::log_sigmoid` | alias_candidate |  |
