# API 映射验证报告

生成时间: 2026-05-26 13:18:55

## 执行摘要

| 批次 | 总数 | 已验证 | 待审核 | 发现别名 | 真正缺失 |
|------|------|--------|--------|----------|----------|
| P1_name_diff | 77 | 77 | 0 | 0 | 0 |
| **总计** | **77** | **77** | **0** | **0** | **0** |

## 发现的别名映射候选

| PyTorch API | Paddle API | 规则 | 置信度 |
|-------------|-----------|------|--------|
| `at::hardswish` | `paddle::experimental::hardswish` | common_naming_differences | medium |
| `at::hardtanh` | `paddle::experimental::hardtanh` | common_naming_differences | medium |
| `at::pow` | `paddle::experimental::elementwise_pow` | paddle_elementwise_alias | high |
| `at::relu` | `paddle::experimental::relu` | common_naming_differences | medium |

## 详细验证结果

### P1_name_diff

- verified_api_h_only: 77
