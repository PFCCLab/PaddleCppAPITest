# API 映射验证报告

生成时间: 2026-05-26 13:40:59

## 执行摘要

| 批次 | 总数 | 已验证 | 待审核 | 发现别名 | 真正缺失 |
|------|------|--------|--------|----------|----------|
| P2_输入参数类型不一致 | 40 | 36 | 4 | 3 | 0 |
| **总计** | **40** | **36** | **4** | **3** | **0** |

## 发现的别名映射候选

| PyTorch API | Paddle API | 规则 | 置信度 |
|-------------|-----------|------|--------|
| `at::celu` | `paddle::experimental::celu` | common_naming_differences | medium |
| `at::gelu` | `paddle::experimental::gelu` | common_naming_differences | medium |
| `at::group_norm` | `paddle::experimental::group_norm` | common_naming_differences | medium |
| `at::leaky_relu` | `paddle::experimental::leaky_relu` | common_naming_differences | medium |
| `at::_fft_c2c` | `paddle::experimental::fft_c2c` | strip_underscore_prefix | high |
| `at::_stack` | `paddle::experimental::stack` | strip_underscore_prefix | high |
| `at::grid_sampler` | `paddle::experimental::grid_sample` | common_naming_differences | medium |

## Kernel 已注册但未暴露到 api.h 的候选

| PyTorch API | CPU Kernel | GPU Kernel |
|-------------|-----------|-----------|
| `at::unbind` | 是 | 是 |

## 详细验证结果

### P2_输入参数类型不一致

- alias_candidate: 3
- kernel_only: 1
- verified_api_h_only: 36

#### 需关注的 API

| API | 状态 | 备注 |
|-----|------|------|
| `at::unbind` | kernel_only |  |
| `at::_fft_c2c` | alias_candidate |  |
| `at::_stack` | alias_candidate |  |
| `at::grid_sampler` | alias_candidate |  |
