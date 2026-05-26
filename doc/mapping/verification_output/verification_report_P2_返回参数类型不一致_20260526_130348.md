# API 映射验证报告

生成时间: 2026-05-26 13:03:48

## 执行摘要

| 批次 | 总数 | 已验证 | 待审核 | 发现别名 | 真正缺失 |
|------|------|--------|--------|----------|----------|
| P2_返回参数类型不一致 | 25 | 5 | 20 | 0 | 0 |
| **总计** | **25** | **5** | **20** | **0** | **0** |

## Kernel 已注册但未暴露到 api.h 的候选

| PyTorch API | CPU Kernel | GPU Kernel |
|-------------|-----------|-----------|
| `at::argsort` | 是 | 是 |
| `at::batch_norm` | 是 | 是 |
| `at::cummax` | 是 | 是 |
| `at::cummin` | 是 | 是 |
| `at::fractional_max_pool2d` | 是 | 是 |
| `at::fractional_max_pool3d` | 是 | 是 |
| `at::kthvalue` | 是 | 是 |
| `at::lstm` | 是 | 是 |
| `at::lu_unpack` | 是 | 是 |
| `at::median` | 是 | 是 |
| `at::mode` | 是 | 是 |
| `at::nanmedian` | 是 | 是 |
| `at::nll_loss` | 是 | 是 |
| `at::norm` | 是 | 是 |
| `at::qr` | 是 | 是 |
| `at::rms_norm` | 否 | 是 |
| `at::svd` | 是 | 否 |
| `at::topk` | 是 | 是 |
| `at::unique_consecutive` | 是 | 是 |

## 详细验证结果

### P2_返回参数类型不一致

- kernel_only: 19
- verified_api_h_only: 5
- yaml_only: 1

#### 需关注的 API

| API | 状态 | 备注 |
|-----|------|------|
| `at::aminmax` | yaml_only |  |
| `at::argsort` | kernel_only |  |
| `at::batch_norm` | kernel_only |  |
| `at::cummax` | kernel_only |  |
| `at::cummin` | kernel_only |  |
| `at::fractional_max_pool2d` | kernel_only |  |
| `at::fractional_max_pool3d` | kernel_only |  |
| `at::kthvalue` | kernel_only |  |
| `at::lstm` | kernel_only |  |
| `at::lu_unpack` | kernel_only |  |
| `at::median` | kernel_only |  |
| `at::mode` | kernel_only |  |
| `at::nanmedian` | kernel_only |  |
| `at::nll_loss` | kernel_only |  |
| `at::norm` | kernel_only |  |
| `at::qr` | kernel_only |  |
| `at::rms_norm` | kernel_only |  |
| `at::svd` | kernel_only |  |
| `at::topk` | kernel_only |  |
| `at::unique_consecutive` | kernel_only |  |
