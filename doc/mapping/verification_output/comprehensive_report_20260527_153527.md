# API 映射综合验证报告

生成时间: 2026-05-27 15:35:27

## 总体概览

- **总验证 API 数**: 287
- **alias_candidate**: 24 (8.4%)
- **kernel_only**: 20 (7.0%)
- **verified_api_h_only**: 176 (61.3%)
- **verified_compat**: 66 (23.0%)
- **yaml_only**: 1 (0.3%)

## 各批次验证结果

| 批次 | 总数 | 状态分布 |
|------|------|----------|
| P0_exact_match | 66 | verified_compat: 66 |
| P1_name_diff | 77 | verified_api_h_only: 77 |
| P2_paddle_参数更多 | 44 | alias_candidate: 5, verified_api_h_only: 39 |
| P2_torch_参数更多 | 22 | alias_candidate: 3, verified_api_h_only: 19 |
| P2_输入参数类型不一致 | 40 | alias_candidate: 3, kernel_only: 1, verified_api_h_only: 36 |
| P2_返回参数类型不一致 | 25 | kernel_only: 19, verified_api_h_only: 5, yaml_only: 1 |
| P3_alias | 13 | alias_candidate: 13 |

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
| `at::celu` | `paddle::experimental::celu` | common_naming_differences | medium |
| `at::conv_transpose2d` | `paddle::experimental::conv2d_transpose` | conv_transpose_alias | high |
| `at::conv_transpose3d` | `paddle::experimental::conv3d_transpose` | conv_transpose_alias | high |
| `at::dropout` | `paddle::experimental::dropout` | common_naming_differences | medium |
| `at::elu` | `paddle::experimental::elu` | common_naming_differences | medium |
| `at::gelu` | `paddle::experimental::gelu` | common_naming_differences | medium |
| `at::grid_sampler` | `paddle::experimental::grid_sample` | common_naming_differences | medium |
| `at::group_norm` | `paddle::experimental::group_norm` | common_naming_differences | medium |
| `at::hardswish` | `paddle::experimental::hardswish` | common_naming_differences | medium |
| `at::hardtanh` | `paddle::experimental::hardtanh` | common_naming_differences | medium |
| `at::huber_loss` | `paddle::experimental::huber_loss` | common_naming_differences | medium |
| `at::instance_norm` | `paddle::experimental::instance_norm` | common_naming_differences | medium |
| `at::layer_norm` | `paddle::experimental::layer_norm` | common_naming_differences | medium |
| `at::leaky_relu` | `paddle::experimental::leaky_relu` | common_naming_differences | medium |
| `at::log_sigmoid` | `paddle::experimental::logsigmoid` | common_naming_differences | medium |
| `at::log_softmax` | `paddle::experimental::log_softmax` | common_naming_differences | medium |
| `at::pow` | `paddle::experimental::elementwise_pow` | paddle_elementwise_alias | high |
| `at::relu` | `paddle::experimental::relu` | common_naming_differences | medium |
| `at::selu` | `paddle::experimental::selu` | common_naming_differences | medium |
| `at::softmax` | `paddle::experimental::softmax` | common_naming_differences | medium |

**总计发现 29 个别名候选**

## Kernel 已注册但未暴露到 api.h

| PyTorch API | Paddle Kernel 文件 | CPU | GPU |
|-------------|-------------------|-----|-----|
| `at::unbind` |  | 是 | 是 |
| `at::argsort` |  | 是 | 是 |
| `at::batch_norm` |  | 是 | 是 |
| `at::cummax` |  | 是 | 是 |
| `at::cummin` |  | 是 | 是 |
| `at::fractional_max_pool2d` |  | 是 | 是 |
| `at::fractional_max_pool3d` |  | 是 | 是 |
| `at::kthvalue` |  | 是 | 是 |
| `at::lstm` |  | 是 | 是 |
| `at::lu_unpack` |  | 是 | 是 |
| `at::median` |  | 是 | 是 |
| `at::mode` |  | 是 | 是 |
| `at::nanmedian` |  | 是 | 是 |
| `at::nll_loss` |  | 是 | 是 |
| `at::norm` |  | 是 | 是 |
| `at::qr` |  | 是 | 是 |
| `at::rms_norm` |  | 否 | 是 |
| `at::svd` |  | 是 | 否 |
| `at::topk` |  | 是 | 是 |
| `at::unique_consecutive` |  | 是 | 是 |

**总计发现 20 个 kernel_only 候选**

## 详细批次结果

### P0_exact_match

**总数**: 66

无需关注

### P1_name_diff

**总数**: 77

**需关注的 API**:

| API | 当前状态 | 验证状态 | 备注 | 别名候选 |
|-----|---------|---------|------|----------|
| `at::hardswish` | - | verified_api_h_only |  | hardswish(medium) |
| `at::hardtanh` | - | verified_api_h_only |  | hardtanh(medium) |
| `at::pow` | - | verified_api_h_only |  | elementwise_pow(high) |
| `at::relu` | - | verified_api_h_only |  | relu(medium) |

### P2_paddle_参数更多

**总数**: 44

**需关注的 API**:

| API | 当前状态 | 验证状态 | 备注 | 别名候选 |
|-----|---------|---------|------|----------|
| `at::dropout` | - | verified_api_h_only |  | dropout(medium) |
| `at::selu` | - | verified_api_h_only |  | selu(medium) |
| `at::_fft_c2r` | - | alias_candidate |  | fft_c2r(high) |
| `at::_fft_r2c` | - | alias_candidate |  | fft_r2c(high) |
| `at::_logcumsumexp` | - | alias_candidate |  | logcumsumexp(high) |
| `at::conv_transpose2d` | - | alias_candidate |  | conv2d_transpose(high) |
| `at::conv_transpose3d` | - | alias_candidate |  | conv3d_transpose(high) |

### P2_torch_参数更多

**总数**: 22

**需关注的 API**:

| API | 当前状态 | 验证状态 | 备注 | 别名候选 |
|-----|---------|---------|------|----------|
| `at::elu` | - | verified_api_h_only |  | elu(medium) |
| `at::huber_loss` | - | verified_api_h_only |  | huber_loss(medium) |
| `at::instance_norm` | - | verified_api_h_only |  | instance_norm(medium) |
| `at::layer_norm` | - | verified_api_h_only |  | layer_norm(medium) |
| `at::log_softmax` | - | verified_api_h_only |  | log_softmax(medium) |
| `at::softmax` | - | verified_api_h_only |  | softmax(medium) |
| `at::_log_softmax` | - | alias_candidate |  | log_softmax(high) |
| `at::_softmax` | - | alias_candidate |  | softmax(high) |
| `at::_standard_gamma` | - | alias_candidate |  | standard_gamma(high) |

### P2_输入参数类型不一致

**总数**: 40

**需关注的 API**:

| API | 当前状态 | 验证状态 | 备注 | 别名候选 |
|-----|---------|---------|------|----------|
| `at::celu` | - | verified_api_h_only |  | celu(medium) |
| `at::gelu` | - | verified_api_h_only |  | gelu(medium) |
| `at::group_norm` | - | verified_api_h_only |  | group_norm(medium) |
| `at::leaky_relu` | - | verified_api_h_only |  | leaky_relu(medium) |
| `at::unbind` | - | kernel_only |  |  |
| `at::_fft_c2c` | - | alias_candidate |  | fft_c2c(high) |
| `at::_stack` | - | alias_candidate |  | stack(high) |
| `at::grid_sampler` | - | alias_candidate |  | grid_sample(medium) |

### P2_返回参数类型不一致

**总数**: 25

**需关注的 API**:

| API | 当前状态 | 验证状态 | 备注 | 别名候选 |
|-----|---------|---------|------|----------|
| `at::aminmax` | - | yaml_only |  |  |
| `at::argsort` | - | kernel_only |  |  |
| `at::batch_norm` | - | kernel_only |  |  |
| `at::cummax` | - | kernel_only |  |  |
| `at::cummin` | - | kernel_only |  |  |
| `at::fractional_max_pool2d` | - | kernel_only |  |  |
| `at::fractional_max_pool3d` | - | kernel_only |  |  |
| `at::kthvalue` | - | kernel_only |  |  |
| `at::lstm` | - | kernel_only |  |  |
| `at::lu_unpack` | - | kernel_only |  |  |
| `at::median` | - | kernel_only |  |  |
| `at::mode` | - | kernel_only |  |  |
| `at::nanmedian` | - | kernel_only |  |  |
| `at::nll_loss` | - | kernel_only |  |  |
| `at::norm` | - | kernel_only |  |  |
| `at::qr` | - | kernel_only |  |  |
| `at::rms_norm` | - | kernel_only |  |  |
| `at::svd` | - | kernel_only |  |  |
| `at::topk` | - | kernel_only |  |  |
| `at::unique_consecutive` | - | kernel_only |  |  |

### P3_alias

**总数**: 13

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
