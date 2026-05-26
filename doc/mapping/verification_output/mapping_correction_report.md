# API 映射表验证修正报告

生成时间: 2026-05-25

## 验证范围

基于 Step 2-1 方法论（libtorch 头文件声明 → native_functions.yaml schema/dispatch → kernel 实现），对映射表中的 API 进行了系统性验证。

已验证批次：
- **P0**: "API 完全一致"类（66 个 API）
- **P1**: "仅参数名不一致"类（79 个 API）
- **P3**: "API 别名"类（18 个 API）
- **P4 抽样**: "功能缺失"类（抽样验证）

## 验证结果概览

| 批次 | 总数 | verified_compat | verified_api_h_only | alias_candidate | kernel_only | truly_missing |
|------|------|-----------------|---------------------|-----------------|-------------|---------------|
| P0 | 66 | 66 | 0 | 0 | 0 | 0 |
| P1 | 79 | 0 | 77 | 2 | 0 | 0 |
| P3 | 18 | 0 | 0 | 13 | 1 | 4 |
| **合计** | **163** | **66** | **77** | **15** | **1** | **4** |

## 发现的映射表分类问题

### 1. "仅参数名不一致"类中的误分类

以下 API 在映射表中被标记为"仅参数名不一致"，但 Paddle `api.h` 中没有同名实现，实际应为"API 别名"：

| PyTorch API | Paddle 实际 API | 当前分类 | 建议修正 |
|-------------|----------------|----------|----------|
| `at::_conj` | `paddle::experimental::conj` | 仅参数名不一致 | **API 别名** |
| `at::log_sigmoid` | `paddle::experimental::logsigmoid` | 仅参数名不一致 | **API 别名** |

### 2. "API 别名"类中的误分类

以下 API 在映射表中被标记为"API 别名"，但验证发现 Paddle `api.h` 中没有对应实现：

| PyTorch API | 映射表中的 Paddle API | 验证结果 | 当前分类 | 建议修正 |
|-------------|----------------------|----------|----------|----------|
| `at::_aminmax` | `paddle::experimental::aminmax` | truly_missing | API 别名 | **功能缺失** |
| `at::_unique` | `paddle::experimental::unique` | truly_missing | API 别名 | **功能缺失** |
| `at::max_pool2d_with_indices` | `paddle::experimental::max_pool2d_with_index` | truly_missing | API 别名 | **功能缺失** |
| `at::max_pool3d_with_indices` | `paddle::experimental::max_pool3d_with_index` | truly_missing | API 别名 | **功能缺失** |
| `at::range` | `paddle::experimental::arange` | kernel_only | API 别名 | **kernel_only**（需暴露到 api.h） |

### 3. 验证确认有效的别名映射

以下 API 别名映射经验证，Paddle `api.h` 中确实存在对应实现：

| PyTorch API | Paddle API | 置信度 |
|-------------|-----------|--------|
| `at::_conj` | `paddle::experimental::conj` | high |
| `at::_fft_c2c` | `paddle::experimental::fft_c2c` | high |
| `at::_fft_c2r` | `paddle::experimental::fft_c2r` | high |
| `at::_fft_r2c` | `paddle::experimental::fft_r2c` | high |
| `at::_log_softmax` | `paddle::experimental::log_softmax` | high |
| `at::_logcumsumexp` | `paddle::experimental::logcumsumexp` | high |
| `at::_softmax` | `paddle::experimental::softmax` | high |
| `at::_stack` | `paddle::experimental::stack` | high |
| `at::_standard_gamma` | `paddle::experimental::standard_gamma` | high |
| `at::conv_transpose2d` | `paddle::experimental::conv2d_transpose` | high |
| `at::conv_transpose3d` | `paddle::experimental::conv3d_transpose` | high |
| `at::grid_sampler` | `paddle::experimental::grid_sample` | medium |
| `at::log_sigmoid` | `paddle::experimental::logsigmoid` | medium |

### 4. Kernel 已注册但未暴露到 api.h 的候选

| PyTorch API | Paddle Kernel 状态 | 说明 |
|-------------|-------------------|------|
| `at::range` | CPU: 是, GPU: 是 | kernel 已注册，但 api.h 中无声明。映射到 `arange` 更合适。 |

### 5. "API 完全一致"类验证结果

66 个 API 全部确认为 `verified_compat`，compat 层已实现完整封装。但其中部分 API（如 `at::abs`）底层实现存在已知差异（非连续张量处理策略不同），compat 层已通过警告机制处理。

### 6. "功能缺失"类抽样验证结果

对 539 个非下划线前缀的"功能缺失"API 进行别名扫描，未发现 Paddle `api.h` 中有对应别名实现。确认大部分"功能缺失"分类准确。

抽样验证的几个 API（`layer_norm`、`dropout`、`embedding`）经验证 Paddle 中有完整实现，但映射表中已正确分类到非缺失类别（差异类）。

## 建议的映射表修正操作

### 修正 1：将误分类的 API 移动至正确类别

```
"仅参数名不一致" → "API 别名":
  - _conj → conj
  - log_sigmoid → logsigmoid

"API 别名" → "功能缺失":
  - _aminmax
  - _unique
  - max_pool2d_with_indices
  - max_pool3d_with_indices
```

### 修正 2：更新 cpp_api_alias_mapping.json

添加/更新以下别名映射（仅保留验证确认有效的）：

```json
{
  "_conj": "conj",
  "_fft_c2c": "fft_c2c",
  "_fft_c2r": "fft_c2r",
  "_fft_r2c": "fft_r2c",
  "_log_softmax": "log_softmax",
  "_logcumsumexp": "logcumsumexp",
  "_softmax": "softmax",
  "_stack": "stack",
  "_standard_gamma": "standard_gamma",
  "conv_transpose2d": "conv2d_transpose",
  "conv_transpose3d": "conv3d_transpose",
  "grid_sampler": "grid_sample",
  "log_sigmoid": "logsigmoid"
}
```

### 修正 3：移除无效的别名映射

以下映射在 Paddle `api.h` 中无对应实现，应从别名映射中移除：

```
_aminmax → aminmax (不存在)
_unique → unique (不存在)
max_pool2d_with_indices → max_pool2d_with_index (不存在)
max_pool3d_with_indices → max_pool3d_with_index (不存在)
range → arange (kernel_only, 未暴露到 api.h)
```

## 验证脚本使用说明

```bash
# 验证单个 API
python doc/verify_api_mapping.py --op abs

# 验证特定批次
python doc/verify_api_mapping.py --batch P0_exact_match
python doc/verify_api_mapping.py --batch P1_name_diff
python doc/verify_api_mapping.py --batch P4_missing --limit 100

# 生成综合报告
python doc/generate_comprehensive_report.py
```

## 后续工作建议

1. **修复分类错误**：按上述修正操作更新映射表
2. **扩展验证范围**：对 P2（其他差异类）142 个 API 进行验证
3. **扩展 P4 验证**：对 790 个"功能缺失"API 进行分批完整验证
4. **P5 验证**：确认 `uniform` 的语义差异
5. **持续维护**：将验证脚本集成到 CI/CD 流程中，定期验证映射表准确性
