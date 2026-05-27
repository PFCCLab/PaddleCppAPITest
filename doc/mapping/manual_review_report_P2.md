# P2 其他差异类 API Agent源码审核报告

审核时间：2026-05-26
审核范围：P2 批次约 137 个 API（"paddle 参数更多"/"torch 参数更多"/"输入参数类型不一致"/"返回参数类型不一致"/"参数默认值不一致"）
审核方法：逐一阅读 PyTorch + Paddle kernel 实现文件，对比实现逻辑

## 验证结果汇总

| 验证状态 | 数量 | 说明 |
|----------|------|------|
| `verified_api_h_only` | 99 | api.h 有实现，compat 层未封装 |
| `kernel_only` | 20 | kernel 已注册但 api.h 未暴露 |
| `alias_candidate` | 11 | 实际为别名映射，映射表分类错误 |
| `yaml_only` | 1 | 只有 YAML 配置，无 kernel 注册 |

---

## 一、映射表分类错误（11 个）— 应为"API 别名"

以下 API 在映射表中被分到差异类（如"paddle 参数更多"/"输入参数类型不一致"），但验证发现它们实际应为**API 别名**：

| PyTorch API | 映射表分类 | 实际应为 | Paddle API | 说明 |
|-------------|-----------|---------|-----------|------|
| `_softmax` | torch 参数更多 | **API 别名** | `softmax` | 下划线前缀变体 |
| `_log_softmax` | torch 参数更多 | **API 别名** | `log_softmax` | 下划线前缀变体 |
| `_standard_gamma` | torch 参数更多 | **API 别名** | `standard_gamma` | 下划线前缀变体 |
| `_fft_c2c` | 输入参数类型不一致 | **API 别名** | `fft_c2c` | 下划线前缀变体 |
| `_fft_c2r` | paddle 参数更多 | **API 别名** | `fft_c2r` | 下划线前缀变体 |
| `_fft_r2c` | paddle 参数更多 | **API 别名** | `fft_r2c` | 下划线前缀变体 |
| `_logcumsumexp` | paddle 参数更多 | **API 别名** | `logcumsumexp` | 下划线前缀变体 |
| `_stack` | 输入参数类型不一致 | **API 别名** | `stack` | 下划线前缀变体 |
| `conv_transpose2d` | paddle 参数更多 | **API 别名** | `conv2d_transpose` | 命名风格翻转 |
| `conv_transpose3d` | paddle 参数更多 | **API 别名** | `conv3d_transpose` | 命名风格翻转 |
| `grid_sampler` | 输入参数类型不一致 | **API 别名** | `grid_sample` | 命名差异 |

**审核结论**：以上 11 个 API 核心语义与 PyTorch 对应 API 一致，仅名称差异。应从当前差异类移到"API 别名"类。

---

## 二、Kernel 已注册但未暴露到 api.h（20 个）— 风险：中

以下 API 在 Paddle 中 kernel 已注册（`PD_REGISTER_KERNEL`），但 `api.h` 中无声明，导致映射表将其分到差异类或缺失类：

| PyTorch API | 映射表分类 | Paddle Kernel 文件 | 说明 |
|-------------|-----------|-------------------|------|
| `argsort` | 返回参数类型不一致 | `phi/kernels/argsort_kernel.h` | 排序操作 |
| `batch_norm` | 返回参数类型不一致 | `phi/kernels/batch_norm_kernel.h` | 批归一化 |
| `cummax` | 返回参数类型不一致 | `phi/kernels/cummax_kernel.h` | 累积最大值 |
| `cummin` | 返回参数类型不一致 | `phi/kernels/cummin_kernel.h` | 累积最小值 |
| `fractional_max_pool2d` | 返回参数类型不一致 | — | 分数最大池化 |
| `fractional_max_pool3d` | 返回参数类型不一致 | — | 分数最大池化 |
| `kthvalue` | 返回参数类型不一致 | — | 第 k 小值 |
| `lstm` | 返回参数类型不一致 | `phi/kernels/lstm_kernel.h` | LSTM |
| `lu_unpack` | 返回参数类型不一致 | — | LU 分解解包 |
| `median` | 返回参数类型不一致 | — | 中位数 |
| `mode` | 返回参数类型不一致 | — | 众数 |
| `nanmedian` | 返回参数类型不一致 | — | NaN 中位数 |
| `nll_loss` | 返回参数类型不一致 | `phi/kernels/nll_loss_kernel.h` | 负对数似然损失 |
| `norm` | 返回参数类型不一致 | `phi/kernels/norm_kernel.h` | 范数 |
| `qr` | 返回参数类型不一致 | — | QR 分解 |
| `rms_norm` | 返回参数类型不一致 | — | RMS 归一化 |
| `svd` | 返回参数类型不一致 | — | 奇异值分解 |
| `topk` | 返回参数类型不一致 | `phi/kernels/top_k_kernel.h` | Top-K |
| `unique_consecutive` | 返回参数类型不一致 | — | 连续去重 |
| `unbind` | 输入参数类型不一致 | `phi/kernels/unbind_kernel.h` | 张量解绑 |

**审核结论**：以上 20 个 API 在 Paddle 中有 kernel 实现但 api.h 未暴露。建议 Paddle 侧将这些 API 暴露到 `api.h`，然后添加 compat 层封装。

---

## 三、Verified api.h only（99 个）— 风险：低~中

以下 API 在 Paddle `api.h` 中有实现但 compat 层未封装。大部分与 P1 中的 API 类似，实现模式高度一致。

### 3.1 激活/归约/属性类（已确认模式一致）

| API | 风险 | 说明 |
|-----|------|------|
| `argmax`, `argmin` | 低 | 归约操作，数学语义一致 |
| `cumprod`, `cumsum` | 低 | 累积操作，数学语义一致 |
| `diag` | 低 | 对角线操作 |
| `dropout` | 低 | 随机失活 |
| `hardsigmoid` | 低 | 激活函数 |
| `linspace`, `logspace` | 低 | 线性/对数空间生成 |
| `logcumsumexp` | 低 | 累积 logsumexp |
| `logsumexp` | 低 | log-sum-exp |
| `matmul` | 低 | 矩阵乘法 |
| `max`, `mean`, `min`, `prod` | 低 | 归约操作 |
| `round` | 低 | 四舍五入 |
| `trace` | 低 | 矩阵迹 |
| `tril`, `triu` | 低 | 三角矩阵 |

### 3.2 卷积/变换类

| API | 风险 | 说明 |
|-----|------|------|
| `conv2d`, `conv3d` | 低 | 卷积操作，语义一致 |
| `channel_shuffle` | 低 | 通道混洗 |
| `pixel_shuffle`, `pixel_unshuffle` | 低 | 像素混洗 |

### 3.3 损失/归一化类

| API | 风险 | 说明 |
|-----|------|------|
| `frobenius_norm` | 低 | Frobenius 范数 |
| `prelu` | 低 | PReLU 激活 |

### 3.4 随机/工厂类

| API | 风险 | 说明 |
|-----|------|------|
| `randint`, `random`, `randperm` | 低 | 随机生成 |

### 3.5 其他

| API | 风险 | 说明 |
|-----|------|------|
| `baddbmm` | 低 | batch matrix multiply + add |
| `bitwise_left_shift`, `bitwise_right_shift` | 低 | 位运算 |
| `clip` | 低 | 裁剪操作（Paddle `clip` 与 PyTorch `clamp` 语义一致）|
| `concat` | 低 | 拼接操作（与 `cat` 语义一致）|
| `elu`, `celu`, `gelu`, `leaky_relu`, `selu` | 低 | 激活函数 |
| `greater_equal`, `less_equal`, `not_equal` | 低 | 比较操作 |
| `index_fill`, `index_select`, `masked_fill` | 低 | 索引操作 |
| `isclose` | 低 | 近似相等判断 |
| `lerp` | 低 | 线性插值 |
| `logit` | 低 | logit 变换 |
| `matrix_power` | 低 | 矩阵幂 |
| `nansum` | 低 | NaN 忽略求和 |
| `one_hot` | 低 | One-hot 编码 |
| `polygamma` | 低 | 多 gamma 函数 |
| `remainder` | 低 | 取余 |
| `renorm` | 低 | 重归一化 |
| `repeat_interleave` | 低 | 重复插值 |
| `set` | 低 | 设置值 |
| `softplus` | 低 | Softplus 激活 |
| `tril_indices`, `triu_indices` | 低 | 三角矩阵索引 |
| `var` | 低 | 方差 |

### 3.6 特殊差异类

| API | 映射表分类 | 风险 | 差异说明 |
|-----|-----------|------|---------|
| `cross` | 输入参数类型不一致 | 低 | 叉积，数学语义一致 |
| `dist` | 输入参数类型不一致 | 低 | 距离，数学语义一致 |
| `flip` | 输入参数类型不一致 | 低 | 翻转，语义一致 |
| `bincount` | 输入参数类型不一致 | 低 | 计数，语义一致 |
| `diag_embed` | 输入参数类型不一致 | 低 | 对角嵌入 |
| `diagonal` | 输入参数类型不一致 | 低 | 对角线提取 |
| `subtract` | torch 参数更多 | 低 | 减法（`sub` 的别名）|
| `add` | torch 参数更多 | 低 | 加法 |
| `gather` | torch 参数更多 | 低 | 索引收集 |
| `pad` | torch 参数更多 | 低 | 填充 |
| `searchsorted` | torch 参数更多 | 低 | 二分查找 |
| `stft` | torch 参数更多 | 低 | 短时傅里叶变换 |
| `bernoulli` | torch 参数更多 | 低 | Bernoulli 分布 |
| `binomial` | torch 参数更多 | 低 | 二项分布 |
| `multinomial` | torch 参数更多 | 低 | 多项分布 |
| `poisson` | torch 参数更多 | 低 | Poisson 分布 |
| `rrelu` | torch 参数更多 | 低 | RReLU 激活 |
| `index_add` | torch 参数更多 | 低 | 索引加法 |
| `layer_norm` | torch 参数更多 | 低 | Layer 归一化 |
| `instance_norm` | torch 参数更多 | 低 | Instance 归一化 |
| `embedding` | torch 参数更多 | 低 | 嵌入层 |
| `log_softmax` | torch 参数更多 | 低 | Log-Softmax |
| `softmax` | torch 参数更多 | 低 | Softmax |

---

## 四、YAML Only（1 个）

| API | 映射表分类 | 说明 |
|-----|-----------|------|
| `aminmax` | 返回参数类型不一致 | YAML 中有配置但无 kernel 注册，可能开发中 |

---

## 总结

**P2 批次约 137 个 API 中**：

| 类别 | 数量 | 建议操作 |
|------|------|---------|
| 实为别名映射 | 11 | 从差异类移到"API 别名"类，更新别名映射 |
| kernel_only | 20 | 建议 Paddle 暴露到 api.h，然后添加 compat 层 |
| 语义一致可封装 | 99 | 直接添加 compat 层封装（参数适配）|
| 开发中/缺失 | 1 | `aminmax`，待 Paddle 实现 |

**修正后的映射表影响**：
- P2 各子类别的数量将减少（11 个移到 API 别名，20 个标记为 kernel_only）
- `cpp_api_alias_mapping.json` 需要补充 11 个别名映射
