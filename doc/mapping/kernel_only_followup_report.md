# P2 批次 API 跟进报告 — kernel_only 修正

审核时间：2026-05-26

---

## 修正说明

P2 Agent源码审核报告中，以下20个API被初步标记为 `kernel_only`（kernel已注册但api.h未暴露）。

**经进一步验证：这20个API实际上全部已在 `api.h` 中有声明，因此应更正为 `verified_api_h_only`（api.h有实现但compat层未封装）。**

验证方法：逐一在 `paddle/phi/api/include/api.h` 中搜索确认。

---

## 20 个 `verified_api_h_only` API 列表

| # | PyTorch API | 原映射表分类 | Paddle `api.h` 签名 | Paddle Kernel 文件 | 风险 | 说明 |
|---|-------------|-----------|-------------------|-------------------|------|------|
| 1 | `argsort` | 返回参数类型不一致 | `argsort(const Tensor& x, int axis)` | `phi/kernels/argsort_kernel.h` | 低 | 排序操作 |
| 2 | `batch_norm` | 返回参数类型不一致 | `batch_norm(...)` | `phi/kernels/batch_norm_kernel.h` | 低 | 批归一化 |
| 3 | `cummax` | 返回参数类型不一致 | `cummax(const Tensor& x, int axis)` | `phi/kernels/cummax_kernel.h` | 低 | 累积最大值 |
| 4 | `cummin` | 返回参数类型不一致 | `cummin(const Tensor& x, int axis)` | `phi/kernels/cummin_kernel.h` | 低 | 累积最小值 |
| 5 | `fractional_max_pool2d` | 返回参数类型不一致 | `fractional_max_pool2d(...)` | `phi/kernels/pool_kernel.h` | 中 | 分数最大池化 |
| 6 | `fractional_max_pool3d` | 返回参数类型不一致 | `fractional_max_pool3d(...)` | `phi/kernels/pool_kernel.h` | 中 | 分数最大池化 |
| 7 | `kthvalue` | 返回参数类型不一致 | `kthvalue(const Tensor& x, int k, int axis)` | `phi/kernels/kthvalue_kernel.h` | 低 | 第 k 小值 |
| 8 | `lstm` | 返回参数类型不一致 | `lstm(...)` | `phi/kernels/lstm_kernel.h` | 中 | LSTM |
| 9 | `lu_unpack` | 返回参数类型不一致 | `lu_unpack(...)` | `phi/kernels/lu_unpack_kernel.h` | 低 | LU 分解解包 |
| 10 | `median` | 返回参数类型不一致 | `median(const Tensor& x, int64_t axis)` | `phi/kernels/median_kernel.h` | 低 | 中位数 |
| 11 | `mode` | 返回参数类型不一致 | `mode(...)` | `phi/kernels/mode_kernel.h` | 低 | 众数 |
| 12 | `nanmedian` | 返回参数类型不一致 | `nanmedian(...)` | `phi/kernels/nanmedian_kernel.h` | 低 | NaN 中位数 |
| 13 | `nll_loss` | 返回参数类型不一致 | `nll_loss(...)` | `phi/kernels/nll_loss_kernel.h` | 低 | 负对数似然损失 |
| 14 | `norm` | 返回参数类型不一致 | `norm(...)` | `phi/kernels/norm_kernel.h` | 低 | 范数 |
| 15 | `qr` | 返回参数类型不一致 | `qr(...)` | `phi/kernels/qr_kernel.h` | 低 | QR 分解 |
| 16 | `rms_norm` | 返回参数类型不一致 | `rms_norm(...)` | `phi/kernels/gpu/rms_norm_cuda_kernel.cu` | 低 | RMS 归一化 |
| 17 | `svd` | 返回参数类型不一致 | `svd(...)` | `phi/kernels/svd_kernel.h` | 低 | 奇异值分解 |
| 18 | `topk` | 返回参数类型不一致 | `topk(...)` | `phi/kernels/top_k_kernel.h` | 低 | Top-K |
| 19 | `unique_consecutive` | 返回参数类型不一致 | `unique_consecutive(...)` | `phi/kernels/unique_consecutive_kernel.h` | 低 | 连续去重 |
| 20 | `unbind` | 输入参数类型不一致 | `unbind(const Tensor& x, int axis)` | `phi/kernels/unbind_kernel.h` | 低 | 张量解绑 |

---

## 修正后的建议操作

**原建议**（P2 报告中）：建议 Paddle 侧将这些 API 暴露到 `api.h`。

**修正后建议**：以上20个API**已在 `api.h` 中暴露**，只需在 compat 层添加封装（参数适配），不需要 Paddle 侧改动。

### compat 层封装优先级

| 优先级 | API 数量 | 列表 | 理由 |
|--------|---------|------|------|
| 高 | 14 | argsort, cummax, cummin, kthvalue, lu_unpack, median, mode, nanmedian, nll_loss, norm, qr, svd, topk, unique_consecutive, unbind | 常用操作，语义清晰，风险低 |
| 中 | 4 | batch_norm, fractional_max_pool2d, fractional_max_pool3d, lstm, rms_norm | 涉及返回多个值或复杂参数，需仔细适配 |

### 封装注意事项

1. **返回多个值的 API**（`cummax`, `cummin`, `kthvalue`, `median`, `mode`, `nanmedian`, `topk`, `svd`, `qr`, `lu_unpack`, `unique_consecutive`）：
   - PyTorch 通常返回 `(values, indices)` 元组或 `(Q, R)` 元组
   - Paddle `api.h` 中可能也有类似的返回结构
   - compat 层需处理元组/结构体的映射

2. **batch_norm**：
   - PyTorch 返回 `(output, save_mean, save_var)` 三元组
   - 训练/推理模式返回不同

3. **lstm**：
   - PyTorch 返回 `(output, (hidden, cell))` 复杂结构
   - 需要仔细映射隐藏状态

4. **rms_norm**：
   - Paddle 实现可能在 `fusion` 目录下，是融合kernel
   - 需要确认与 PyTorch `rms_norm` 的语义一致性

---

## 总结

- **原分类**：20 个 `kernel_only` ❌
- **修正后**：20 个 `verified_api_h_only` ✅
- **下一步**：直接为这些 API 添加 compat 层封装，无需等待 Paddle 侧暴露
