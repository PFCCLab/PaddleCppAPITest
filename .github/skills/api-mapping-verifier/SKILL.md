---
name: api-mapping-verifier
description: '按需验证单个或批次 API 的映射准确性，基于 Step 2-1 追踪方法深入 PyTorch 实现链路。支持发现问题、给出修复建议、执行修复。'
argument-hint: '目标 API 名（如 abs）或批次名（P0/P1/P2/P3/P4/P5）'
---

# API 映射表按需验证 Skill

基于 Step 2-1 方法论的按需验证工作流，用于验证单个 API 或批次 API 的映射分类是否准确。

## 何时使用

- 怀疑某个 API 的映射分类不正确
- 新增 compat 接口后验证映射表是否需要更新
- 排查具体 API 的兼容性差异
- 验证 Paddle 新增实现是否已正确映射

## 输入参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `op` | string | — | 单个 API 名称（如 `abs`） |
| `batch` | string | — | 验证批次：P0/P1/P2/P3/P4/P5 |
| `pytorch_src_dir` | string | `D:/Lenovo/pytorch` | PyTorch 源码路径（用于追踪 kernel） |
| `paddle_src_dir` | string | `D:/Lenovo/Paddle` | Paddle 源码路径 |
| `libtorch_ops_dir` | string | `D:/Lenovo/libtorch/include/ATen/ops` | libtorch 头文件路径 |

**注意**：`op` 和 `batch` 二选一，同时传入时优先 `op`。

## 输出产物

1. **验证结果 JSON** — 结构化追踪记录
2. **终端输出** — 人类可读的结果摘要
3. **修复建议** — 基于验证状态的具体操作建议

## 工作流

> **核心原则**：脚本只负责**定位**和**提取表层信息**（头文件签名、kernel 文件路径），**具体 C++ 实现逻辑的审核由 Agent 逐一阅读源码完成**。

### Step 1. 脚本定位（自动化）

脚本通过 `verify_api_mapping.py` 完成以下工作：

```bash
cd "$PCAT_ROOT/doc/mapping"
# 单 API
python verify_api_mapping.py --op "$op"
# 批次
python verify_api_mapping.py --batch "$batch"
```

脚本输出：
- 验证状态（verified_compat / verified_api_h_only / alias_candidate / ...）
- **PyTorch kernel 实现文件路径**（如 `aten/src/ATen/native/UnaryOps.cpp:543`）
- **Paddle kernel 实现文件路径**（如 `paddle/phi/kernels/cpu/abs_kernel.cc:25`）
- 头文件签名对比结果

### Step 2. Agent 源码审核（核心步骤）

**根据脚本定位的文件路径，逐一阅读 C++ 实现文件**，对比以下维度：

#### 2.1 必须阅读的源码位置

| 框架 | 文件类型 | 典型路径示例 |
|------|---------|-------------|
| PyTorch | CPU kernel | `aten/src/ATen/native/cpu/UnaryOpsKernel.cpp` |
| PyTorch | CUDA kernel | `aten/src/ATen/native/cuda/AbsKernel.cu` |
| PyTorch | 高层封装 | `aten/src/ATen/native/UnaryOps.cpp` |
| Paddle | CPU kernel | `paddle/phi/kernels/cpu/abs_kernel.cc` |
| Paddle | CUDA kernel | `paddle/phi/kernels/gpu/abs_kernel.cu` |
| Paddle | Functor | `paddle/phi/kernels/funcs/activation_functor.h` |

#### 2.2 审核检查清单

| 检查项 | PyTorch 关注点 | Paddle 关注点 | 差异影响 |
|--------|--------------|--------------|---------|
| **核心数学运算** | 实际调用的函数（`std::abs`, `sum_stub` 等） | Functor 中的运算（`Acos<T>`, `SumFunctor`） | 数学语义是否一致 |
| **数据类型处理** | `AT_DISPATCH_*` 宏、complex 分支 | `if constexpr`、float16 特化 | dtype 支持范围是否一致 |
| **空张量处理** | `TensorIterator.numel() == 0` | `if (x.numel() == 0)` | 边界行为是否一致 |
| **精度累积** | `should_use_acc_buffer`、中间 float 缓冲 | `Cast` 到 float32 | 低精度输入结果是否一致 |
| **非连续张量** | `TensorIterator` 自动处理 | 是否检查 `is_contiguous()` | 布局敏感操作是否有差异 |
| **异常/断言** | `TORCH_CHECK`、`AT_ASSERT` | `PADDLE_ENFORCE` | 异常触发时机是否一致 |
| **in-place 限制** | complex 禁止、维度检查 | 是否由上层框架处理 | in-place 语义是否一致 |
| **向量化实现** | AVX512、Vectorized | Eigen 向量化 | 性能差异，不影响语义 |

#### 2.3 审核结论模板

对每 个 API，填写以下审核记录：

```markdown
## at::<op_name> 源码审核

**PyTorch 实现**（文件:行号）：
```cpp
// 粘贴核心实现代码
```

**Paddle 实现**（文件:行号）：
```cpp
// 粘贴核心实现代码
```

**审核结论**：

| 维度 | PyTorch | Paddle | 差异 |
|------|---------|--------|------|
| 核心运算 | ... | ... | ... |
| 数据类型 | ... | ... | ... |
| 边界条件 | ... | ... | ... |
| 异常语义 | ... | ... | ... |

**风险评级**：低 / 中 / 高
**理由**：...
```

### Step 3. 头文件签名对比（脚本辅助）

脚本对比签名层面的差异：

| 验证状态 | 含义 | 修复建议 |
|----------|------|----------|
| `verified_compat` | compat 层已实现 | 需Agent 审核实现逻辑 |
| `verified_api_h_only` | api.h 有实现，compat 层未封装 | 可考虑添加 compat 层封装 |
| `alias_candidate` | 发现别名映射候选 | 需 Agent 确认别名语义等价 |
| `kernel_only` | kernel 已注册但未暴露到 api.h | 需 Paddle 侧暴露到 api.h |
| `truly_missing` | 真正缺失 | 确认是否真的无对应实现 |

### Step 4. 修复建议与执行

**场景 A：`verified_api_h_only` → 应添加 compat 层**
- 参考同类型 API 的 compat 层实现模板
- 建议创建 `paddle/phi/api/include/compat/ATen/ops/<op>.h`

**场景 B：分类错误 → 应修正映射表**
- 结合源码审核结论，判断当前分类是否准确
- 给出 `fix_mapping.py` 的修复参数

**场景 C：发现实现语义差异 → 更新文档**
- 在映射表备注中注明差异（如 `at::abs` 的非连续张量处理差异）
- 更新差异文档

执行修复：
1. 更新 `cpp_api_alias_mapping.json`（如需）
2. 运行 `fix_mapping.py` 更新映射表
3. 删除/更新相关差异文档
4. 重新运行验证确认修复成功

## 决策分支

### 分支 A：验证通过（verified_compat）
- 输出追踪详情供参考
- 无需修复

### 分支 B：api.h 有实现但 compat 层未封装
- 给出 compat 层封装模板
- 用户确认后创建 compat 头文件

### 分支 C：发现别名候选
- 给出别名映射建议
- 用户确认后更新别名映射文件

### 分支 D：真正缺失
- 检查是否有组合实现方案
- 记录到"功能缺失"跟踪列表

## 使用示例

```bash
# 验证单个 API
/api-mapping-verifier --op abs

# 验证批次
/api-mapping-verifier --batch P0_exact_match

# 验证并自动修复
/api-mapping-verifier --batch P1_name_diff --auto_fix
```

## 质量标准

1. **追踪完整**：展示从 libtorch 声明到 kernel 实现的完整链路
2. **建议具体**：给出可直接执行的修复操作，不只是描述问题
3. **可验证**：修复后必须重新验证确认

## 与 api-mapping-updater 的区别

| | api-mapping-verifier | api-mapping-updater |
|--|---------------------|---------------------|
| 触发方式 | 按需（用户指定 API） | 定期（cron/schedule） |
| 验证范围 | 单个 API 或单批次 | 全量或大批次 |
| 修复策略 | 用户确认后执行 | 高置信度自动修复 |
| 输出 | 终端摘要 + 追踪详情 | 完整报告 + 审核队列 |
| 典型场景 | 排查具体问题 | 定期维护 |
