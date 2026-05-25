---
name: api-mapping-updater
description: '定期触发验证并自动更新 API 映射表。基于 verify_api_mapping.py 全量验证，检测分类漂移，自动修复高置信度问题，低置信度进入人工审核队列。'
argument-hint: '可选批次名（P0/P1/P2/P3/P4/P5/all），不传则全量验证'
---

# API 映射表定期更新 Skill

基于 Step 2-1 追踪验证的映射表自动维护工作流，定期检测 PyTorch/Paddle 更新导致的映射分类漂移。

## 上游调用上下文

| 上游调用方 | 调用时机 | 期望传入字段 | 处理策略 |
|-----------|---------|-------------|---------|
| cron / schedule | 定期触发（如每周） | `batch=all` | 全量验证 + 自动修复 + 生成报告 |
| 用户手动 | 映射表疑似过时 | `batch=P4_missing` 等 | 指定批次验证 |
| CI/CD | Paddle/PyTorch 版本升级后 | `batch=all` | 全量验证确认兼容性 |

## 何时使用

- 定期（每周/每月）自动验证映射表准确性
- Paddle 或 PyTorch 版本升级后验证映射是否仍有效
- 批量发现映射表中的分类漂移
- 自动修复高置信度的映射错误

## 输入参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `batch` | string | `all` | 验证批次：P0/P1/P2/P3/P4/P5/all |
| `auto_fix` | bool | `true` | 是否自动修复高置信度问题 |
| `output_dir` | string | `doc/verification_output` | 报告输出目录 |
| `libtorch_ops_dir` | string | `D:/Lenovo/libtorch/include/ATen/ops` | libtorch 头文件路径 |
| `paddle_src_dir` | string | `D:/Lenovo/Paddle` | Paddle 源码路径 |
| `pytorch_src_dir` | string | `D:/Lenovo/pytorch` | PyTorch 源码路径 |

## 输出产物

1. **验证报告** `verification_report_YYYY-MM-DD.md` — 人类可读的差异报告
2. **JSON 结果** `verification_results_*.json` — 结构化验证结果
3. **修复日志** `fix_log_YYYY-MM-DD.md` — 自动修复操作记录
4. **人工审核队列** `manual_review_queue.json` — 需人工确认的条目

## 工作流

### Step 1. 环境检查

确认以下路径存在且可访问：
- `libtorch_ops_dir` — ATen/ops 头文件
- `paddle_src_dir/paddle/phi/api/include/api.h` — Paddle API 声明
- `pytorch_src_dir/aten/src/ATen/native/native_functions.yaml` — PyTorch 原生函数定义

### Step 2. 执行验证

```bash
cd "$PCAT_ROOT/doc"
python verify_api_mapping.py --batch "$batch"
```

### Step 3. 差异检测

对比本次验证结果与历史结果（`verification_output/` 中最近的 JSON 文件）：
- 新增 `truly_missing`（Paddle 侧实现被移除）
- 新增 `verified_api_h_only`（Paddle 新增实现但未封装 compat）
- 新增 `alias_candidate`（发现新别名映射）
- 状态变更（如 `verified_compat` → `truly_missing`）

### Step 4. 自动修复（高置信度）

对以下场景自动修复：

| 场景 | 修复操作 | 置信度 |
|------|---------|--------|
| 别名映射在 Paddle api.h 中无实现 | 从 `cpp_api_alias_mapping.json` 移除 | high |
| 映射表中有重复条目 | 删除重复，保留正确分类 | high |
| `verified_compat` API 的 compat 层文件缺失 | 降级为 `verified_api_h_only` | medium |
| 发现新的 `strip_underscore_prefix` 别名 | 添加到 `cpp_api_alias_mapping.json` | high |

修复后运行 `fix_mapping.py` 更新 `cpp_api_mapping_cn.md`。

### Step 5. 生成报告

```bash
python generate_comprehensive_report.py
```

报告包含：
- 各批次验证统计
- 新发现的问题列表
- 自动修复记录
- 需人工审核的条目

### Step 6. 人工审核队列

以下问题进入人工审核队列（不自动修复）：
- 语义差异判断（API 行为是否真正一致）
- 新的别名候选（非规则匹配）
- `kernel_only` 候选（是否暴露到 api.h）
- 分类争议（如"仅参数名不一致" vs "API 别名"）

### Step 7. 提交 PR

若存在自动修复：
1. `git add doc/`
2. pre-commit 检查
3. `git commit`
4. `git push origin doc`
5. （可选）创建 PR 到 master

## 决策分支

### 分支 A：无差异发现
- 验证结果与上次一致
- 仅更新报告时间戳，不提交代码

### 分支 B：发现高置信度问题
- 执行自动修复
- 提交修复后的映射表

### 分支 C：发现低置信度问题
- 生成人工审核队列
- 不自动修复，等待人工确认

### 分支 D：Paddle/PyTorch 版本升级
- 执行全量验证
- 重点检查 `verified_compat` 类是否仍有效
- 检查新增 API 是否需要映射

## 质量标准

1. **可追溯**：每次验证都有时间戳和完整报告
2. **低风险**：高置信度自动修复才执行，低置信度必须人工审核
3. **可回滚**：每次自动修复前备份 `cpp_api_mapping_cn.md` 和 `cpp_api_alias_mapping.json`
4. **不破坏**：自动修复后验证总数不变（1096 → 1096）

## 常见隐患

### 隐患 1：自动修复过度
**❌ 错误**：自动移除了实际上存在的别名映射（因 api.h 未暴露但 kernel 已注册）
**✅ 要求**：`kernel_only` 的 API 不移除别名映射，仅标记状态

### 隐患 2：验证耗时过长
**❌ 错误**：每次全量验证 1096 个 API，耗时数小时
**✅ 要求**：支持增量验证（仅验证变更分类的 API）

### 隐患 3：报告被覆盖
**❌ 错误**：每次验证覆盖上次的报告
**✅ 要求**：报告文件名带时间戳，保留历史报告

## 推荐触发词

- "定期验证映射表"
- "检查映射表是否有漂移"
- "Paddle 升级后验证映射"
- "全量验证 API 映射"
- "验证 P4 缺失类"
