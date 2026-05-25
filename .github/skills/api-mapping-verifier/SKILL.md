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

### Step 1. 执行验证

单 API 验证：
```bash
cd "$PCAT_ROOT/doc"
python verify_api_mapping.py --op "$op"
```

批次验证：
```bash
python verify_api_mapping.py --batch "$batch"
```

### Step 2. 结果分析

解析验证状态，输出分析：

| 验证状态 | 含义 | 修复建议 |
|----------|------|----------|
| `verified_compat` | compat 层已实现 | 分类正确，无需操作 |
| `verified_api_h_only` | api.h 有实现，compat 层未封装 | 可考虑添加 compat 层封装 |
| `alias_candidate` | 发现别名映射候选 | 添加到 `cpp_api_alias_mapping.json` |
| `kernel_only` | kernel 已注册但未暴露到 api.h | 需 Paddle 侧暴露到 api.h |
| `truly_missing` | 真正缺失 | 确认是否真的无对应实现 |
| `yaml_only` | 只有 YAML 配置，无 kernel 注册 | 检查是否开发中 |

### Step 3. Step 2-1 追踪详情

对 `verified_compat` 和 `verified_api_h_only` 的 API，展示追踪详情：

**PyTorch 侧**：
- libtorch 头文件声明位置
- 是否 dispatcher 转发
- native_functions.yaml schema 与 dispatch
- kernel 实现文件位置

**Paddle 侧**：
- api.h 签名
- ops.yaml 配置
- kernel 注册状态（CPU/GPU）
- compat 层封装状态

### Step 4. 修复建议

根据验证结果给出具体修复操作：

**场景 A：`verified_api_h_only` → 应添加 compat 层**
- 参考同类型 API 的 compat 层实现模板
- 建议创建 `paddle/phi/api/include/compat/ATen/ops/<op>.h`

**场景 B：`alias_candidate` → 应更新别名映射**
- 给出具体的 JSON 条目
- 建议添加到 `cpp_api_alias_mapping.json`

**场景 C：分类错误 → 应修正映射表**
- 指出当前分类和应修正的分类
- 给出 `fix_mapping.py` 的修复参数

### Step 5. 执行修复（用户确认后）

用户确认后执行修复操作：
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
