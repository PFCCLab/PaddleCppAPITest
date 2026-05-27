# PyTorch C++ API → Paddle C++ API 映射表维护工具

本目录包含生成和维护 `cpp_api_mapping_cn.md` 映射表的核心脚本及辅助模块。

---

## 目录结构

```
doc/mapping/
├── README.md                          # 本文档
├── cpp_api_mapping_cn.md              # 主映射表（Markdown）
├── cpp_api_alias_mapping.json         # 别名映射配置
├── cpp_api_alias_candidates.json      # 别名候选（供 Agent 审核）
├── generate_cpp_api_mapping.py        # [核心] 生成映射表
├── verify_api_mapping.py              # [核心] 验证映射表
├── fix_mapping.py                     # [核心] 自动修复映射表
├── discover_cpp_api_aliases.py        # [核心] 发现别名候选
├── generate_comprehensive_report.py   # [辅助] 生成综合验证报告
├── verify/                            # 验证工具模块
│   ├── pytorch_tracer.py              # PyTorch 实现链追踪
│   ├── paddle_tracer.py               # Paddle 实现链追踪
│   └── alias_detector.py              # 别名检测规则
├── verification_output/               # 验证报告输出目录
└── cpp_*_diff/                        # 差异对比文档目录（8类）
```

---

## 环境依赖

- Python 3.8+
- PyTorch C++ 头文件（`libtorch/include/ATen/ops`）
- Paddle 源码（`paddle/phi/api/include/api.h` 及 compat 层）
- PyTorch 源码（可选，用于 kernel 实现追踪验证）

### 路径配置

脚本优先从**环境变量**读取仓库路径，未设置时使用默认路径。

| 环境变量 | 默认值 | 说明 |
|----------|--------|------|
| `PADDLE_ROOT` | `D:/Lenovo/Paddle` | Paddle 仓库根目录 |
| `PYTORCH_ROOT` | `D:/Lenovo/pytorch` | PyTorch 源码目录 |
| `TORCH_DIR` | `D:/Lenovo/libtorch` | libtorch 安装目录 |

**建议在使用前先设置环境变量**（参考 `add-compat-api` skill）：

```bash
export PADDLE_ROOT=~/Paddle
export PYTORCH_ROOT=~/pytorch
export TORCH_DIR=~/libtorch
```

Windows (PowerShell):
```powershell
$env:PADDLE_ROOT = "D:\Lenovo\Paddle"
$env:PYTORCH_ROOT = "D:\Lenovo\pytorch"
$env:TORCH_DIR = "D:\Lenovo\libtorch"
```

也可通过命令行参数覆盖：
```bash
python generate_cpp_api_mapping.py \
  --libtorch-ops-dir /path/to/libtorch/include/ATen/ops \
  --paddle-api-h /path/to/Paddle/paddle/phi/api/include/api.h
```

---

## 核心脚本

### 1. generate_cpp_api_mapping.py — 生成映射表

**作用**：解析 libtorch 和 Paddle 头文件函数签名，自动生成完整的 `cpp_api_mapping_cn.md` 及 8 类差异对比文档。

**使用场景**：
- 首次生成映射表
- Paddle / PyTorch 版本升级后重新生成
- 别名映射更新后重新生成（别名会影响分类）

```bash
python generate_cpp_api_mapping.py \
  --libtorch-ops-dir D:/Lenovo/libtorch/include/ATen/ops \
  --paddle-compat-dir D:/Lenovo/Paddle/paddle/phi/api/include/compat/ATen/ops \
  --paddle-api-h D:/Lenovo/Paddle/paddle/phi/api/include/api.h \
  --output cpp_api_mapping_cn.md
```

**输出产物**：
- `cpp_api_mapping_cn.md` — 主映射表
- `cpp_args_name_diff/` — 仅参数名不一致的差异文档
- `cpp_args_default_value_diff/` — 参数默认值不一致的差异文档
- `cpp_input_args_type_diff/` — 输入参数类型不一致的差异文档
- `cpp_output_args_type_diff/` — 返回参数类型不一致的差异文档
- `cpp_paddle_more_args/` — paddle 参数更多的差异文档
- `cpp_torch_more_args/` — torch 参数更多的差异文档
- `cpp_api_alias_diff/` — API 别名的差异文档
- `cpp_semantic_mismatch/` — 语义差异的差异文档

**分类逻辑**（按优先级）：
1. API 完全一致 — compat 层头文件存在且签名一致
2. API 别名 — 匹配 `cpp_api_alias_mapping.json` 中的映射
3. 功能缺失 — Paddle `api.h` 和 compat 层均无实现
4. 差异类 — `api.h` 有同名实现，按签名对比结果细分：
   - 返回参数类型不一致 → 输入参数类型不一致 → torch 参数更多 / paddle 参数更多
   → 参数默认值不一致 → 仅参数名不一致 → 仅 API 调用方式不一致（兜底）

---

### 2. verify_api_mapping.py — 验证映射表

**作用**：基于 Step 2-1 方法论验证映射表分类的准确性。追踪 PyTorch 实现链（头文件声明 → native_functions.yaml → kernel 实现文件）和 Paddle 实现链（api.h → ops.yaml → kernel 注册 → compat 层），对比双方实现状态。

**使用场景**：
- 定期验证映射表是否仍然准确
- Paddle / PyTorch 版本升级后确认兼容性
- 批量发现映射表中的分类漂移

```bash
# 验证指定批次
python verify_api_mapping.py --batch P0_exact_match
python verify_api_mapping.py --batch P1_name_diff
python verify_api_mapping.py --batch P4_missing

# 验证单个 API
python verify_api_mapping.py --op abs

# 验证全部（分批次）
python verify_api_mapping.py --all

# 限制每批次验证数量（快速测试）
python verify_api_mapping.py --batch P0_exact_match --limit 10
```

**输出产物**（写入 `verification_output/`）：
- `verification_results_<batch>_<timestamp>.json` — 结构化验证结果
- `verification_report_<batch>_<timestamp>.md` — 人类可读报告

**验证状态说明**：
| 状态 | 含义 |
|------|------|
| `verified_compat` | compat 层已实现，签名一致 |
| `verified_api_h_only` | `api.h` 有实现但 compat 层未封装 |
| `alias_candidate` | 发现可能的别名映射 |
| `kernel_only` | kernel 已注册但 `api.h` 未暴露 |
| `truly_missing` | 确实无实现 |
| `yaml_only` | 只有 YAML 配置，无 kernel 注册 |

---

### 3. fix_mapping.py — 自动修复映射表

**作用**：根据验证结果自动修复 `cpp_api_mapping_cn.md` 中的高置信度问题。

**使用场景**：
- `verify_api_mapping.py` 发现分类错误后批量修复
- 清理重复条目

```bash
python fix_mapping.py
```

**可自动修复的场景**：
- 映射表中有重复条目 → 删除重复
- `verified_api_h_only` API 被错误分类到差异类 → 修正分类

---

### 4. discover_cpp_api_aliases.py — 发现别名候选

**作用**：自动发现 PyTorch 与 Paddle 之间的候选别名映射，输出 JSON 供 Agent 审核。

**使用场景**：
- 生成或更新 `cpp_api_alias_mapping.json`
- 发现 PyTorch / Paddle 版本升级后新增的别名

```bash
python discover_cpp_api_aliases.py \
  --libtorch-ops-dir D:/Lenovo/libtorch/include/ATen/ops \
  --paddle-api-h D:/Lenovo/Paddle/paddle/phi/api/include/api.h \
  --output cpp_api_alias_candidates.json
```

**检测规则**：
1. 去掉 PyTorch 前缀 `_` 后的名称与 Paddle 函数名匹配
2. 命名风格翻转：`conv_transposeNd` ↔ `convNd_transpose`
3. 已知语义映射：`range` → `arange`
4. 字符串相似度（Levenshtein ratio ≥ 0.85）作为补充发现

---

## 标准工作流

### 工作流 A：从头生成映射表（首次或版本升级后）

```bash
# 1. 生成映射表
python generate_cpp_api_mapping.py --output cpp_api_mapping_cn.md

# 2. 发现别名候选（路径从环境变量自动读取）
python discover_cpp_api_aliases.py --output cpp_api_alias_candidates.json

# 3. Agent 审核候选别名，更新 cpp_api_alias_mapping.json

# 4. 重新生成（别名会参与分类）
python generate_cpp_api_mapping.py --output cpp_api_mapping_cn.md
```

### 工作流 B：定期验证与修复

```bash
# 1. 验证各批次
python verify_api_mapping.py --batch P0_exact_match
python verify_api_mapping.py --batch P1_name_diff
python verify_api_mapping.py --batch P2_paddle_more_args

# 2. 自动修复高置信度问题
python fix_mapping.py

# 3. 生成综合报告
python generate_comprehensive_report.py
```

---

## Agent 源码审核

脚本验证仅覆盖**表层信息**（头文件签名、kernel 文件路径、重复条目）。

**核心数学语义是否一致，必须由 Agent 逐一阅读 C++ 实现文件确认。**

审核维度：
- 核心数学运算（`std::abs` vs `std::abs`）
- 数据类型处理（`AT_DISPATCH_*` vs `if constexpr`）
- 空张量处理（`numel() == 0` 检查）
- 精度累积（中间 float 缓冲）
- 非连续张量处理（`is_contiguous()`）
- 异常/断言（`TORCH_CHECK` vs `PADDLE_ENFORCE`）
- in-place 限制（complex 禁止等）

审核产物：`manual_review_report_P<N>.md`

---

## 注意事项

1. **脚本只辅助定位 kernel 文件，语义审核由 Agent 完成**
2. **generate_cpp_api_mapping.py 会覆盖现有映射表**，执行前建议备份
3. 验证状态 `kernel_only` 的 API **不要**从别名映射中移除（kernel 已注册但 `api.h` 未暴露）
4. 每次自动修复前备份 `cpp_api_mapping_cn.md` 和 `cpp_api_alias_mapping.json`
5. 验证后确保总数不变（1096 → 1096）
