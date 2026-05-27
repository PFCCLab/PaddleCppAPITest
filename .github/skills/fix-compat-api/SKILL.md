---
name: fix-compat-api
description: '根据 PR 链接、GitHub Actions 链接或 review/comment 链接定位问题并修复 Paddle C++ compat 接口。Use when: 需要按外部链接驱动修复 c10/ATen/torch 兼容层接口并完成编译、测试和回归。'
argument-hint: '传入一个链接（PR / Actions / review comment）与仓库路径参数'
---

# Fix Compat API From Link

根据外部链接定位兼容性问题来源，执行最小修复并完成验证闭环。
流程参考 `~/Paddle/.humanize` 中的兼容修复实践。

## 输入参数

- `INPUT_LINK`: 必填。三选一：
  - PR 链接：`https://github.com/<org>/<repo>/pull/<number>`
  - Actions 链接：`https://github.com/<org>/<repo>/actions/runs/<run_id>`
  - review/comment 链接：
    - `https://github.com/<org>/<repo>/pull/<number>#discussion_r...`
    - `https://github.com/<org>/<repo>/pull/<number>/files#r...`
    - `https://github.com/<org>/<repo>/pull/<number>#issuecomment-...`
- `PADDLE_ROOT`: 默认 `~/Paddle`
- `PYTORCH_ROOT`: 默认 `~/pytorch`
- `PCAT_ROOT`: 默认 `~/PaddleCppAPITest`
- `TORCH_DIR`: 默认 `~/libtorch`

建议先设置：

```bash
PADDLE_ROOT=~/Paddle
PYTORCH_ROOT=~/pytorch
PCAT_ROOT=~/PaddleCppAPITest
TORCH_DIR=~/libtorch
```

## 适用场景

- 评审意见指向 compat 接口行为不一致
- CI/Actions 报错指向 c10/ATen/torch 兼容层
- 需要按链接快速复现、修复并回归验证

## 链接分流规则

### 分支 A：输入为 PR 链接

1. 提取 PR 编号
2. 收集最新 review comments 与普通 comments（优先 unresolved / 最新）
3. 仅筛选 compat 相关问题（`paddle/phi/api/include/compat`、`test/cpp/compat`）

### 分支 B：输入为 Actions 链接

1. 提取失败 job 与失败步骤
2. 锁定失败测试/文件（优先 c10、ATen、torch compat 相关）
3. 必须反查关联 PR，再定位对应评论与代码上下文

### 分支 C：输入为 review/comment 链接

1. 直接读取评论内容与上下文代码片段
2. 回溯到对应 PR 与文件位置
3. 将该评论作为本轮修复主目标

### 分支 D：链接无法识别

- 明确提示支持的 URL 格式
- 请求用户重新提供可解析链接

### 横向退出条件：根因为接口完全缺失

若 Step 1 解析后发现链接所指根因是 **"Paddle 侧完全没有该接口"**
（而非行为不一致），本 skill 不直接走 Step 2–5。
请退出当前 fix 循环，改用 [add-compat-api](../add-compat-api/SKILL.md) skill
启动新增循环。`add-compat-api` 完成后，回到本 skill 重新输入原链接，
验证问题是否消失。

> 本引用为**说明性引用**，不会自动触发 `add-compat-api`——避免一次输入
> 同时启动两条驱动型循环造成递归。请用户/Claude 明确在两个 skill 间切换。

## Step 0. 环境检测与自动配置

进入主流程之前，按以下顺序检测并按需配置环境。完整命令模板与 fallback 决策见 [`../add-compat-api/references/Step0.md`](../add-compat-api/references/Step0.md)（与 `add-compat-api` 共享同一份 references）。

1. **检测 NVIDIA GPU**（决定 libtorch CPU/CUDA 版本，用 `nvidia-smi`）
2. **PaddleCppAPITest**（fork 工作流，缺失则自动克隆并配置 `origin`/`upstream`）
3. **pytorch**（upstream，缺失则浅克隆，仅供参考）
4. **libtorch**（缺失则下载并解压；URL 按上一步检测结果选 CPU / cu126）
5. **Paddle 仓库**（**不自动克隆**——缺失时提示用户手动 fork + 克隆 + 配置 upstream，并暂停等待）
   - fix 流程通常已有 Paddle 仓库（因为是链接驱动、对已有代码做修复）；若缺失，提示并暂停等待用户配置
6. **Paddle wheel**（缺失则提示用户安装；不自动 `pip install`，因为版本须与 Paddle build 输出一致）

> 安全约定：克隆、下载、`pip install` 这类"本地、可逆"操作可在用户已知意图下直接执行；但**不要主动改用户已存在仓库的 `remote`**（remote 是用户工作流，意外覆盖会丢失工作）。

## 修复流程（循环执行）

### Step 1. 解析需求并定义本轮目标

1. 从链接提取本轮要修复的接口清单（建议 1-3 项）
2. 明确每项接口的语义基线：
   - 参数/返回类型
   - dtype/shape 规则
   - 异常触发时机
3. 约束修复范围：仅处理本轮链接指向问题

### Step 2. 对照 PyTorch 实现

1. 在 `$PYTORCH_ROOT` 查找目标接口声明与实现
2. 记录语义关键点，不照搬无关实现
3. 确认不引入 PyTorch 和 Paddle 都不存在的新设计

### Step 3. 在 Paddle 侧实施最小修复

1. 修改 compat 接口实现：`$PADDLE_ROOT/paddle/phi/api/include/compat`
2. 新增或补充测试：`$PADDLE_ROOT/test/cpp/compat`
3. **若本轮修复涉及 `$PCAT_ROOT/test` 下用例**（链接指向 `result_cmp` diff、
   或修复改变了输出格式 / dtype / 异常路径），按统一规范扩写或新增。
   测试规范见 [compatibility-testing](../compatibility-testing/SKILL.md)，
   本文档不复述。

   调用该 skill 时传入：

   - `PCAT_ROOT=$PCAT_ROOT`
   - 算子名：链接指向的接口
   - 修复点：`dtype 推导` / `异常路径` / `shape 边界` / `API 变体` 等
   - 已有测试路径：`$PCAT_ROOT/test/<分类>/<OpName>Test.cpp`

   返回后只针对修复点新增最小化用例，不重排无关测试。
4. 必要时补充 `$PCAT_ROOT/test` 下其余兼容回归用例
   （按上一调用返回的 checklist 自检强制项）。

### Step 4. 编译与测试验证

```bash
cd "$PADDLE_ROOT/build"
ninja -j"$(nproc)"
ctest -R "ATen|c10|torch"
```

### Step 5. 安装 wheel 并执行外部回归

```bash
pip install "$PADDLE_ROOT"/build/python/dist/*.whl --force-reinstall --no-deps

cd "$PCAT_ROOT/build"
cmake .. -DTORCH_DIR="$TORCH_DIR" -DENABLE_COVERAGE=ON -G Ninja
ninja -j"$(nproc)"

cd "$PCAT_ROOT"
bash test/result_cmp.sh ./build/
```

### Step 6. 判定是否继续循环

- 若本轮链接对应问题仍未消失：回到 Step 2
- 若本轮问题已修复且验证通过：进入收尾

## 完成标准

同时满足以下条件才算完成：

- 链接对应的问题点已被代码与测试覆盖
- `$PADDLE_ROOT/build` 下 `ninja -j"$(nproc)"` 成功
- `ctest -R "ATen|c10|torch"` 全部通过
- `$PCAT_ROOT` 下 `bash test/result_cmp.sh ./build/` 中相关用例通过
- 未引入 PyTorch 与 Paddle 上游都不存在的实现
- 文档已通过 [compat-doc-authoring](../compat-doc-authoring/SKILL.md) 归档，且其 Step 5 校验全部通过

## 文档收尾模板

完成后更新文档（固定写入 `$PCAT_ROOT/doc/` 对应专题文档）：

```markdown
## Compat 修复记录（YYYY-MM-DD）

### 1) 输入链接
- 链接类型：PR / Actions / review comment
- 原始链接：
- 关联 PR（如可定位）：

### 2) 问题与根因
- 问题接口：
- 触发场景：
- 根因说明：

### 3) 修复内容
- Paddle compat 改动文件：
- 新增/修改测试：
- PyTorch 对齐依据：

### 4) 验证结果
- ninja：通过/失败
- ctest -R "ATen|c10|torch"：通过/失败
- result_cmp：通过/失败

### 5) 风险与后续
- 已知风险：
- 后续待办：
```

## 文档归档（调用 compat-doc-authoring）

按上节"Compat 修复记录"模板填好五段后，调用
[compat-doc-authoring](../compat-doc-authoring/SKILL.md) 完成入库与格式校验。

调用该 skill 时传入：

- `PCAT_ROOT=$PCAT_ROOT`
- 调用模式：`append-to-existing`
- 目标文档：`$PCAT_ROOT/doc/<topic>.md`
- 上游模板名：`Compat 修复记录`
- 已填段落：粘贴上节"Compat 修复记录（YYYY-MM-DD）"五段完整填好的 Markdown

下游会做的事：

- 若本次修复改变了 API 对比表中某行的状态（`🔧` ↔ `✅`），由下游负责改表
- 回填 `## 兼容性统计` 数字
- 检查链接编号、PR 编号是否在"备注"列被回填
- 按 Step 5 校验 checklist 全项过审

## Step 7. 提交 commit 并创建 PR

闭环验证通过且文档已回填后，按以下流程提交。完整命令模板见 [`../add-compat-api/references/Step7.md`](../add-compat-api/references/Step7.md)（与 `add-compat-api` 共享同一份 references）。

1. **从 fork 主分支 checkout 新分支**（`git fetch upstream && git checkout -B fix/<pr-or-issue-num>-<YYYYMMDD> upstream/develop`）
2. **commit 改动**（commit message 首行使用 `[Cpp API Compatibility] <Compat 修复记录标题>`）
3. **征求用户同意后 push 到 fork**（`git push origin <branch>`——这是发出去的动作，**push 前必须明确询问用户**）
4. **征求用户同意后 `gh pr create` 到 upstream**（`--repo PaddlePaddle/Paddle --base develop`——同样需要用户确认；若本轮修复源自外部 PR/Actions/comment 链接，PR 描述里应**引用原链接**便于追溯）
5. **如果还改了 PCAT 测试**：在 `$PCAT_ROOT` 上重复 1-4 步，`--repo PFCCLab/PaddleCppAPITest --base master`
6. **等待 CI 完成并按结果分流**（`gh pr checks <PR_NUM> --watch`）
   - CI 通过 → 等待 reviewer，本流程结束
   - CI 失败 → 调查失败是否由本 PR 引起（命令与判断标准见 [`../add-compat-api/references/Step7.md`](../add-compat-api/references/Step7.md) 第 6 节）：
     - **是** → 返回 [`Step 2`](#step-2-对照-pytorch-实现) 修复；同一分支上 commit + push（**push 仍需用户同意**），PR 自动更新，不发新 PR
     - **否** → `gh pr comment <PR_NUM> --body "/re-run all-failed"` 重新触发 CI，回到本步骤继续 watch

> 安全约定：**`git push` 与 `gh pr create` 每一次执行前必须征求用户同意**（不是"在整个 Step 7 开始时一次性确认"，而是这两条命令各确认一次）。修复后的 push 同样适用——同一分支不豁免。这与系统提示"对影响他人的动作要逐次确认"一致。

## 常见错误

- 只修测试输出，不修接口语义
- 跳过 `ctest` 或 `result_cmp` 直接结束
- 为“临时通过”引入上游不存在的实现
- 一次改动过大，导致无法定位回归来源
