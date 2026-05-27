---
name: add-compat-api
description: '循环新增 Paddle 对 PyTorch C++ 兼容接口。Use when: 有新增 PyTorch C++ API 需求时，在 Paddle compat 层新增接口与测试，并执行双仓编译、ctest、wheel 安装和回归验证。'
argument-hint: '传入 Paddle 仓库路径与 PyTorch 仓库路径（可选：PaddleCppAPITest 路径）'
---

# Add Compat API Loop

用于在 Paddle 兼容层中持续新增 PyTorch C++ API：
先补测试，再实现新接口，持续编译和回归，直到新增接口相关用例通过。

## 输入参数

- `PADDLE_ROOT`: Paddle 仓库路径，例如 `~/Paddle`
- `PYTORCH_ROOT`: PyTorch 仓库路径，例如 `~/pytorch`
- `PCAT_ROOT`: PaddleCppAPITest 路径，默认 `~/PaddleCppAPITest`
- `TORCH_DIR`: libtorch 路径，默认 `~/libtorch`

建议先设置：

```bash
PADDLE_ROOT=~/Paddle
PYTORCH_ROOT=~/pytorch
PCAT_ROOT=~/PaddleCppAPITest
TORCH_DIR=~/libtorch
```

## 适用场景

- 有新增 PyTorch C++ API 需要在 Paddle compat 层实现
- `bash test/result_cmp.sh ./build/` 仍出现 `FAILED/SKIPPED/DIFF`
- 需要新增 Device、Tensor、c10、ATen 等 compat 接口和测试

## Step 0. 环境检测与自动配置

进入主流程之前，按以下顺序检测并按需配置环境。完整命令模板与 fallback 决策见 [`references/Step0.md`](references/Step0.md)。

1. **检测 NVIDIA GPU**（决定 libtorch CPU/CUDA 版本，用 `nvidia-smi`）
2. **PaddleCppAPITest**（fork 工作流，缺失则自动克隆并配置 `origin`/`upstream`）
3. **pytorch**（upstream，缺失则浅克隆，仅供参考）
4. **libtorch**（缺失则下载并解压；URL 按上一步检测结果选 CPU / cu126）
5. **Paddle 仓库**（**不自动克隆**——缺失时提示用户手动 fork + 克隆 + 配置 upstream，并暂停等待）
6. **Paddle wheel**（缺失则提示用户安装；不自动 `pip install`，因为版本须与 Paddle build 输出一致）

> 安全约定：克隆、下载、`pip install` 这类"本地、可逆"操作可在用户已知意图下直接执行；但**不要主动改用户已存在仓库的 `remote`**（remote 是用户工作流，意外覆盖会丢失工作）。

## 主流程（循环执行）

### Step 1. 确定本轮新增接口范围

1. 明确本轮要新增的接口（建议一次聚焦 1-3 个接口）
2. 在 `$PCAT_ROOT/test/` 中定位或补充对应测试（例如 Device 相关）
3. 明确接口行为基线：参数、返回、异常语义

### Step 2. 参考 PyTorch 实现并新增 Paddle compat 接口

1. 先在 `$TORCH_DIR` 中查找目标接口声明，然后在 `$PYTORCH_ROOT` 中查找目标接口实现。
   追踪方法参考 [references/Step2-1.md](references/Step2-1.md) 与
   [references/Step2-2.md](references/Step2-2.md)。
2. 在 `$PADDLE_ROOT/paddle/phi/api/include/compat` 中新增接口
3. 在 `$PADDLE_ROOT/test/cpp/compat` 中新增对应测试
4. **同时**在 `$PCAT_ROOT/test/` 下新增/扩展跨框架对比测试。
   测试规范见 [compatibility-testing](../compatibility-testing/SKILL.md)
   （命名空间 `at::test`、`<OpName>Test` 类、`write_<op>_result_to_file`、
   Shape 四档与 Dtype 四基础类型覆盖、异常用 `std::exception` 不取 `e.what()`、
   空格分隔单行输出格式、新算子 checklist），本文档不复述。

   调用该 skill 时传入：

   - `PCAT_ROOT=$PCAT_ROOT`
   - 算子名：本轮要新增的接口（如 `at::abs`、`at::abs_`）
   - 覆盖目标：Shape 标量/小/大/边界，Dtype kFloat/kDouble/kInt/kLong
   - 输出路径：`$PCAT_ROOT/test/<分类>/<OpName>Test.cpp`

   返回后照其骨架写入测试文件，并对照其 checklist 自检强制项（标 `*` 项）。
5. 保持与 PyTorch 行为一致：
   - 参数语义
   - 返回类型与 dtype/shape
   - 异常触发时机

### Step 3. 编译 Paddle 并跑兼容测试

```bash
cd "$PADDLE_ROOT/build"
ninja -j"$(nproc)"
ctest -R "ATen|c10|torch"
```

若此步失败，先修复 Paddle 侧编译或测试问题，再继续。

### Step 4. 安装新 wheel

```bash
pip install "$PADDLE_ROOT"/build/python/dist/*.whl --force-reinstall --no-deps
```

### Step 5. 回到 PaddleCppAPITest 复编并复测

```bash
cd "$PCAT_ROOT/build"
ninja -j"$(nproc)"

cd "$PCAT_ROOT"
bash test/result_cmp.sh ./build/
```

### Step 6. 判定是否继续循环

- 若新增接口相关用例仍有 `FAILED/SKIPPED/DIFF`：回到 Step 2，进入下一轮
- 若新增接口相关用例通过：进入收尾步骤

## 分支决策

### 分支 A：PaddleCppAPITest 编译失败（接口缺失）

- 优先补齐 compat 声明与最小实现
- 回到 Step 2 完善接口与测试后，再执行 Step 3-5 验证

### 分支 B：Paddle 编译通过但 `ctest` 失败

- 先修复回归，再安装 wheel
- 不要跳过 `ctest` 直接进入回归对比

### 分支 C：新增接口行为不一致

- 对照 PyTorch 检查 dtype 推导、边界输入、异常行为
- 仅处理本轮新增接口导致的问题，不处理历史遗留差异

### 分支 D：wheel 安装异常

- 确认 `build/python/dist/` 下有最新 whl
- 必要时先清理旧包后重装

## 完成标准

同时满足以下条件才算完成：

- `$PADDLE_ROOT/build` 下 `ninja -j"$(nproc)"` 成功
- `$PADDLE_ROOT/build` 下 `ctest -R "ATen|c10|torch"` 通过
- `$PCAT_ROOT` 下 `bash test/result_cmp.sh ./build/` 中新增接口相关用例通过
- 文档已通过 [compat-doc-authoring](../compat-doc-authoring/SKILL.md) 归档，且其 Step 5 校验全部通过

## 文档收尾要求

完成新增后，按以下固定模板更新文档（推荐记录到
`$PCAT_ROOT/doc/` 下的专题文档，必要时同步相关文档）：

```markdown
## 对齐迭代记录（YYYY-MM-DD）

### 1) 接口变更
- 接口名：
- 变更类型：新增
- Paddle 兼容层位置：
- 参考 PyTorch 位置：

### 2) 测试覆盖
- 测试文件：
- 新增/修改用例：
- 覆盖点：shape / dtype / 边界 / 异常

### 3) 新增接口验证结果
- 新增前状态（缺失）：
- 新增后验证结果：
- 关键行为说明：

### 4) 构建与回归结果
- Paddle 编译：通过/失败
- ctest (ATen|c10|torch)：通过/失败
- result_cmp：无差异/仍有差异

### 5) 未完成项与下一轮计划
- 未完成接口：
- 下一轮优先级：
```

## 文档归档（调用 compat-doc-authoring）

按上节"对齐迭代记录"模板填好五段后，**不要**直接写入 `$PCAT_ROOT/doc/`。
本 skill 不维护"✅🔧❌ 状态符号 / 优先级 P0–P3 / 测试用例 checkbox /
兼容性统计表 / 关键差异说明"这些格式规范，统一交给
[compat-doc-authoring](../compat-doc-authoring/SKILL.md) 完成入库与校验。

调用该 skill 时传入：

- `PCAT_ROOT=$PCAT_ROOT`
- 调用模式：`append-to-existing`（追加到已有专题文档）
- 目标文档：`$PCAT_ROOT/doc/<topic>.md`
- 上游模板名：`对齐迭代记录`
- 已填段落：粘贴上节"对齐迭代记录（YYYY-MM-DD）"五段完整填好的 Markdown

下游必须完成的额外工作（本 skill 不重复约束）：

- 把本轮新增接口加入文档 API 对比表，状态符号选 `✅` 或 `🔧`
- 回填 `## 兼容性统计` 表的三行数字
- 对每个标 `🔧` 的条目，在"关键差异说明"中补一小节
- 发布前按 compat-doc-authoring 的 Step 5 校验 checklist 全项过审

只有下游确认"格式合规"后，本轮才算完成。

## Step 7. 提交 commit 并创建 PR

闭环验证通过且文档已回填后，按以下流程提交。完整命令模板见 [`references/Step7.md`](references/Step7.md)。

1. **从 fork 主分支 checkout 新分支**（`git fetch upstream && git checkout -B add/<api>-<YYYYMMDD> upstream/develop`）
2. **commit 改动**（commit message 首行使用 `[Cpp API Compatibility] <对齐迭代记录标题>`）
3. **征求用户同意后 push 到 fork**（`git push origin <branch>`——这是发出去的动作，**push 前必须明确询问用户**）
4. **征求用户同意后 `gh pr create` 到 upstream**（`--repo PaddlePaddle/Paddle --base develop`——同样需要用户确认）
5. **如果还改了 PCAT 测试**：在 `$PCAT_ROOT` 上重复 1-4 步，`--repo PFCCLab/PaddleCppAPITest --base master`
6. **等待 CI 完成并按结果分流**（`gh pr checks <PR_NUM> --watch`）
   - CI 通过 → 等待 reviewer，本流程结束
   - CI 失败 → 调查失败是否由本 PR 引起（命令与判断标准见 [`references/Step7.md`](references/Step7.md) 第 6 节）：
     - **是** → 返回 [`Step 2`](#step-2-参考-pytorch-实现并新增-paddle-compat-接口) 修复；同一分支上 commit + push（**push 仍需用户同意**），PR 自动更新，不发新 PR
     - **否** → `gh pr comment <PR_NUM> --body "/re-run all-failed"` 重新触发 CI，回到本步骤继续 watch

> 安全约定：**`git push` 与 `gh pr create` 每一次执行前必须征求用户同意**（不是"在整个 Step 7 开始时一次性确认"，而是这两条命令各确认一次）。修复后的 push 同样适用——同一分支不豁免。这与系统提示"对影响他人的动作要逐次确认"一致。

## 推荐执行节奏

- 每轮只处理一组强相关接口，避免一次修改过大
- 每轮都完整执行“编译→测试→安装→回归”闭环
- 以新增接口相关用例通过作为验收基线
