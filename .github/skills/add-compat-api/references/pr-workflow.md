# Step 7 参考：提交 commit 并创建 PR

支撑 [`add-compat-api`](../SKILL.md) 与 [`fix-compat-api`](../../fix-compat-api/SKILL.md) 的 Step 7。覆盖分支命名、commit、push、PR 创建。

## 触发条件

只在以下条件**全部满足**后才进入本流程：

- `ninja -j"$(nproc)"`（在 `$PADDLE_ROOT/build`）成功
- `ctest -R "ATen|c10|torch"` 通过
- `bash test/result_cmp.sh ./build/`（在 `$PCAT_ROOT`）相关用例通过
- 文档已按三层回填规则更新（见 SKILL.md 的"完成标准 / 文档收尾模板"）

任一未通过 → **不要**进入本流程。

## 1) 从 fork 主分支 checkout 新分支

### Paddle 侧

```bash
cd "$PADDLE_ROOT"
git fetch upstream
# add 流程
git checkout -B "add/<api-name>-$(date +%Y%m%d)" upstream/develop
# fix 流程
git checkout -B "fix/<pr-or-issue-num>-$(date +%Y%m%d)" upstream/develop
```

分支命名规则：
- add：`add/<api-name>-<YYYYMMDD>`（例：`add/abs-20260519`）
- fix：`fix/<pr-num>-<YYYYMMDD>`（例：`fix/78652-20260519`）

> 前提：Paddle 仓库已配置 `origin = <user>/Paddle` 与 `upstream = PaddlePaddle/Paddle`。若 remote 缺失，**提示用户配置后再继续**，不要主动改用户仓库的 remote。

## 2) commit 改动

```bash
git add paddle/phi/api/include/compat/<改动文件>
git add test/cpp/compat/<改动测试>
git commit -m "[Cpp API Compatibility] <对齐迭代记录或 Compat 修复记录的标题>"
```

commit message 首行模板：

- add：`[Cpp API Compatibility] Add <api>` 或 `[Cpp API Compatibility] Align <module>`
- fix：`[Cpp API Compatibility] Fix <api> <行为>`（如 `Fix at::chunk negative dim handling`）

正文（如有）：可粘贴 `doc/mismatch_api_record.md` 对应条目的"问题与根因 / 修复内容"段。

## 3) push 到 fork —— **执行前必须征求用户同意**

```bash
git push origin "$(git rev-parse --abbrev-ref HEAD)"
```

> 安全约定：`git push` 把改动发到 GitHub fork，对用户可见、可被他人引用。**Claude 不得在用户未明确同意的情况下执行 push**。用文字明确询问："是否 push 到 origin fork？"

push 失败时的常见原因：
- 分支名冲突（用 `--force-with-lease` 而非 `--force`，且征求用户同意）
- 远程拒绝（fork 仓库的保护规则）

## 4) `gh pr create` 到 upstream —— **执行前必须征求用户同意**

### Paddle 侧 add

```bash
gh pr create \
  --repo PaddlePaddle/Paddle \
  --base develop \
  --head "$GITHUB_USER:$(git rev-parse --abbrev-ref HEAD)" \
  --title "[Cpp API Compatibility] Add <api>" \
  --body "$(cat <<'EOF'
## 背景
<引用 doc/mismatch_api_record.md 对应"对齐迭代记录"段>

## 改动
- compat 接口：<path>
- 新增测试：<path>

## 验证
- ninja: 通过
- ctest -R "ATen|c10|torch": 通过
- result_cmp: 通过

## 关联
- 上游 PyTorch 实现：<libtorch / native_functions.yaml 位置>
EOF
)"
```

### Paddle 侧 fix

同上，但 `--title` 改为 `[Cpp API Compatibility] Fix <api> ...`，且 `## 关联` 段加一行 `- 修复链接：<原 PR / Actions / comment 链接>`。

> 安全约定：`gh pr create` 直接打开 PR，会通知 reviewer、触发 CI、占用 PR 编号。**Claude 不得在用户未明确同意的情况下执行**。用文字明确询问："是否发 PR 到 PaddlePaddle/Paddle？"

## 5) 如果还改了 PCAT 测试 → PCAT fork 上重复 1-4 步

```bash
cd "$PCAT_ROOT"
git fetch upstream
git checkout -B "test/<api>-$(date +%Y%m%d)" upstream/master
git add test/<改动文件>
git commit -m "test(<api>): align with Paddle compat <api> 行为"
# 同样：先确认再 push
git push origin <branch>
# 同样：先确认再 PR
gh pr create --repo PFCCLab/PaddleCppAPITest --base master ...
```

PCAT PR 通常**引用** Paddle 侧 PR 编号（在 PR body 里加 `Related: PaddlePaddle/Paddle#xxxxx`）。

## 失败处理

- commit 失败（pre-commit hook 等）：修复后**新建** commit（不要 `--amend`），重跑闭环验证
- push 失败：诊断后重试（不要 `--force` 除非用户明确同意）
- gh pr create 失败：检查 `gh auth status`，确认 base/head 分支无误
