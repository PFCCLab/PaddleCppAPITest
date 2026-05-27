# Step 0 参考：环境检测与自动配置

支撑 [`add-compat-api`](../SKILL.md) 与 [`fix-compat-api`](../../fix-compat-api/SKILL.md) 的 Step 0。覆盖 GPU 探测、克隆、libtorch 下载、Paddle 仓库提示。

## 变量约定

- `PADDLE_ROOT`（默认 `~/Paddle`）
- `PYTORCH_ROOT`（默认 `~/pytorch`）
- `PCAT_ROOT`（默认 `~/PaddleCppAPITest`）
- `TORCH_DIR`（默认 `~/libtorch`）
- `GITHUB_USER`：优先 `gh api user --jq .login`，fallback `git config user.name`，再 fallback 提示用户输入

## 1) 检测 GPU 与决定 libtorch 变体

```bash
if command -v nvidia-smi &>/dev/null && nvidia-smi -L 2>/dev/null | grep -q GPU; then
  LIBTORCH_VARIANT=cu126        # 与项目 env_and_run.sh 一致（Paddle wheel 也是 cu126）
  echo "Detected NVIDIA GPU → libtorch CUDA 12.6"
else
  LIBTORCH_VARIANT=cpu
  echo "No NVIDIA GPU → libtorch CPU"
fi
```

注：`nvidia-smi` 输出里的 `CUDA Version: X.Y` 是驱动支持的最高 CUDA 版本，不是当前 CUDA toolkit；只要 ≥ 12.6 就可以用 cu126 libtorch。低于 12.6（如 CUDA 11.x）的环境本 skill 不覆盖，提示用户。

## 2) PaddleCppAPITest（fork 工作流）

```bash
if [ ! -d "$PCAT_ROOT/.git" ]; then
  GITHUB_USER=$(gh api user --jq .login 2>/dev/null || git config user.name)
  [ -z "$GITHUB_USER" ] && { echo "请提供 GitHub 用户名"; exit 1; }
  git clone "https://github.com/$GITHUB_USER/PaddleCppAPITest.git" "$PCAT_ROOT"
  cd "$PCAT_ROOT"
  git remote add upstream https://github.com/PFCCLab/PaddleCppAPITest.git
fi
```

如果 fork 不存在（clone 失败），提示用户先到 https://github.com/PFCCLab/PaddleCppAPITest fork。

## 3) pytorch（upstream，仅供参考）

```bash
if [ ! -d "$PYTORCH_ROOT/.git" ]; then
  git clone --depth 1 https://github.com/pytorch/pytorch.git "$PYTORCH_ROOT"
fi
```

提示：浅克隆仍约 500MB-1GB，确认磁盘空间充足后再执行。

## 4) libtorch

```bash
if [ ! -f "$TORCH_DIR/lib/libtorch.so" ]; then
  case "$LIBTORCH_VARIANT" in
    cu126)
      URL="https://xly-devops.bj.bcebos.com/PaddleCPPApiTest/libtorch-shared-with-deps-2.9.1%2Bcu126.zip" ;;
    cpu)
      URL="https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-2.9.1%2Bcpu.zip" ;;
  esac
  mkdir -p "$(dirname "$TORCH_DIR")"
  wget -q --show-progress "$URL" -O /tmp/libtorch.zip
  unzip -q /tmp/libtorch.zip -d "$(dirname "$TORCH_DIR")"
  rm /tmp/libtorch.zip
fi
```

CUDA URL 来源：项目 `env_and_run.sh`（已验证可用）。

## 5) Paddle 仓库（不自动克隆）

```bash
if [ ! -d "$PADDLE_ROOT/.git" ]; then
  echo "未检测到 Paddle 仓库（$PADDLE_ROOT）。"
  echo "请先 fork PaddlePaddle/Paddle，然后："
  echo "  git clone https://github.com/$GITHUB_USER/Paddle.git $PADDLE_ROOT"
  echo "  cd $PADDLE_ROOT"
  echo "  git remote add upstream https://github.com/PaddlePaddle/Paddle.git"
  echo "配置完成后重试。"
  exit 1
fi
```

不自动克隆的原因：Paddle 仓库非常大（数 GB）、编译复杂、用户通常已有自定义构建配置，不宜由 skill 创建。

## 6) Paddle wheel 检测

```bash
if ! python3 -c "import paddle" 2>/dev/null; then
  echo "Paddle Python 包未安装。"
  echo "推荐：在完成 Paddle build 后 pip install \$PADDLE_ROOT/build/python/dist/*.whl --force-reinstall --no-deps"
  echo "或先用 env_and_run.sh 里的预编译 wheel URL 安装。"
fi
```

不自动 `pip install` 的原因：wheel 版本必须与 `$PADDLE_ROOT/build` 输出对齐，盲装会和后续 ctest 验证脱节。

## 失败退避

任何一步失败（除"提示用户"以外）：

- 打印失败步骤与错误消息
- 提示用户手动处理
- **不要进入主流程**（避免在缺失依赖下跑出无意义的失败）

## 安全约定

- 克隆、下载、解压这类**本地、可逆**操作可在用户已知意图下直接执行
- **不要主动改用户已存在仓库的 `remote` 配置**（remote 是用户工作流，意外覆盖会丢失工作）
- **不要自动 `pip install`**（wheel 与 build 版本必须严格对齐）
- 任何"克隆 / 下载到一个用户已存在的目录"之前，先用 `[ -d ... ]` / `[ -f ... ]` 检测，避免覆盖
