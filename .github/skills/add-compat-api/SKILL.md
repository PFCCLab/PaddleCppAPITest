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

## 主流程（循环执行）

### Step 1. 确定本轮新增接口范围

1. 明确本轮要新增的接口（建议一次聚焦 1-3 个接口）
2. 在 `$PCAT_ROOT/test/` 中定位或补充对应测试（例如 Device 相关）
3. 明确接口行为基线：参数、返回、异常语义

### Step 3. 参考 PyTorch 实现并新增 Paddle compat 接口

1. 在 `$PYTORCH_ROOT` 中查找目标接口实现（声明 + 实现）
2. 在 `$PADDLE_ROOT/paddle/phi/api/include/compat` 中新增接口
3. 在 `$PADDLE_ROOT/test/cpp/compat` 中新增对应测试
4. 保持与 PyTorch 行为一致：
   - 参数语义
   - 返回类型与 dtype/shape
   - 异常触发时机

### Step 4. 编译 Paddle 并跑兼容测试

```bash
cd "$PADDLE_ROOT/build"
ninja -j"$(nproc)"
ctest -R "ATen|c10|torch"
```

若此步失败，先修复 Paddle 侧编译或测试问题，再继续。

### Step 5. 安装新 wheel

```bash
pip uninstall paddlepaddle-gpu -y
pip install "$PADDLE_ROOT"/build/python/dist/*.whl
```

### Step 6. 回到 PaddleCppAPITest 复编并复测

```bash
cd "$PCAT_ROOT/build"
ninja -j"$(nproc)"

cd "$PCAT_ROOT"
bash test/result_cmp.sh ./build/
```

### Step 7. 判定是否继续循环

- 若新增接口相关用例仍有 `FAILED/SKIPPED/DIFF`：回到 Step 3，进入下一轮
- 若新增接口相关用例通过：进入收尾步骤

## 分支决策

### 分支 A：PaddleCppAPITest 编译失败（接口缺失）

- 优先补齐 compat 声明与最小实现
- 回到 Step 3 完善接口与测试后，再执行 Step 4-6 验证

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
- 文档已更新：新增接口迭代记录（以及相关专题文档）

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

## 推荐执行节奏

- 每轮只处理一组强相关接口，避免一次修改过大
- 每轮都完整执行“编译→测试→安装→回归”闭环
- 以新增接口相关用例通过作为验收基线
