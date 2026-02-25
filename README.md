# PaddleCPPAPITest

## 项目概述

一个用于验证 PaddlePaddle 和 PyTorch API 兼容性的 C++ 测试框架项目，依托 API 单元测试和持续守护流程，降低第三方库从 PyTorch 迁移到 PaddlePaddle 的技术门槛。

## 依赖要求

### 系统要求

- CMake >= 3.18
- C++17
- Python 3.x (用于检测 PaddlePaddle)

### 第三方库依赖

- PaddlePaddle (通过 Python 包自动检测)
- PyTorch (libtorch，默认路径: /usr/lib/libtorch/)
- Google Test (源码依赖，项目自动下载和构建)

## 快速开始

### 1. 克隆项目

```bash
git clone <project-url>
cd PaddleCPPAPITest
```

### 2. 配置构建环境

```bash
mkdir build && cd build
cmake ../PaddleCPPAPITest -DTORCH_DIR=<libtorch path> -G Ninja
```

### 3. 编译项目

```bash
ninja
```

### 4. 运行测试

#### 运行 PaddlePaddle 测试

```bash
./paddle/paddle_TensorTest
```

#### 运行 PyTorch 测试

```bash
./torch/torch_TensorTest
```

#### 运行所有测试

```bash
ctest
```

#### 运行对比脚本

```bash
cd .. && ./test/result_cmp.sh build
```

## 代码风格

项目已配置以下代码风格工具：

- **clang-format**: C++ 代码格式化
- **Ruff**: Python 代码检查
- **pre-commit**: Git 提交前检查

## 兼容性测试 Skill

项目内置了 Claude Code Skill（`.claude/skills/compatibility-testing/`），用于规范化 Paddle/PyTorch C++ API 兼容性测试的编写流程。该 Skill 在涉及 API 测试编写、跨框架对比验证等场景时自动激活。

### 适用场景

- 新增 ATen 算子的兼容性测试（如 `test/ops/AbsTest.cpp`）
- 排查 Paddle 与 PyTorch 在特定算子上的行为差异
- 扩展现有测试的 shape / dtype 覆盖范围

### 规范内容

| 类别 | 说明 |
|------|------|
| 文件结构 | 统一的头文件引用、`at::test` 命名空间、`FileManerger` 结果输出 |
| Shape 覆盖 | 标量 `{}`、小 shape、大 shape（10000+）、含零维度、非连续内存布局 |
| Dtype 覆盖 | `kFloat` / `kDouble` / `kInt` / `kLong` / `kBool` 等 8 种标准类型 |
| 算子分类 | Creation、Math、Shape、Indexing、Comparison、Reduction 六大类 |
| 对比流程 | 同源代码分别编译 Torch / Paddle 版本，通过输出文件 diff 验证一致性 |

完整指南见 [SKILL.md](.claude/skills/compatibility-testing/SKILL.md)。

