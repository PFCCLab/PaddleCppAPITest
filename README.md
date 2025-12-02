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
cmake ../ -DTORCH_DIR=<libtorch path> -G Ninja
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

## 代码风格

项目已配置以下代码风格工具：
- **clang-format**: C++ 代码格式化
- **flake8**: Python 代码检查
- **pre-commit**: Git 提交前检查
