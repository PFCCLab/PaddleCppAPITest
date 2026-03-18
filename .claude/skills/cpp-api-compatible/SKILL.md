| name | description |
|------|-------------|
| cpp-api-compatible | 在进行Paddle兼容层的bug修复、接口添加、性能优化或对齐精度时使用 |

# 添加或修复Paddle兼容层 C++ API

按照TDD工作流完成Paddle针对PyTorch C++ API的兼容接口对齐工作。

## When to Activate

- 修复Paddle对PyTorch的C++ API兼容层中接口行为与Torch原生行为的差异问题
- 在Paddle兼容层中新增对PyTorch C++ API兼容接口
- 需要让Paddle兼容层的接口输出与Libtorch完全一致时

## Core Principles

### 1. 测试驱动开发 (TDD)
- **先写测试，再写实现**：根据测试用例的预期输出倒推兼容层实现
- **最小化实现**：只实现测试用例需要的最小功能，不做过度设计
- **测试即文档**：测试用例清晰描述了接口的预期行为

### 2. 对齐标准
- **唯一标准**：Paddle兼容层的输出必须与Libtorch原生输出完全一致
- **数值一致性**：包括 shape、dtype、具体数值、异常行为
- **无diff原则**：运行 `./test/result_cmp.sh build` 后输出无 FAILED/SKIPPED/DIFF

### 3. 增量迭代
- 从简单用例开始，逐步覆盖复杂场景
- 每通过一个测试用例就是一个里程碑

## Compatibility Work Steps

### Step 1: 检查、分析接口测试

查看提供的测试文件（测试文件路径在 prompt 中），理解测试用例的预期行为：

```cpp
// 示例测试结构
TEST_F(TensorTest, SomeOperation) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();  // 首个用例创建文件

  // 测试逻辑
  at::Tensor input = at::ones({2, 3}, at::kFloat);
  at::Tensor result = input.some_operation();

  // 输出格式: [dim] [numel] [values...]
  file << std::to_string(result.dim()) << " ";
  file << std::to_string(result.numel()) << " ";
  file.saveFile();
}
```

### Step 2: 运行测试并分析差异

#### 情况A：接口不存在
- 编译时会报链接错误或编译错误
- 查看错误信息确定需要实现的接口

#### 情况B：接口已存在但有差异
- 运行测试脚本分析差异
```bash
cd PaddleCppAPITest/build
rm -rf *
cmake .. \
  -DCMAKE_C_COMPILER=/usr/bin/gcc-9 \
  -DCMAKE_CXX_COMPILER=/usr/bin/g++-9
make -j$(nproc)
cd ..
./test/result_cmp.sh build
```

- 查看差异分析文档（文档路径在 prompt 中提供）
- 差异文档包含：diff的测试用例位置、输出对比、初步问题分析

### Step 3: 添加或修复接口

#### 实现新接口
- 根据测试输出进行最小化实现
- 无需添加注释
- 重新运行测试确保通过

#### 修复已有接口
- 按差异文档分析问题根因
- 最小化修改兼容层实现
- 确保测试输出结果和Torch完全一致

```bash
cd PaddleCppAPITest
./test/result_cmp.sh build
```
运行脚本输出无FAILED/SKIPPED/DIFF即可。

### Step 4: 代码优化

- 删掉无用的死代码
- 优化实现方式、提升性能
- 增强代码可读性
- 添加必要的简短注释

### Step 5: 优化后验证

- 测试输出文件中无 fail、skipp、diff 作为达标的唯一标准
- 未达标请返回 Step 4 修复

## Working Patterns

### 1. 单步调试模式
每次修改后立即运行测试，快速验证：
```bash
cd PaddleCppAPITest/build
make -j$(nproc)
cd ..
./test/result_cmp.sh build
```

### 2. 对比分析模式
当出现 diff 时，按以下顺序排查：
1. 检查测试用例输出格式是否一致
2. 检查边界条件处理是否一致
3. 检查异常抛出时机是否一致
4. 检查数值精度是否一致

### 3. 最小复现模式
- 提取产生 diff 的最小测试用例
- 单独调试该用例直到通过
- 再运行完整测试套件

## Common Mistakes

### ❌ WRONG: 盲目添加代码
看到测试失败就添加大量代码，而不分析根本原因。

### ✅ CORRECT: 最小化实现
只添加测试用例明确要求的最小逻辑，通过测试后再考虑优化。

---

### ❌ WRONG: 修改测试用例迎合实现
为了让测试通过而修改测试用例的预期输出。

### ✅ CORRECT: 修改实现对齐测试
测试用例代表正确的预期行为，应该修改兼容层实现来匹配。

---

### ❌ WRONG: 提交未验证的代码
修改完代码不运行测试就直接提交。

### ✅ CORRECT: 验证后再提交
必须运行 `result_cmp.sh` 确认无 diff 后再提交。

## Best Practices

### 1. 从简单用例开始
先通过最基础的测试用例，再逐步覆盖复杂场景。

### 2. 保留中间版本
在进行大幅度修改前，保存可以工作的中间版本，方便回退。

### 3. 精确查找接口定义
- Paddle兼容层接口通常位于 `Paddle/` 目录下
- 使用 grep 搜索相关接口的实现
- 参考已有实现的风格

### 4. 理解差异根因
遇到 diff 时，深入分析差异产生的根本原因：
- 是算法实现不同？
- 是边界条件处理不同？
- 是异常抛出时机不同？

### 5. 增量编译
仅修改测试或兼容层时，使用增量编译节省时间：
```bash
cd build
make -j$(nproc)
```

## Success Metrics

- [ ] **编译通过**：使用 gcc-9 编译无错误
- [ ] **测试通过**：运行 `result_cmp.sh` 无 FAILED/SKIPPED
- [ ] **结果一致**：运行 `result_cmp.sh` 无 DIFF
- [ ] **代码精简**：无死代码、不必要的注释
- [ ] **增量验证**：每次修改后都运行测试确认
