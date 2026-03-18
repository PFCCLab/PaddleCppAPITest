| name | description |
|------|-------------|
| compatibility-testing | PaddlePaddle 与 PyTorch C++ API 兼容性测试开发，确保兼容接口的完全对齐。 |

# Cpp API 兼容性测试开发

## When to Activate

- 编写或扩展 `PaddleCppAPITest/test` 下的兼容性测试。
- 验证 Paddle 兼容层与 PyTorch 对同一 API 的行为一致性。
- 定位某个接口在两个框架间的输出差异。

## Core Principles

项目构建系统通过 `create_paddle_tests()` 将同一份源码编译出 `torch_*` 和 `paddle_*` 两套可执行文件。测试结果通过文本形式序列化到本地，通过 diff 进行验证。

### 基础文件结构
测试文件统一位于 `PaddleCppAPITest/test`，命名为 `<OpName>Test.cpp`（如 `AbsTest.cpp`）：
1. **命名空间**：固定为 `at::test`。
2. **全局参数**：使用 `extern paddle_api_test::ThreadSafeParam g_custom_param;` 获取当前运行的文件名标识。
3. **测试夹具**：继承自 `::testing::Test`。

### 结果输出格式 (序列化)
为保证 diff 结果严谨，需实现专用的结果写入函数。
- **强制要求**：输出需使用空格分隔的纯文本 `[ndim] [numel] [size_0] ... [val_0] ...`。
- **IO规则**：当前测试文件内，**首个用例使用 `createFile()`** 创建文件，**后续所有用例使用 `openAppend()`** 追加。

```cpp
static void write_op_result_to_file(FileManerger* file, const at::Tensor& result) {
  *file << std::to_string(result.dim()) << " ";
  *file << std::to_string(result.numel()) << " ";
  float* data = result.data_ptr<float>(); // 需根据 result.scalar_type() 动态分发
  for (int64_t i = 0; i < result.numel(); ++i) {
    *file << std::to_string(data[i]) << " ";
  }
}
```

### 覆盖率与鲁棒性要求

验证 API 时必须确保足够的边界测试。禁止为了满足覆盖率堆砌无效实例化，应调用真实的逻辑和方法。不允许使用 `#if USE_PADDLE_API` 做条件分支。

### Shape 与边界维度覆盖
每个算子测试 **必须** 覆盖以下场景：
1. **标量**：`{}` (0-d tensor，非 `{1}`)。
2. **常规小/大 Tensor**：如 `{4}`, `{2, 3}` / `{10000}`, `{10, 20, 30}`。
3. **极端边界**：含零维度 (`{0}`, `{2, 0}`)、全一维度 (`{1, 1, 1}`)、非连续 Tensor (如 `transpose()`/`as_strided()` 产物)。

### Dtype 类型覆盖
基础类型至少覆盖：`kFloat` (float32)、`kDouble` (float64 base-line)、`kInt` (int32)、`kLong` (int64)。

### 异常与边界行为捕获
非法输入引发的异常两端可能不同（报错信息跨度广或一端不抛出），捕获并输出异常行为以供 Diff：
```cpp
try {
  at::Tensor res = at::some_op(invalid_tensor);
  // ... 写入成功结果
} catch (const c10::Error& e) {
  file << "c10::Error: " << e.what(); // 优先捕获 c10::Error
} catch (const std::exception& e) {
  file << "exception: " << e.what();
}
```

### 注意事项
- **CMakeLists 限制**：一般开发不要擅自修改 `CMakeLists.txt`。如有新增组件/目录或改进意见，请优先**向开发者提问确认**后再执行。
- **增量编译策略**：节约生命，仅修改 `test` 时请直接进 `build` 敲击 `make -j$(nproc)`。
- **约定输出**：所有流程均成功后请简述测试开发结果（如新增或修改了哪些文件、覆盖了哪些测试场景、是否发现 Diff 以及后续计划）输出一份 `md` 文档供开发者或其他大模型 review。

## Compatibility Work Steps

### Step 1: 自动添加测试
- 查看PaddleCppAPITest/api_coverage_report.txt中的： "## ❌ 未测试的API列表" 。
- 取下方第一个 "###" 对应的接口，在test目录下确定没有相关测试后添加测试。
- 按上述要求开发测试，完成后运行bash PaddleCppAPITest/coverage/analyze_api.sh。
- 检查相关接口是否在 "## ❌ 未测试的API列表" 下消失，消失请直接进行Step 2。
- 如果你添加了对应接口的测试用例，并且运行了analyze_api.sh，相关接口仍显示未测试，请修改coverage/analyze_api_coverage.py脚本中的接口列表，确保它能正确的识别你添加的测试用例，同时在工作结束后向开发者汇报。
### Step 2: 验证测试表现

完成开发后，按以下严格的流程式验证测试表现。

- 编译阶段

```bash
conda activate paddle
cd PaddleCppAPITest/build
# ⚠️ 注意：如果是初次编译或涉及 CMakeLists.txt 改动，需清理缓存：
rm -rf *

# ⚠️ 注意：如果 **仅仅修改了测试代码 (.cpp)**，请跳过上述 rm -rf 操作以节省时间直接 make！

cmake .. \
  -DCMAKE_C_COMPILER=/usr/bin/gcc-9 \
  -DCMAKE_CXX_COMPILER=/usr/bin/g++-9
make -j$(nproc)
```
> **排错**：如果编译失败，请根据报错信息在对应测试文件中补充头文件或修正语法，复杂问题请到Paddle仓和Libtorch仓考察后修复测试文件，不允许直接修复Paddle源码，所有修复同样需满足上述的开发规范要求。

- 运行测试与分析对比

借助对比脚本运行两套二进制文件，自动生成分析报告并进行 Diff 对比。
```bash
cd ..
./test/result_cmp.sh build
```
执行完毕后，在 `/tmp/paddle_cpp_api_test/` 目录下将生成 **最新增量时间戳** 的测试分析报告。由于历史报告不会被覆盖，**请务必查看最新版本的文件**！

### Step 3: 结果排查与 Diff 登记

请在最新的分析文件中搜索以下关键字：

1. **FAILED / SKIPPED**：存在测试直接崩溃或失败，需回到代码修复缺陷。
2. **DIFF**：发现 Torch 与 Paddle 结果不一致（数值不匹配、报错异常等）。

**遇到 Diff 时的标准处理动作**：
1. 在 `.cpp` 测试文件中找到产生 Diff 的具体 Test Case。
2. 在代码中标记 Diff（例如：`// [DIFF] Paddle returns NaN, Torch returns 0`），不要释掉输出diff的测试用例，也不要规避，因为它们是后续分析的关键证据。
3. 分析 Diff 代表的底层差异逻辑。
4. **整理差异并追加写入到** `PaddleCppAPITest/doc/mismatch_api_record.md`，务必严格对齐该文件现有的 Markdown 格式。测试代码**不需要**为了迎合 Torch 强行修改预期。

## Success Metrics

在完成测试开发后，请务必逐项检查以下内容：

- [ ] **构建**：是否按增量/全量正确完成编译。
- [ ] **输出流**：首个用例确保调用 `createFile()`，其余均使用 `openAppend()`。
- [ ] **基础覆盖**：形状涵盖（标量、空Tensor、大/小Tensor），Dtype 涵盖（float32/64、int32/64）。
- [ ] **结果分析**：是否运行了 `result_cmp.sh` 并检查了最新日志；是否将产生的 diff 按规范录入 `mismatch_api_record.md`。
