#### 记录 PaddleCppAPITest 仓库中曾经出现过的接口差异，便于回溯排查过程。当前基线已在 2026-03-23 通过 `bash test/result_cmp.sh ./build/` 对齐；以下内容主要作为历史归档，不代表现状仍然存在 diff。测试文件中仍保留了 `[DIFF]` 注释，便于检索当时的差异背景。

---

## 2026-03-28 兼容性对齐更新

### 本轮修复（已解决）

| 测试项 | 修复前 Paddle | 修复后 Paddle | PyTorch | 状态 |
|--------|---------------|---------------|---------|------|
| EqualTest.ExceptionTest | `Null pointer error, the impl_ of Tensor should not be Null` | `Expected a proper Tensor but got None` | `Expected a proper Tensor but got None` | ✅ 已对齐 |
| EqualTest.NotEqualDtype | 历史观测：异常 | `1` | `1` | ✅ 已对齐 |
| SelectTest.SelectNegativeDim | 历史观测：崩溃 (SIGABRT) | `1 3 0.000000 1.000000 2.000000` | `1 3 0.000000 1.000000 2.000000` | ✅ 已对齐 |
| SparseTensorTest.SparseCOOInferSize | 历史观测：`0 2 2` | `2 3 3` | `2 3 3` | ✅ 已对齐 |
| StreamTest.OstreamOperator | `Stream(device_type=0...)` | `stream 17 on device cpu:0` | `stream 17 on device cpu:0` | ✅ 已对齐 |
| StreamTest.NativeHandleCPU | 抛出 `PD_CHECK` 错误 | 抛出包含 `not supported` 的异常 | 抛出包含 `not supported` 的异常 | ✅ 已对齐 |

### 本轮未解决（已知差距）

| 测试项 | 差距描述 | 原因分析 |
|--------|----------|----------|
| SelectTest.SelectException | 缺少 C++ stack trace | Paddle 兼容层暂未实现异常堆栈跟踪机制 |
| StdTest.StdException | 缺少 C++ stack trace | 同上，需要更复杂的异常处理基础设施 |
| StreamTest.CudaQuerySynchronizeAndNativeHandle | 内存地址值不同 | 运行时环境差异（非兼容性问题） |
| TensorFactoryTest.TensorFromBoolArrayRef | 数值差异 (5 10 vs 5 11) | 需要进一步分析 |

### 本轮修改文件

- `/home/may/Paddle/paddle/phi/api/include/compat/ATen/ops/equal.h` - 添加 undefined tensor 检查
- `/home/may/Paddle/paddle/phi/api/include/compat/ATen/ops/select.h` - 更新错误消息格式
- `/home/may/Paddle/paddle/phi/api/include/compat/ATen/ops/std.h` - 更新错误消息格式
- `/home/may/Paddle/paddle/phi/api/include/compat/c10/core/Stream.h` - 修改 ostream 输出格式
- `/home/may/Paddle/paddle/phi/api/include/compat/c10/core/Stream.cpp` - 修改 native_handle 行为
- `/home/may/Paddle/test/cpp/compat/c10_layout_test.cc` - 补充 `SparseCooTensorInferSize` 的 shape 断言，锁定当前对齐行为
- `/home/may/PaddleCppAPITest/doc/ATen/ops/mismatch_api_record.md` - 将 `SparseTensor`、`Equal`、`Select` 更新为“历史差异 + 当前已对齐”
- `/home/may/PaddleCppAPITest/doc/mismatch_api_record.md` - 更新总览中的已解决项与未解决项

---

## 按差异类型分组（便于 Review）

| 分类 | 测试 | 主要表现 |
|---|---|---|
| 语义差异（设计/规范不同） | `AccumulateTypeTest`、`DefaultDtypeTest`、`DeviceGuardTest`、`DeviceTest`、`HalfBFloat16Test`、`ScalarTypeTest`、`SparseTensorTest`、`TensorFactoryTest`、`TensorOptionsTest`、`TensorTest` | 默认值、枚举值、字符串规范或推断规则不同，且可稳定复现 |
| 环境差异（运行时条件相关） | `CUDADataTypeTest`、`EmptyOpsTest` | CUDA 可用性与构建形态影响输出分支（如 `cuda_empty` vs `cuda_not_available`） |
| 实现缺口/兼容层行为差异 | `EqualTest`、`SelectTest`、`TensorPtrTest`、`OptionalArrayRefTest` | 异常路径、崩溃风险、typed ptr 能力缺口或悬空引用行为差异 |

## 关键差异摘要（节选）

| 测试 | Torch（节选） | Paddle（节选） |
|---|---|---|
| AccumulateTypeTest | `... Bool->11 ...` | `... Bool->10 ...` |
| CUDADataTypeTest | `... cuda_empty cuda_empty_int` | `... cuda_not_available cuda_not_available` |
| DefaultDtypeTest | `... 15 ... 9` | `... 11 ... 8` |
| DeviceGuardTest | `cpu -1 ...` | `cpu 0 ...` |
| DeviceTest | `cpu cpu:0 cuda:0 cuda:1 / 0 1 0 1` | `cpu:0 cpu:0 gpu:0 gpu:1 / 1 1 1 1` |
| EqualTest | `... 0 0 1 ...` | `... 0 exception ...` → `... 0 "Expected a proper Tensor" ...` |
| HalfBFloat16Test | `... 5 15` | `... 5 11` |
| ScalarTypeTest | `... QInt8 QUInt8 ...` | `... UNKNOWN_SCALAR UNKNOWN_SCALAR ...` |
| SelectTest | `... SelectNegativeDim 的真实结果 ...` | `known_crash_on_negative_dim` → 异常消息格式已对齐（但缺少 stack trace） |
| SparseTensorTest | `... InferSize: 2 2 2 ...` | `... InferSize: 0 2 2 ...` |
| TensorFactoryTest | `... bool scalar_type=11 ...` | `... bool scalar_type=10 ...` |
| TensorOptionsTest | `... device_index=-1 ...` | `... device_index=0 ...` |
| TensorPtrTest | `const_ptr[0], mut_ptr[0]` | `typed_ptr_unavailable_on_paddle_compat ...` |
| TensorTest | `... device.type=0, get_device=-1 ...` | `... device.type=1, get_device=0 ...` |

## 建议跟踪优先级

1. **P0（稳定语义差异）**：`Device*`、`TensorOptionsTest`、`DefaultDtypeTest`、`HalfBFloat16Test`、`TensorFactoryTest`、`TensorTest`。
2. **P1（稀疏/存储一致性）**：`SparseTensorTest`、`AccumulateTypeTest`、`ScalarTypeTest`。
3. **P2（环境与实现缺口）**：`CUDADataTypeTest`、`EmptyOpsTest`、`EqualTest`、`SelectTest`、`TensorPtrTest`、`OptionalArrayRefTest`。

---

## 按目录拆分

- `c10/core`: [doc/c10/core/mismatch_api_record.md](/home/may/PaddleCppAPITest/doc/c10/core/mismatch_api_record.md)
- `c10/util`: [doc/c10/util/mismatch_api_record.md](/home/may/PaddleCppAPITest/doc/c10/util/mismatch_api_record.md)
- `c10/cuda`: [doc/c10/cuda/mismatch_api_record.md](/home/may/PaddleCppAPITest/doc/c10/cuda/mismatch_api_record.md)
- `ATen`: [doc/ATen/mismatch_api_record.md](/home/may/PaddleCppAPITest/doc/ATen/mismatch_api_record.md)
- `ATen/core`: [doc/ATen/core/mismatch_api_record.md](/home/may/PaddleCppAPITest/doc/ATen/core/mismatch_api_record.md)
- `ATen/cuda`: [doc/ATen/cuda/mismatch_api_record.md](/home/may/PaddleCppAPITest/doc/ATen/cuda/mismatch_api_record.md)
- `ATen/ops`: [doc/ATen/ops/mismatch_api_record.md](/home/may/PaddleCppAPITest/doc/ATen/ops/mismatch_api_record.md)

说明：
- 根文档保留总览、修复摘要和导航。
- 详细历史差异已按当前 `doc/` 目录结构拆分到各子目录。
