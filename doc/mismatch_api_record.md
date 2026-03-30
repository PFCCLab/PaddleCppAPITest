#### 记录 PaddleCppAPITest 仓库中曾经出现过的接口差异，便于回溯排查过程。当前基线已在 2026-03-23 通过 `bash test/result_cmp.sh ./build/` 对齐；以下内容主要作为历史归档，不代表现状仍然存在 diff。测试文件中仍保留了 `[DIFF]` 注释，便于检索当时的差异背景。

---

## 2026-03-30 ATen Indexing / DeviceGuard 复核

### 本轮复核（已确认对齐）

| 测试项 | 当前 Paddle | PyTorch | 结论 |
|--------|-------------|---------|------|
| `IndexingTest.TensorIndexing` / `SliceIndexing` | `index({Slice(...)})` 路径输出与 Torch 一致 | 一致 | ✅ 已对齐 |
| `DeviceGuardTest.DeviceOfTensor` / `DeviceOfOptionalTensor` | CPU 设备输出为 `cpu -1` | `cpu -1` | ✅ 已对齐 |

说明：

- `doc/ATen/mismatch_api_record.md` 中关于 `ATen/indexing.h`、`std::vector<Slice>` 专用入口以及 `device_of` 返回 `cpu:0` 的旧结论已回写为历史信息，不再代表当前基线。
- 本轮复核使用的直接验证文件为 `test/ATen/IndexingTest.cpp` 与 `test/ATen/DeviceGuardTest.cpp`。

---

## 2026-03-30 CUDAContext 回归纳入

### 本轮复核（已确认对齐）

| 测试项 | 当前 Paddle | PyTorch | 结论 |
|--------|-------------|---------|------|
| `CUDAContextTest.GetDeviceProperties` / `GetCurrentDeviceProperties` / `GetCurrentCUDAStream` | 已进入常规 `result_cmp`；无 CUDA 运行时时统一输出 `cuda_runtime_unavailable`，有 CUDA 时比较稳定设备/stream 摘要 | 一致 | ✅ 已纳入回归 |

说明：

- 原 `test/ATen/cuda/unmatch_CUDAContextTest.cpp` 已迁移为 `test/ATen/cuda/CUDAContextTest.cpp`。
- 本轮同时将文档中的旧结论“未进入常规回归”回写为历史信息，不再代表当前基线。

### 本轮修改文件

- `/home/may/PaddleCppAPITest/test/ATen/cuda/CUDAContextTest.cpp` - 新增常规回归测试，替代原 `unmatch` 路径
- `/home/may/PaddleCppAPITest/doc/ATen/cuda/mismatch_api_record.md` - 将 `CUDAContext` 节更新为“历史差异 + 当前已纳入回归”
- `/home/may/PaddleCppAPITest/doc/mismatch_api_record.md` - 增补本轮汇总

---

## 2026-03-30 CUDADataType 复核

### 本轮复核（已确认对齐）

| 测试项 | 当前 Paddle | PyTorch | 结论 |
|--------|-------------|---------|------|
| `CUDADataTypeTest.GetCudaDataType` / `GetCudaDataTypeBFloat16` / `GetCudaDataTypeComplex` | 常规回归输出一致；`Bool` 分支当前统一记录为 `bool_unsupported` | 一致 | ✅ 已对齐 |
| `CUDADataTypeTest.EmptyCUDA` / `EmptyCudaDifferentDtype` | 同一运行环境下进入相同分支；当前环境均为 `cuda_not_available` | 一致 | ✅ 已对齐 |

说明：

- 旧文档把 `ScalarTypeToCudaDataType(Bool)` 记成了 Paddle 单边差异，但当前 Torch 实现同样不支持 `Bool -> cudaDataType`；本轮已将测试改成显式记录共享异常分支。
- `empty_cuda` 系列的可观察输出仍受 CUDA 运行时影响，但在 `result_cmp` 的同机执行前提下，两端当前输出一致。
- 当前 compat `c10::ScalarType` 侧尚未暴露 `ComplexHalf` / `Float4_e2m1fn_x2`，因此本轮结论仅覆盖现有已暴露类型。

### 本轮修改文件

- `/home/may/PaddleCppAPITest/test/ATen/cuda/CUDADataTypeTest.cpp` - 显式记录 `Bool` 的共享异常分支，并修正文档性注释
- `/home/may/PaddleCppAPITest/doc/ATen/cuda/mismatch_api_record.md` - 将 `CUDADataType` 节更新为“历史差异 + 当前已对齐”
- `/home/may/PaddleCppAPITest/doc/mismatch_api_record.md` - 增补本轮汇总，并从待跟踪项中移除 `CUDADataTypeTest`

---

## 2026-03-30 Utils 回归纳入

### 本轮复核（已确认对齐）

| 测试项 | 当前 Paddle | PyTorch | 结论 |
|--------|-------------|---------|------|
| `UtilsTest.TensorCPU` / `TensorBackend` / `TensorComplexCPU` / `TensorComplexBackend` | 已进入常规 `result_cmp`；公开 `at::tensor(ArrayRef<T>, TensorOptions)` 与复数重载输出一致，backend 分支当前统一回写 `cuda_runtime_unavailable` | 一致 | ✅ 已纳入回归 |

说明：

- 原 `test/ATen/unmatch_UtilsTest.cpp` 已迁移为 `test/ATen/UtilsTest.cpp`。
- 本轮不再把 `at::detail::tensor_*` 内部 helper 作为外部比较目标，而是改为通过公开 `at::tensor` 入口覆盖 `ATen/Utils.cpp` 的实现路径。

### 本轮修改文件

- `/home/may/PaddleCppAPITest/test/ATen/UtilsTest.cpp` - 新增常规回归测试，覆盖 `ATen/Utils` 对应公开构造入口
- `/home/may/PaddleCppAPITest/doc/ATen/mismatch_api_record.md` - 新增 `Utils` 历史差异与当前回归状态说明
- `/home/may/PaddleCppAPITest/doc/mismatch_api_record.md` - 增补本轮汇总

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

## 2026-03-29 Device 接口补齐

### 本轮修复（已解决）

| 测试项 | 修复前 Paddle | 修复后 Paddle | PyTorch | 状态 |
|--------|---------------|---------------|---------|------|
| DeviceTest.DeviceStr | 历史观测：`cpu:0 cpu:0 gpu:0 gpu:1` | `cpu cpu:0 cuda:0 cuda:1` | `cpu cpu:0 cuda:0 cuda:1` | ✅ 已对齐 |
| DeviceTest.HasIndex | 历史观测：`1 1 1 1` | `0 1 0 1` | `0 1 0 1` | ✅ 已对齐 |
| DeviceTest.StrictStringParsing / PredicatesAndHash / SetIndexAndTensorDevice | 历史上未覆盖 | 已新增并对齐 | 已对齐 | ✅ 已对齐 |
| TensorOptionsTest.DeviceIndex | 历史观测：`0` | `-1` | `-1` | ✅ 已对齐 |

### 本轮修改文件

- `/home/may/Paddle/paddle/phi/api/include/compat/c10/core/DeviceType.h` - 新增 `PrivateUse1`/`kPrivateUse1` 别名与 `hash<DeviceType>`
- `/home/may/Paddle/paddle/phi/api/include/compat/c10/core/Device.h` - 补齐 `operator!=`、`set_index`、设备谓词、`supports_as_strided` 与 `hash<Device>`
- `/home/may/Paddle/paddle/phi/api/include/compat/c10/core/Device.cpp` - 对齐严格字符串解析与 `privateuseone` 支持；字符串解析状态枚举为规避 Windows `ERROR` 宏污染使用了 Windows-safe 命名
- `/home/may/Paddle/test/cpp/compat/c10_Device_test.cc` - 补充 compat 单测覆盖新增接口
- `/home/may/PaddleCppAPITest/test/c10/core/DeviceTest.cpp` - 增加 `Device` 行为/接口对比测试
- `/home/may/PaddleCppAPITest/doc/c10/core/mismatch_api_record.md` - 更新 `Device` 历史差异说明

---

## 2026-03-29 Exception 宏对齐

### 本轮修复（已解决）

| 测试项 | 修复前 Paddle | 修复后 Paddle | PyTorch | 状态 |
|--------|---------------|---------------|---------|------|
| ExceptionTest.TorchCheckEqFailure | 历史实现报错前缀为 `Expected 3 == 4 ... but got false`，测试通过 `#if USE_PADDLE_API` 单独走 try-catch | 统一输出 `1`，表示异常文本包含 `Check failed: 3 == 4 (3 vs. 4). ` | `1` | ✅ 已对齐 |
| ExceptionTest.TorchCheckNe | 历史实现报错前缀与 Torch 不同，测试按平台分叉 | 统一输出 `1 1`，表示成功路径与失败消息校验都对齐 | `1 1` | ✅ 已对齐 |

### 本轮说明

- 本轮同时修正文档中的历史误记：当前 PyTorch `TORCH_CHECK_OP` 派生宏失败路径是抛异常，不是旧文档中记录的 `EXPECT_DEATH` / `abort()`。
- `TORCH_CHECK_EQ / NE / LE / LT / GE / GT` 共享 `TORCH_CHECK_OP`，因此本轮对齐的是比较类异常宏的公共报错前缀。

### 本轮修改文件

- `/home/may/Paddle/paddle/phi/api/include/compat/c10/util/Exception.h` - 对齐 `TORCH_CHECK_OP` 派生宏报错前缀
- `/home/may/PaddleCppAPITest/test/c10/util/ExceptionTest.cpp` - 移除平台分叉，统一校验异常消息
- `/home/may/PaddleCppAPITest/doc/c10/util/mismatch_api_record.md` - 将 `Exception` 节更新为“历史差异 + 当前已对齐”
- `/home/may/PaddleCppAPITest/doc/mismatch_api_record.md` - 增补本轮汇总

---

## 按差异类型分组（便于 Review）

| 分类 | 测试 | 主要表现 |
|---|---|---|
| 语义差异（设计/规范不同） | `AccumulateTypeTest`、`DefaultDtypeTest`、`DeviceGuardTest`、`DeviceTest`、`HalfBFloat16Test`、`ScalarTypeTest`、`SparseTensorTest`、`TensorFactoryTest`、`TensorOptionsTest`、`TensorTest` | 默认值、枚举值、字符串规范或推断规则不同，且可稳定复现 |
| 环境差异（运行时条件相关） | `EmptyOpsTest` | CUDA 可用性与构建形态影响输出分支（如 `cuda_tensor` vs `cuda_not_available`） |
| 实现缺口/兼容层行为差异 | `EqualTest`、`SelectTest`、`TensorPtrTest`、`OptionalArrayRefTest` | 异常路径、崩溃风险、typed ptr 能力缺口或悬空引用行为差异 |

## 关键差异摘要（节选）

| 测试 | Torch（节选） | Paddle（节选） |
|---|---|---|
| AccumulateTypeTest | `... Bool->11 ...` | `... Bool->10 ...` |
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
3. **P2（环境与实现缺口）**：`EmptyOpsTest`、`EqualTest`、`SelectTest`、`TensorPtrTest`、`OptionalArrayRefTest`。

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
