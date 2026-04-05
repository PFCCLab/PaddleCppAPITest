#### 记录 PaddleCppAPITest 仓库中曾经出现过的接口差异，便于回溯排查过程。当前基线已在 2026-04-03 重新通过 `bash test/result_cmp.sh ./build/` 复核；目前仍有少量已归档的已知 diff。测试文件中仍保留了 `[DIFF]` 注释，便于检索当时的差异背景。

---

## 2026-04-06 XPU 编译修复与测试分类调整

### 本轮修复（Paddle 内部 ctest 验证）

| 测试项 | 修复前 | 修复后 | PyTorch | 状态 |
|--------|--------|--------|---------|------|
| XPU 环境编译 `test/cpp/compat` | 编译错误：CUDA 测试在 XPU 环境被错误编译为 CPU 测试 | 编译成功 | 一致 | ✅ 已对齐 |
| ATen CUDA 测试分类 | 7 个 CUDA 测试错误标记为 cc_test | 正确标记为 nv_test | 一致 | ✅ 已对齐 |

说明：

- 修复 PR #78580 中的 XPU 相关编译问题。
- 根本原因：`test/cpp/compat/CMakeLists.txt` 中将需要 CUDA 的测试（`ATen_all_test`、`ATen_as_strided_test`、`ATen_basic_test`、`ATen_from_blob_test`、`ATen_index_test`、`ATen_transpose_test`、`ATen_viewAs_test`）错误地标记为 `cc_test`（CPU 编译），在 XPU 环境（无 CUDA）编译失败。
- 解决方案：将上述 7 个测试从 `cc_test` 移至 `if(WITH_GPU)` 条件下的 `nv_test`，确保仅在 CUDA 环境下编译。
- 验证结果：`ninja -j16` 编译成功，`ctest -R c10` 16/16 通过，`ctest -R ATen` 48/48 通过。
- 此修复与 PyTorch 语义对齐：PyTorch 中这些测试同样依赖 CUDA 环境。

### 本轮修改文件

- `/home/may/Paddle/test/cpp/compat/CMakeLists.txt` - 将 7 个 CUDA 测试从 cc_test 移至 nv_test
- `/home/may/PaddleCppAPITest/doc/mismatch_api_record.md` - 添加本轮汇总

---

## 2026-04-05 TensorOptions 命名空间修复与 Mac-CPU 编译兼容

### 本轮修复（Paddle 内部 ctest 验证）

| 测试项 | 修复前 | 修复后 | PyTorch | 状态 |
|--------|--------|--------|---------|------|
| Mac-CPU 编译 `torch::TensorOptions` | 编译错误：type hidden by declaration | 编译成功 | 一致 | ✅ 已对齐 |
| `torch::from_blob` 使用 `torch::TensorOptions` | macOS/Clang 上失败 | 正常编译运行 | 一致 | ✅ 已对齐 |

说明：

- 修复 PR #78580 中的 Mac-CPU 编译问题。
- 根本原因：`c10/core/TensorOptions.h` 通过 `namespace torch { using namespace c10; }` 导出 `c10::TensorOptions` 到 `torch` 命名空间，与 `ATen/core/TensorBody.h` 中的类型别名产生冲突。
- 解决方案：移除 `TensorOptions.h` 中的 `torch` 命名空间导出，使 `torch::TensorOptions` 通过 `torch/types.h -> at::TensorOptions` 解析。
- 验证结果：`ninja -j16` 编译成功，`ctest -R c10` 16/16 通过，`ctest -R ATen` 48/48 通过。
- 详细记录见：`/home/may/PaddleCppAPITest/doc/c10/core/mismatch_api_record.md`

### 本轮修改文件

- `/home/may/Paddle/paddle/phi/api/include/compat/c10/core/TensorOptions.h` - 修复命名空间导出
- `/home/may/Paddle/paddle/fluid/pybind/torch_compat.h` - 将 `DispatchKey::CPU` 改为 `c10::DispatchKey::CPU`
- `/home/may/PaddleCppAPITest/doc/c10/core/mismatch_api_record.md` - 添加详细记录
- `/home/may/PaddleCppAPITest/doc/mismatch_api_record.md` - 添加本轮汇总

---

## 2026-04-04 Quantized Types 与 Float4_e2m1fn_x2 语义对齐（Paddle 内部 ctest）

### 本轮修复（已解决）

| 测试项 | 修复前 Paddle | 修复后 Paddle | PyTorch | 状态 |
|--------|---------------|---------------|---------|------|
| `qint8` / `qint32` / `quint8` / `quint4x2` / `quint2x4` | 缺少 `using underlying = ...` 别名，无 `alignas(...)` | 已添加 `using underlying` 和对应 `alignas` | 一致 | ✅ 已对齐 |
| `Float4_e2m1fn_x2` | 字段名为 `x`，无比较运算符 | 字段名改为 `val_`，添加 `operator==/!=` | 一致 | ✅ 已对齐 |

说明：

- 本轮修复针对 reviewer 指出的两类问题：
  1. quantized wrapper 类型缺少 `underlying` 类型别名，下游使用 `AT_DISPATCH_CASE_QINT` 等宏时会依赖 `typename scalar_t::underlying`。
  2. `Float4_e2m1fn_x2` 字段名与 PyTorch 不一致（`x` vs `val_`），且缺少比较运算符，导致依赖这些公开成员的代码不兼容。
- 对齐后通过 Paddle 内部 `ctest -R c10 --output-on-failure` 与 `ctest -R ATen --output-on-failure` 验证，全部测试通过。
- 参考 PyTorch 实现：`torch/headeronly/util/qint8.h`、`qint32.h`、`quint8.h`、`quint4x2.h`、`quint2x4.h`、`Float4_e2m1fn_x2.h`。

### 本轮修改文件

- `/home/may/Paddle/paddle/phi/api/include/compat/c10/util/qint8.h` - 添加 `using underlying = int8_t` 和 `alignas(1)`
- `/home/may/Paddle/paddle/phi/api/include/compat/c10/util/qint32.h` - 添加 `using underlying = int32_t` 和 `alignas(4)`
- `/home/may/Paddle/paddle/phi/api/include/compat/c10/util/quint8.h` - 添加 `using underlying = uint8_t` 和 `alignas(1)`
- `/home/may/Paddle/paddle/phi/api/include/compat/c10/util/quint4x2.h` - 添加 `using underlying = uint8_t` 和 `alignas(1)`
- `/home/may/Paddle/paddle/phi/api/include/compat/c10/util/quint2x4.h` - 添加 `using underlying = uint8_t` 和 `alignas(1)`
- `/home/may/Paddle/paddle/phi/api/include/compat/c10/util/Float4_e2m1fn_x2.h` - 字段名 `x` 改为 `val_`，添加 `operator==/!=`
- `/home/may/PaddleCppAPITest/doc/mismatch_api_record.md` - 增补本轮汇总

---

## 2026-04-03 result_cmp 基线复核（PaddleCppAPITest）

### 本轮复核

| 测试项 | 当前 Paddle | PyTorch | 结论 |
|--------|-------------|---------|------|
| `FlattenTest.UnflattenSymint` | `3 24 2 3 4` | `3 24 2 3 4` | ✅ 已对齐 |
| `ArrayRefTest.FromInitializerList` | 已改为在同一完整表达式内消费 `initializer_list` 支撑的 `ArrayRef`，当前输出 `3 5 10 15` | `3 5 10 15` | ✅ 已对齐 |
| `AbsTest.NonContiguousTensor` | 输出值顺序与 Torch 不同 | 非连续张量处理策略不同 | ⚠️ 已知差异（设计/规范不同） |
| `EqualTest.ExceptionTest` | 测试侧已规范化为稳定异常前缀，当前 `result_cmp` 一致 | 一致 | ✅ 已对齐（仍缺完整 C++ stack trace） |
| `OptionalArrayRefTest` | 指针地址、悬空引用随机值不同 | 同上 | ⚠️ 已知差异（运行时/UB） |
| `StreamTest.CudaQuerySynchronizeAndNativeHandle` | `native_handle` 地址值不同 | `0` | ⚠️ 运行时环境差异 |

说明：

- 本轮通过修改 `test/ATen/ops/FlattenTest.cpp`，将 `c10::SymIntArrayRef sizes({3, 4})` 改为先用 `std::vector<c10::SymInt>` 存储再构造 `SymIntArrayRef`，规避了 GCC 13 `-O3` 下 `ArrayRef<int64_t>` 列表初始化的临时对象生命周期问题。
- 本轮补充将 `test/c10/util/ArrayRefTest.cpp` 中 `FromInitializerList` / `VectorArrayRefComparison` 的悬空 `ArrayRef` 写法改为稳定形式，`ArrayRefTest` 已不再出现在 `DIFFER` 列表中。
- 本轮之后，`result_cmp.sh` 中未解决的 DIFFER 剩余 **3 项**，均属于历史已知的非兼容性差异或环境差异。

### 本轮修改文件

- `/home/may/PaddleCppAPITest/test/ATen/ops/FlattenTest.cpp`
- `/home/may/PaddleCppAPITest/test/c10/util/ArrayRefTest.cpp`
- `/home/may/PaddleCppAPITest/doc/ATen/ops/mismatch_api_record.md`
- `/home/may/PaddleCppAPITest/doc/mismatch_api_record.md`

---

## 2026-04-02 CUDAStream review blocker 收敛（Paddle 内部 ctest）

### 本轮复核

| 测试项 | 当前 Paddle | PyTorch | 结论 |
|--------|-------------|---------|------|
| `getStreamFromPool(true)` 默认参数与重载分派 | bool 重载已恢复 `device_index = -1` 默认参数，不再误绑到 `int priority` 重载并返回低优先级 stream | 同签名、同语义 | ✅ 已对齐 |
| `CUDAStream::raw_stream()` | ~~当前 compat surface 继续保留~~ 已删除（PyTorch 无此接口） | 上游无该旧入口 | ✅ 已删除，与 PyTorch 对齐 |

说明：

- 这轮修复的是 reviewer 指出的两个 blocker，均位于 `/home/may/Paddle/paddle/phi/api/include/compat/c10/cuda/CUDAStream.h`。
- Paddle 内部新增 `/home/may/Paddle/test/cpp/compat/c10_Stream_test.cc` 回归，并在 `/home/may/Paddle/build` 下通过了 `ninja -j16`、`ctest -R c10 --output-on-failure`、`ctest -R ATen --output-on-failure`。
- 详细记录见 [doc/c10/cuda/cuda_stream.md](/home/may/PaddleCppAPITest/doc/c10/cuda/cuda_stream.md) 与 [doc/c10/cuda/mismatch_api_record.md](/home/may/PaddleCppAPITest/doc/c10/cuda/mismatch_api_record.md)。

### 本轮修改文件

- `/home/may/Paddle/paddle/phi/api/include/compat/c10/cuda/CUDAStream.h`
- `/home/may/Paddle/test/cpp/compat/c10_Stream_test.cc`
- `/home/may/PaddleCppAPITest/doc/c10/cuda/cuda_stream.md`
- `/home/may/PaddleCppAPITest/doc/c10/cuda/mismatch_api_record.md`
- `/home/may/PaddleCppAPITest/doc/mismatch_api_record.md`

---

## 2026-04-02 Event 语义补齐（Paddle 内部 ctest）

### 本轮复核

| 测试项 | 当前 Paddle | PyTorch | 结论 |
|--------|-------------|---------|------|
| `c10::Event` lazy-create / timing / raw-stream compatibility | 首次 `record()` 才按目标 device 创建 event，`elapsedTime()` 按 `EventFlag` 真正返回计时结果；同时临时保留 `record(const cudaStream_t&)` 兼容旧下游 | 上游 `Event` 语义一致；无 raw-stream 旧接口 | ✅ 语义已对齐，兼容扩展暂保留 |

说明：

- 这轮主要解决 reviewer 指出的两类差异：构造阶段错误绑定 device，以及 `elapsedTime()` 固定返回 `0.0`。
- 验证来自 Paddle 内部 `ctest -R c10` / `ctest -R ATen`；新增覆盖见 `/home/may/Paddle/test/cpp/compat/c10_Event_test.cc` 与 `/home/may/Paddle/test/cpp/compat/ATen_record_stream_test.cc`。
- `PaddleCppAPITest` 现有 `EventCompatTest` 仍主要覆盖基础构造/属性/CPU 异常行为；详细说明见 [doc/c10/core/event.md](/home/may/PaddleCppAPITest/doc/c10/core/event.md) 与 [doc/c10/core/mismatch_api_record.md](/home/may/PaddleCppAPITest/doc/c10/core/mismatch_api_record.md)。

### 本轮修改文件

- `/home/may/Paddle/paddle/phi/api/include/compat/c10/core/Event.h`
- `/home/may/Paddle/paddle/phi/api/include/compat/c10/cuda/CUDAStream.h`
- `/home/may/Paddle/paddle/phi/api/include/compat/ATen/ops/record_stream.h`
- `/home/may/Paddle/test/cpp/compat/c10_Event_test.cc`
- `/home/may/Paddle/test/cpp/compat/ATen_record_stream_test.cc`
- `/home/may/PaddleCppAPITest/doc/c10/core/event.md`
- `/home/may/PaddleCppAPITest/doc/c10/core/mismatch_api_record.md`
- `/home/may/PaddleCppAPITest/doc/mismatch_api_record.md`

---

## 2026-03-30 Event 回归纳入

### 本轮复核（已确认纳入回归）

| 测试项 | 当前 Paddle | PyTorch | 结论 |
|--------|-------------|---------|------|
| `EventCompatTest.EventDefault` / `EventWithFlag` / `EventRecordThrows` / `EventRecordOnceThrows` / `EventMove` / `EventDevice` | 已进入常规 `result_cmp`；`c10::Event` 构造、`EventFlag`、移动语义、属性读取及 CPU 路径异常行为均已对齐 | 一致 | ✅ 已纳入回归 |

说明：

- 原 `test/c10/core/unmatch_EventTest.cpp` 中记录的历史差异（条件编译包裹、构造函数缺少 `EventFlag`、非 CUDA 构建下 `c10::Event` 不可用）当前 compat 实现已与 PyTorch 对齐。
- `c10::EventPool` 为 Paddle 私有扩展，不纳入跨库对齐测试；原 `unmatch_EventTest.cpp` 保留为历史归档。
- 详细记录见 [doc/c10/core/mismatch_api_record.md](/home/may/PaddleCppAPITest/doc/c10/core/mismatch_api_record.md)。

### 本轮修改文件

- `/home/may/Paddle/paddle/phi/api/include/compat/c10/core/Event.h` - 补齐 `c10::Event` 跨平台兼容实现
- `/home/may/PaddleCppAPITest/test/c10/core/EventCompatTest.cpp` - 新增 `c10::Event` 回归测试
- `/home/may/PaddleCppAPITest/doc/c10/core/mismatch_api_record.md` - 更新 Event 回归状态
- `/home/may/PaddleCppAPITest/doc/mismatch_api_record.md` - 增补本轮汇总

---

## 2026-03-30 Allocator 回归纳入

### 本轮复核（已确认纳入回归）

| 测试项 | 当前 Paddle | PyTorch | 结论 |
|--------|-------------|---------|------|
| `AllocatorCompatTest` 新增 `DefaultConstructor` / `Clear` / `CopySemanticsDeleted` / `NoSingleArgConstructor` / `ConstructorWithDataAndDevice` | 已进入常规 `result_cmp`；历史差异点已对齐并纳入回归 | 一致 | ✅ 已纳入回归 |

说明：

- 原 `test/c10/core/unmatch_AllocatorTest.cpp` 中记录的历史差异（单参数构造默认值、拷贝语义、默认/clear 后 `get_deleter()`、`device()` 类型、`allocation()` 方法）当前 compat 实现已与 PyTorch 对齐。
- 上述差异点已通过在 `AllocatorCompatTest.cpp` 中补充测试用例的方式纳入常规回归；原 `unmatch_AllocatorTest.cpp` 保留为历史归档，不再参与 `result_cmp`。
- 详细记录见 [doc/c10/core/mismatch_api_record.md](/home/may/PaddleCppAPITest/doc/c10/core/mismatch_api_record.md)。

### 本轮修改文件

- `/home/may/PaddleCppAPITest/test/c10/core/AllocatorCompatTest.cpp` - 补充 `get_deleter()`、`device().str()`、拷贝语义、单参数构造等回归测试
- `/home/may/PaddleCppAPITest/doc/c10/core/mismatch_api_record.md` - 更新 Allocator 回归状态
- `/home/may/PaddleCppAPITest/doc/mismatch_api_record.md` - 增补本轮汇总

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
- 当前 compat `c10::ScalarType` 已重新暴露 `ComplexHalf` / `Float4_e2m1fn_x2`；但 `CUDADataType` 的转换分支仍未覆盖这两项，因此本轮结论仍以当前 conversion switch 已支持的子集为准。

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
| 语义差异（设计/规范不同） | `AbsTest`、`HalfBFloat16Test`、`TensorFactoryTest`、`DefaultDtypeTest`、`ScalarTypeTest`、`TensorOptionsTest` | 非连续张量处理策略、默认值、枚举值或推断规则不同，且可稳定复现 |
| 环境差异（运行时条件相关） | `EmptyOpsTest`、`StreamTest` | CUDA 可用性、构建形态、运行时句柄值影响输出分支 |
| 实现缺口/兼容层行为差异 | `EqualTest`、`OptionalArrayRefTest` | 异常 stack trace 基础设施缺口、typed ptr 能力缺口或悬空引用行为差异 |

## 关键差异摘要（节选，2026-04-03 基线）

| 测试 | Torch（节选） | Paddle（节选） | 性质 |
|---|---|---|---|
| `AbsTest` | `3.000000 2.000000 1.000000 ...` | `3.000000 0.000000 2.000000 ...` | 非连续张量处理策略不同（设计差异） |
| `HalfBFloat16Test` | `... 5 15`（已注释不比对） | `... 5 11`（已注释不比对） | ScalarType 枚举值差异 |
| `OptionalArrayRefTest` | `1 4 <addr> ...` / `0.000000` | `1 4 <different_addr> ...` / `-0.000000` | 地址差异 + 部分 UB 场景随机值 |
| `StreamTest` | `... sync_ok 1 0 0` | `... sync_ok 1 0 <handle_addr>` | `native_handle` 运行时环境差异 |
| `ScalarTypeTest` | `... QInt8 QUInt8 ...` | 编译/链接缺 `isQIntType` 等 helper | 仍有 9 个 helper 未接入 compat |

## 建议跟踪优先级

1. **P0（稳定语义差异）**：`AbsTest`（非连续策略）、`HalfBFloat16Test`、`TensorFactoryTest`、`DefaultDtypeTest`。
2. **P1（接口补齐）**：`ScalarTypeTest`（剩余 9 个 helper）、`TensorOptionsTest`（`requires_grad` 创建路径）。
3. **P2（环境与实现缺口）**：`EmptyOpsTest`、`StreamTest`、`EqualTest`、`OptionalArrayRefTest`。

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
