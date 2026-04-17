## 2026-04-17 Event 历史归档测试删除

### 本轮复核

| 测试项 | 当前 Paddle | PyTorch | 结论 |
|--------|-------------|---------|------|
| `EventCompatTest.EventDefault` / `EventWithFlag` / `EventRecordThrows` / `EventRecordOnceThrows` / `EventMove` / `EventDevice` | 重新定向执行 `./build/paddle/paddle_EventCompatTest` 与 `./build/torch/torch_EventCompatTest`，结果文件 `diff` 无差异 | 一致 | ✅ 持续对齐 |

说明：

- `test/c10/core/unmatch_EventTest.cpp` 本身仍只是在记录历史差异，且按 `CMakeLists.txt` 中的 `unmatch_*.cpp` 过滤规则并不会参与常规构建或 `result_cmp`。
- 本轮已再次确认真正承担回归职责的是 `test/c10/core/EventCompatTest.cpp`，因此删除 `test/c10/core/unmatch_EventTest.cpp`，避免“已对齐但仍保留旧 unmatch 文件”造成误导。
- `c10::EventPool` 仍属于 Paddle 私有扩展，无对应 libtorch API，继续不纳入跨库对齐测试。

### 本轮修改文件

- `/home/may/PaddleCppAPITest/test/c10/core/unmatch_EventTest.cpp` - 删除已失效的历史归档测试文件
- `/home/may/PaddleCppAPITest/test/c10/core/EventCompatTest.cpp` - 更新文件头注释，说明旧归档文件已删除
- `/home/may/PaddleCppAPITest/doc/c10/core/mismatch_api_record.md` - 增补本轮复核与删除结论
- `/home/may/PaddleCppAPITest/doc/mismatch_api_record.md` - 同步顶层汇总

---

## 2026-04-05 TensorOptions Mac-CPU 编译修复（Paddle 内部 ctest 已验证）

### 本轮修复

| 测试项 | 修复前 Paddle | 修复后 Paddle | PyTorch | 状态 |
|--------|---------------|---------------|---------|------|
| `torch::TensorOptions` 命名空间解析 | `TensorOptions.h` 通过 `using namespace c10` 将 `c10::TensorOptions` 导出到 `torch` 命名空间，导致 macOS/Clang 上 "type hidden by declaration" 编译错误 | 移除 `namespace torch { using namespace c10; }` 导出；`torch::TensorOptions` 现在通过 `torch/types.h -> at::TensorOptions` 解析，与 PyTorch 一致 | 一致 | ✅ 已对齐 |
| `torch::from_blob` 配合 `torch::TensorOptions` | 在 macOS/Clang 上编译失败 | 使用 `torch::from_blob(..., torch::TensorOptions().dtype(...))` 可正常编译，命名空间解析路径与 PyTorch 一致 | 一致 | ✅ 已对齐 |

说明：

- 修复了 PR #78580 中提到的 Mac-CPU 编译错误：在使用 `torch::from_blob(pp, {3}, torch::TensorOptions().dtype(torch::kInt64))` 时，clang 报告 "a type named 'TensorOptions' is hidden by a declaration in a different namespace"。
- 根本原因是 `c10/core/TensorOptions.h` 第 377-379 行的 `namespace torch { using namespace c10; }` 将整个 `c10` 命名空间导入 `torch`，与 `ATen/core/TensorBody.h` 中的 `using TensorOptions = c10::TensorOptions;` 产生冲突。
- 解决方案：移除 `TensorOptions.h` 中的 `torch` 命名空间导出，使 `torch::TensorOptions` 通过 `torch/types.h` 中的 `using at::TensorOptions` 解析，最终指向 `c10::TensorOptions`，与 PyTorch 的解析路径一致。
- 验证：`ninja -j16` 编译成功，`ctest -R c10` 16/16 通过，`ctest -R ATen` 48/48 通过。
- 相关文件修改：
  - `/home/may/Paddle/paddle/phi/api/include/compat/c10/core/TensorOptions.h` - 移除 `namespace torch { using namespace c10; }`
  - `/home/may/Paddle/paddle/fluid/pybind/torch_compat.h` - 将 `DispatchKey::CPU` 改为 `c10::DispatchKey::CPU`（`torch/library.h` 已间接包含 DispatchKey 头文件）

### 本轮修改文件

- `/home/may/Paddle/paddle/phi/api/include/compat/c10/core/TensorOptions.h` - 移除 `torch` 命名空间对 `c10` 的导出，解决 macOS/Clang 编译错误
- `/home/may/Paddle/paddle/fluid/pybind/torch_compat.h` - 将 `DispatchKey::CPU` 改为 `c10::DispatchKey::CPU`
- `/home/may/PaddleCppAPITest/doc/c10/core/mismatch_api_record.md` - 记录本轮 TensorOptions 命名空间修复

---

## 2026-04-02 Event 语义补齐（Paddle 内部 ctest 已验证）

### 本轮复核

| 测试项 | 当前 Paddle | PyTorch | 结论 |
|--------|-------------|---------|------|
| `c10::Event::record` / `block` / `elapsedTime` / legacy raw-stream compatibility | `Event` 首次 `record()` 时才按目标 stream 的 device lazy-create；`EventFlag::BACKEND_DEFAULT` 启用 timing；`elapsedTime()` 不再返回固定 `0.0`；同时暂时保留 `record(const cudaStream_t&)` 兼容旧下游 | 上游 `Event` 语义一致；无 raw-stream 旧接口 | ✅ 核心语义已对齐，兼容扩展暂保留 |

说明：

- 修复了 review 指出的两类 blocker：构造阶段错误绑定 device，以及 `elapsedTime()`/timing 语义静默失真。
- Paddle 不再依赖旧版 `EventPool` 预创建语义；当前实现与 PyTorch 一样，在第一次 `record()` 时才真正 materialize backend event。
- 本轮新增验证来自 Paddle 内部：
  - `/home/may/Paddle/test/cpp/compat/c10_Event_test.cc`
  - `/home/may/Paddle/test/cpp/compat/ATen_record_stream_test.cc`
  - `/home/may/Paddle/build` 下 `ctest -R c10`
  - `/home/may/Paddle/build` 下 `ctest -R ATen`
- `PaddleCppAPITest/test/c10/core/EventCompatTest.cpp` 目前仍主要覆盖构造、属性和 CPU 异常主路径；timing / lazy-create / raw-stream 兼容路径尚未单独纳入 `result_cmp`。

### 本轮修改文件

- `/home/may/Paddle/paddle/phi/api/include/compat/c10/core/Event.h` - 改为 lazy-create，补齐 device index 与 timing 语义
- `/home/may/Paddle/paddle/phi/api/include/compat/c10/cuda/CUDAStream.h` - ~~恢复 `raw_stream()` 兼容入口~~ 已删除（PyTorch 无此接口）
- `/home/may/Paddle/paddle/phi/api/include/compat/ATen/ops/record_stream.h` - 恢复 `record_stream(cudaStream_t)` 兼容重载
- `/home/may/Paddle/test/cpp/compat/c10_Event_test.cc` - 新增 Event 语义回归
- `/home/may/Paddle/test/cpp/compat/ATen_record_stream_test.cc` - 补充 raw-stream 兼容路径验证
- `/home/may/PaddleCppAPITest/doc/c10/core/event.md` - 更新 Event 头文件级对齐矩阵
- `/home/may/PaddleCppAPITest/doc/c10/core/mismatch_api_record.md` - 记录本轮 Event 语义补齐

---

## 2026-03-30 Event 回归纳入

### 本轮复核（已确认纳入回归）

| 测试项 | 当前 Paddle | PyTorch | 结论 |
|--------|-------------|---------|------|
| `EventCompatTest.EventDefault` / `EventWithFlag` / `EventRecordThrows` / `EventRecordOnceThrows` / `EventMove` / `EventDevice` | 已进入常规 `result_cmp`；`c10::Event` 构造、`EventFlag`、移动语义、属性读取及 CPU 路径异常行为均已对齐 | 一致 | ✅ 已纳入回归 |

说明：

- 原 `test/c10/core/unmatch_EventTest.cpp` 中记录的历史差异（`#include <c10/core/Event.h>` 被 `#ifndef USE_PADDLE_API` 包裹、`c10::Event` 缺少 `EventFlag` 构造函数、非 CUDA 构建下 `c10::Event` 不可用）当前 compat 实现已与 PyTorch 对齐。
- `c10::EventPool` 属于 Paddle 私有扩展，无对应 libtorch API，不纳入跨库对齐测试；对应历史归档文件已在 2026-04-17 删除。
- 上述可对齐差异点已通过新建 `EventCompatTest.cpp` 的方式纳入常规回归；`EventCompatTest` 当前在 `result_cmp` 中已完全 `MATCH`。

### 本轮修改文件

- `/home/may/Paddle/paddle/phi/api/include/compat/c10/core/Event.h` - 将 `c10::Event` 移出 `#ifdef PADDLE_WITH_CUDA`，补齐 `EventFlag` 枚举与完整构造/属性/异常接口
- `/home/may/PaddleCppAPITest/test/c10/core/EventCompatTest.cpp` - 新增 `c10::Event` 跨库对齐回归测试
- `/home/may/PaddleCppAPITest/doc/c10/core/mismatch_api_record.md` - 更新 Event 回归状态
- `/home/may/PaddleCppAPITest/doc/mismatch_api_record.md` - 增补本轮汇总

---

## 2026-03-30 Allocator 回归纳入

### 本轮复核（已确认纳入回归）

| 测试项 | 当前 Paddle | PyTorch | 结论 |
|--------|-------------|---------|------|
| `AllocatorCompatTest.DefaultConstructor` / `Clear` / `CopySemanticsDeleted` / `NoSingleArgConstructor` / `ConstructorWithDataAndDevice` | 已进入常规 `result_cmp`；`get_deleter()`、拷贝删除、单参数构造缺失、`device().str()` 等输出一致 | 一致 | ✅ 已纳入回归 |

说明：

- 原 `test/c10/core/unmatch_AllocatorTest.cpp` 中记录的历史差异（单参数构造默认值、拷贝语义、默认/clear 后 `get_deleter()`、`device()` 类型、`allocation()` 方法）当前 compat 实现已与 PyTorch 对齐。
- 上述差异点已通过在 `AllocatorCompatTest.cpp` 中补充测试用例的方式纳入常规回归；原 `unmatch_AllocatorTest.cpp` 保留为历史归档，不再参与 `result_cmp`。
- `AllocatorCompatTest` 当前在 `result_cmp` 中已完全 `MATCH`。

### 本轮修改文件

- `/home/may/PaddleCppAPITest/test/c10/core/AllocatorCompatTest.cpp` - 补充 `get_deleter()`、`device().str()`、拷贝语义、单参数构造等回归测试
- `/home/may/PaddleCppAPITest/doc/c10/core/mismatch_api_record.md` - 更新 Allocator 回归状态
- `/home/may/PaddleCppAPITest/doc/mismatch_api_record.md` - 增补本轮汇总

---

# Allocator

> Paddle 头文件：`c10/core/Allocator.h`
>
> 2026-03-29 复核：本节记录的 `DataPtr` 历史行为差异已对齐；当前 compat `Allocator.h` 还补齐了 `mutable_get()`、allocator 注册接口、`InefficientStdFunctionContext`、`CaptureId_t/MempoolId_t/MempoolIdHash`。详细 API 对齐矩阵见 [doc/c10/core/allocator.md](/home/may/PaddleCppAPITest/doc/c10/core/allocator.md)。

## 当前状态

当前 `DataPtr` / `Allocator` 核心行为与 PyTorch 已对齐，当前新增并已验证的直接接口包括：

1. `CaptureId_t`、`MempoolId_t`、`MempoolIdHash`
2. `DataPtr::mutable_get()`
3. `InefficientStdFunctionContext` 与 `InefficientStdFunctionContext::makeDataPtr()`
4. `SetAllocator()`、`GetAllocator()`、`AllocatorRegisterer`、`REGISTER_ALLOCATOR`
5. `Allocator::is_simple_data_ptr()` 语义修正为 `get() == get_context()`

---

## 当前测试覆盖

测试文件：`test/c10/core/AllocatorCompatTest.cpp`

### 关键测试项

1. `DefaultConstructor` / `ConstructorWithDataAndDevice` / `ConstructorWithDeleter`
2. `MutableGet`
3. `CaptureAndMempoolTypes`
4. `InefficientStdFunctionContextMakeDataPtr`
5. `IsSimpleDataPtrSemantics`
6. `SetAndGetAllocatorPriority`
7. `RegisterAllocatorMacro`
8. `CopySemanticsDeleted` / `NoSingleArgConstructor`（2026-03-30 新增，覆盖原 `unmatch_AllocatorTest.cpp` 历史差异点）

---

## 当前对齐结果

| 测试用例 | Paddle/Torch 当前输出 |
|---------|----------------------|
| `DefaultConstructor` | `1 1 1 1` |
| `ConstructorWithDataAndDevice` | `1 1 1.000000 2.000000 1` |
| `MutableGet` | `1 9.000000` |
| `Clear` | `1 1 1 1 1 1` |
| `CopySemanticsDeleted` | `1 1` |
| `NoSingleArgConstructor` | `1` |
| `CaptureAndMempoolTypes` | `1 1 1` |
| `InefficientStdFunctionContextMakeDataPtr` | `1 1 1 1` |
| `IsSimpleDataPtrSemantics` | `1 0 0` |
| `SetAndGetAllocatorPriority` | `1 1 1` |
| `RegisterAllocatorMacro` | `1` |

补充说明：

- `bash test/result_cmp.sh ./build/` 中 `paddle_AllocatorCompatTest` 与 `torch_AllocatorCompatTest` 已完全 `MATCH`。
- 当前仓库全量仍存在其他历史 diff，但不再来自 `AllocatorCompatTest`。

---

## 历史背景

本节原先记录过以下差异：单参数构造默认 `device`、拷贝语义、默认/`clear()` 后 `get_deleter()`、`device()` 返回类型，以及 Paddle 独有的 `allocation()` 方法。对应历史归档测试仍保留在 `test/c10/core/unmatch_AllocatorTest.cpp`，仅用于回溯，不代表现状。

---

历史对齐 PR：https://github.com/PFCCLab/PaddleCppAPITest/pull/42/changes#diff

---

# Device

> Paddle 头文件：`c10/core/Device.h`、`c10/core/DeviceType.h`
>
> 2026-03-29 复核：`DeviceTest` 覆盖的 `index/has_index/str/type/tensor.device()` 等主路径差异已对齐；详细头文件级对齐矩阵见 [doc/c10/core/device.md](/home/may/PaddleCppAPITest/doc/c10/core/device.md) 与 [doc/c10/core/device_type.md](/home/may/PaddleCppAPITest/doc/c10/core/device_type.md)。当前剩余缺口主要集中在 `DeviceType` 扩展 backend 枚举、`DeviceTypeName()` 与 privateuse1 backend 注册接口。

## 当前状态

当前 compat 已对齐 `Device.h` 的核心行为与常用接口，文档保留的旧 diff 不再复现；但 `DeviceType.h` 仍有一批 PyTorch 扩展 API 未覆盖。当前状态可概括为：

1. `DeviceType::PrivateUse1` / `kPrivateUse1`
2. `Device::operator!=()`
3. `Device::set_index()`
4. `is_privateuseone()`、`is_xpu()`、`is_ipu()` 等设备谓词
5. `supports_as_strided()`
6. `std::hash<c10::Device>` 与 `std::hash<c10::DeviceType>`
7. 与 PyTorch 一致的严格字符串解析规则
8. 未覆盖的 `DeviceType` 主要缺口：扩展枚举/常量别名、`DeviceTypeName()`、`register_privateuse1_backend()`、`get_privateuse1_backend()`、`is_privateuse1_backend_registered()`

其中 `Device(const std::string&)` 的解析状态机对外行为已与 PyTorch 对齐；仅内部实现为了兼容 Windows 头文件里的 `ERROR` 宏污染，将状态枚举命名从上游的 `START/INDEX_START/INDEX_REST/ERROR` 改成了 `kStart/kIndexStart/kIndexRest/kError`。

详细 API 逐项状态见：

- [doc/c10/core/device.md](/home/may/PaddleCppAPITest/doc/c10/core/device.md)
- [doc/c10/core/device_type.md](/home/may/PaddleCppAPITest/doc/c10/core/device_type.md)

---

## 当前测试覆盖

测试文件：`test/c10/core/DeviceTest.cpp`

以下用例覆盖 `Device.h` 的共享 backend 主路径，不覆盖 `DeviceType.h` 中尚未实现的扩展 backend 枚举与 privateuse1 backend 注册接口。

### 1. `DeviceStr`

校验 `cpu`、`cpu:0`、`cuda:0`、`cuda:1` 的字符串输出。

### 2. `HasIndex`

校验默认 `CPU/CUDA` 设备的 `index = -1` 与 `has_index() = false` 语义。

### 3. `StrictStringParsing`

校验 `privateuseone` 设备解析，以及 `cuda:-1`、`cuda:01`、`cuda:1:2`、`cpu::0` 等非法字符串抛异常。

### 4. `PredicatesAndHash`

校验设备谓词、`supports_as_strided()`、`operator!=()`、`operator==()` 以及 `unordered_map<c10::Device, ...>` 可用性。

### 5. `SetIndexAndTensorDevice`

校验 `set_index()`，以及默认 CPU tensor / 显式 CPU tensor 的 `device()` 语义。

---

## 当前对齐结果

| 测试用例 | Paddle/Torch 当前输出 |
|---------|----------------------|
| `DeviceStr` | `cpu cpu:0 cuda:0 cuda:1` |
| `HasIndex` | `0 1 0 1` |
| `StrictStringParsing` | `1 privateuseone:3 1 1 1 1` |
| `PredicatesAndHash` | `1 1 1 1 1 0 1 0 1 1 7 3` |
| `SetIndexAndTensorDevice` | `0 0 1 cpu:0 1 2 1 cuda:2 0 -1 0 cpu 0 -1 0 cpu` |

字段说明：

- `StrictStringParsing`：第 1 列表示 `privateuseone:3` 解析成功；后 4 列表示非法字符串均正确抛异常。
- `PredicatesAndHash`：依次对应 `is_cpu`、`is_cuda`、`is_xpu`、`is_ipu`、`is_privateuseone`、`is_mps`、`cpu.supports_as_strided`、`ipu.supports_as_strided`、`cpu != cuda`、`cuda == cuda:0`、`unordered_map[cuda:0]`、`unordered_map[cpu]`。
- `SetIndexAndTensorDevice`：前两组分别是 `cpu.set_index(0)`、`cuda.set_index(2)` 的 `(type index has_index str)`；后两组分别是默认 CPU tensor 与显式 CPU tensor 的 `(type index has_index str)`。
- 上表结果说明 `Device.h` 主路径当前已对齐；`DeviceType.h` 的扩展枚举与 helper 缺口需参考详细文档，不体现在这组测试输出中。

---

## 历史背景

本节原先记录过以下差异：默认 index 语义、`cpu/cuda` 字符串表示、`DeviceType::PrivateUse1` 补齐、默认 CPU tensor 的 `device()` 表达方式。它们已经在 compat 中修复，旧内容仅作为回溯背景保留。`DeviceType.h` 中剩余的扩展 backend / helper 缺口属于后续头文件级补齐范围，不属于这批历史 diff 的残留。

历史对齐 PR：https://github.com/PaddlePaddle/Paddle/pull/78066

---

# BFloat16

> Paddle 头文件：`c10\util\BFloat16.h`

## 差异点列表

1.  **BFloat16 ScalarType 枚举值**：PyTorch 为 **11**，Paddle 为 **15**
2.  **ComplexFloat ScalarType 枚举值**：PyTorch 为 **8**，Paddle 为 **9**

---

## Diff 测试用例位置

测试文件：`test/c10/util/HalfBFloat16Test.cpp`

### 测试用例原文

```cpp
// ScalarType 对应关系
// [DIFF] PyTorch输出: 5 11, PaddlePaddle输出: 5 15 (BFloat16枚举值不同)
TEST_F(HalfBFloat16Test, ScalarTypeMapping) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << std::to_string(static_cast<int>(at::kHalf)) << " ";
  // file << std::to_string(static_cast<int>(at::kBFloat16)) << " "; // [DIFF]
  file.saveFile();
}
```

---

## 输出对比

| 字段 | Paddle 输出 | Torch 输出 |
|------|------------|------------|
| kHalf | 5 | 5 |
| kBFloat16 | 15 | 11 |

---

## 初步问题分析

Paddle 与 PyTorch 的 ScalarType 枚举值定义不同：BFloat16 在 PyTorch=11，Paddle=15；ComplexFloat 在 PyTorch=8，Paddle=9。这是两个框架设计上的差异，需要在兼容层进行映射对齐。

---

# DefaultDtype

> Paddle 头文件：`c10\core\DefaultDtype.h`

## 差异点列表

1.  **BFloat16 枚举值**：PyTorch 为 **11**，Paddle 为 **15**
2.  **ComplexFloat（复数类型）枚举值**：PyTorch 为 **8**，Paddle 为 **9**

---

## Diff 测试用例位置

测试文件：`test/c10/core/DefaultDtypeTest.cpp`

### 测试用例原文

```cpp
// 设置默认 dtype 到 BFloat16
TEST_F(DefaultDtypeTest, SetDefaultDtypeBFloat16) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();

  c10::ScalarType before = c10::get_default_dtype();
  file << c10::toString(before) << " ";

  c10::set_default_dtype(c10::ScalarType::BFloat16);
  at::Tensor t = at::zeros({1}, at::TensorOptions().dtype(c10::ScalarType::BFloat16));
  file << c10::toString(t.scalar_type()) << " ";

  c10::set_default_dtype(before);
  file.saveFile();
}
```

---

## 输出对比

| 测试用例 | Paddle 输出 | Torch 输出 |
|---------|------------|------------|
| SetDefaultDtypeBFloat16 | `double bfloat16` | `double bfloat16` |

（注：输出内容相同，但内部枚举值不同）

---

## 初步问题分析

与 BFloat16 接口类似，Paddle 与 PyTorch 的 ScalarType 枚举值定义不同。当设置默认 dtype 为 BFloat16 时，两个框架都能正常工作，但底层的枚举值存在差异。

---


---

# ScalarType 扩展类型函数

> Paddle 头文件：`c10/core/ScalarType.h`

## 当前状态（2026-04-02 更新）

本轮已在 `paddle/phi/api/include/compat/c10/core/ScalarType.h` 修复 reviewer 指到的 source-level 回归，并在 `/home/may/Paddle/build` 下完成以下验证：

- `ninja -j16`
- `ctest -R c10 --output-on-failure`
- `ctest -R ATen --output-on-failure`

### 本轮已修复

| 项目 | 当前状态 |
|------|---------|
| 扩展枚举值 | 已重新暴露 `ComplexHalf`、`Float8_e5m2fnuz`、`Float8_e4m3fnuz`、`Float8_e8m0fnu`、`Float4_e2m1fn_x2` |
| `toString()` | 已恢复 `QInt*`、`Bits*`、`ComplexHalf`、扩展 Float8 / Float4 的字符串化，不再回退为 `UNKNOWN_SCALAR` |
| `elementSize()` | 已恢复 `QInt8` / `QUInt8` / `QInt32`、`Bits*`、`ComplexHalf`、扩展 Float8 / Float4 的尺寸判断 |
| 类型判定 helper | `isFloat8Type()`、`isReducedFloatingType()`、`isComplexType()` 已覆盖上述恢复类型 |
| `c10::NumScalarTypes` | 已恢复，当前值为 `47` |

### 仍待补齐

PaddleCppAPITest 里 `#ifndef USE_PADDLE_API` 保护的这组 `ScalarType` API 还没有全部补完，当前仍缺以下 9 项：

| 函数 | 说明 |
|------|------|
| `c10::isQIntType()` | 判断量化整型 |
| `c10::isBitsType()` | 判断位类型 |
| `c10::isBarebonesUnsignedType()` | 判断裸无符号整型 |
| `c10::toQIntType()` | 转换为量化整型 |
| `c10::toUnderlying()` | 量化类型的底层类型 |
| `c10::isUnderlying()` | 判断底层类型关系 |
| `c10::toRealValueType()` | 复数类型转实数类型 |
| `c10::toComplexType()` | 实数类型转复数类型 |
| `c10::canCast()` | 类型间是否可转换 |

---

## Diff 测试用例位置

测试文件：`test/c10/core/ScalarTypeTest.cpp`

### 测试用例原文

```cpp
// 测试 c10::isQIntType
TEST_F(ScalarTypeTest, IsQIntType) {
  file << std::to_string(c10::isQIntType(c10::ScalarType::QInt8)) << " ";
  file << std::to_string(c10::isQIntType(c10::ScalarType::QUInt8)) << " ";
  file << std::to_string(c10::isQIntType(c10::ScalarType::QInt32)) << " ";
  file << std::to_string(c10::isQIntType(c10::ScalarType::QUInt4x2)) << " ";
  file << std::to_string(c10::isQIntType(c10::ScalarType::QUInt2x4)) << " ";

  file << std::to_string(c10::isQIntType(c10::ScalarType::Int)) << " ";
  file << std::to_string(c10::isQIntType(c10::ScalarType::Float)) << " ";
  file << std::to_string(c10::isQIntType(c10::ScalarType::Byte)) << " ";
  file.saveFile();
}

// 测试 c10::isBitsType
TEST_F(ScalarTypeTest, IsBitsType) {
  file << std::to_string(c10::isBitsType(c10::ScalarType::Bits1x8)) << " ";
  file << std::to_string(c10::isBitsType(c10::ScalarType::Bits2x4)) << " ";
  file << std::to_string(c10::isBitsType(c10::ScalarType::Bits4x2)) << " ";
  file << std::to_string(c10::isBitsType(c10::ScalarType::Bits8)) << " ";
  file << std::to_string(c10::isBitsType(c10::ScalarType::Bits16)) << " ";

  file << std::to_string(c10::isBitsType(c10::ScalarType::Int)) << " ";
  file << std::to_string(c10::isBitsType(c10::ScalarType::Float)) << " ";
  file << std::to_string(c10::isBitsType(c10::ScalarType::Byte)) << " ";
  file.saveFile();
}

// 测试 c10::canCast
TEST_F(ScalarTypeTest, CanCast) {
  file << std::to_string(
              c10::canCast(c10::ScalarType::Int, c10::ScalarType::Long))
       << " ";
  file << std::to_string(
              c10::canCast(c10::ScalarType::Float, c10::ScalarType::Double))
       << " ";

  file << std::to_string(c10::canCast(c10::ScalarType::ComplexFloat,
                                      c10::ScalarType::ComplexDouble))
       << " ";

  file << std::to_string(
              c10::canCast(c10::ScalarType::Bool, c10::ScalarType::Int))
       << " ";

  file << std::to_string(c10::canCast(c10::ScalarType::ComplexFloat,
                                      c10::ScalarType::Float))
       << " ";

  file << std::to_string(
              c10::canCast(c10::ScalarType::Float, c10::ScalarType::Int))
       << " ";

  file << std::to_string(c10::canCast(c10::ScalarType::Double,
                                      c10::ScalarType::Long))
       << " ";

  file << std::to_string(
              c10::canCast(c10::ScalarType::Int, c10::ScalarType::Bool))
       << " ";

  file << std::to_string(
              c10::canCast(c10::ScalarType::Float, c10::ScalarType::Bool))
       << " ";
  file.saveFile();
}

// 测试 NumScalarTypes 常量
TEST_F(ScalarTypeTest, NumScalarTypes) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << std::to_string(c10::NumScalarTypes) << " ";
  file.saveFile();
}
```

---

## 输出对比

| 测试用例 | Paddle 输出 | Torch 输出 |
|---------|------------|------------|
| IsQIntType | 编译报错 | 正常输出 |
| IsBitsType | 编译报错 | 正常输出 |
| CanCast | 编译报错 | 正常输出 |
| NumScalarTypes | 当前 compat 已实现，值为 `47` | 正常输出 |

---

## 初步问题分析

1. **已修复的回归**：扩展 `ScalarType` 枚举、`toString()`、`elementSize()`、`isFloat8Type()`、`isReducedFloatingType()`、`isComplexType()` 以及 `NumScalarTypes` 已恢复到当前 compat 期望状态。

2. **剩余缺口**：`isQIntType`、`isBitsType`、`isBarebonesUnsignedType`、`toQIntType`、`toUnderlying`、`isUnderlying`、`toRealValueType`、`toComplexType`、`canCast` 仍未接入，因此 PaddleCppAPITest 中对应 `ScalarTypeTest` 还不能整体移出 backlog。

---

## 修复方向

继续在 Paddle compat 的 `c10/core/ScalarType.h` 中补齐剩余 9 个 helper，再将对应测试逐步移出 `#ifndef USE_PADDLE_API` 块。

---


---

# TensorOptions（`requires_grad` / `device_index`）

> Paddle 头文件：`c10/core/TensorOptions.h`
>
> 2026-03-29 复核：`device_index()` 已随 `Device` 语义一并对齐；当前 `TensorOptionsTest.DeviceIndex` 的 Paddle/Torch 输出均为 `-1`。本节当前剩余的已知差异只在 `requires_grad` 创建路径。

## 当前状态

1. **`at::empty()` 不支持含 `requires_grad` 的 `TensorOptions`**：Paddle 在通过 `at::empty({...}, opts)` 创建 tensor 时，若 `opts` 含有 `requires_grad(true)` 会抛出异常。PyTorch 完整支持。当前测试已绕过：将含 `requires_grad` 的 `opts` 与用于创建 tensor 的 `opts_for_dtype` 分离，单独测试 `requires_grad()` 的读取，但实际上 Paddle 无法通过 `TensorOptions` 在 tensor 创建时传递梯度需求。
2. **`device_index()` 已对齐**：`c10::TensorOptions().device(c10::Device(c10::kCPU)).device_index()` 当前 Paddle/Torch 均返回 `-1`。

---

## 当前测试用例位置

测试文件：`test/c10/core/TensorOptionsTest.cpp`

### 相关测试用例

```cpp
TEST_F(TensorOptionsTest, DeviceIndex) {
  auto opts = c10::TensorOptions().device(c10::Device(c10::kCPU));

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();

  file << std::to_string(opts.device_index()) << " ";
  file.saveFile();
}

TEST_F(TensorOptionsTest, ChainedSetters) {
  auto opts = c10::TensorOptions()
      .dtype(at::kDouble)
      .requires_grad(true);  // Paddle 不支持通过 TensorOptions 传递 requires_grad

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();

  file << std::to_string(opts.requires_grad().value()) << " ";
  file.saveFile();
}
```

---

## 当前结果

| 测试用例 | Paddle 输出 | Torch 输出 |
|---------|------------|------------|
| DeviceIndex | `-1` | `-1` |
| ChainedSetters | `1` | `1` |

---

## 当前问题分析

1. **requires_grad 传递**：Paddle 不支持通过 TensorOptions 在创建 tensor 时传递 requires_grad 参数，会抛出异常。

---


---

# DefaultDtype（`get_default_complex_dtype`）

> Paddle 头文件：`c10/core/DefaultDtype.h`

## 差异点列表

1. **默认复数类型不一致**：PyTorch 默认 `ComplexFloat`（枚举值 `9`），Paddle 默认 `ComplexDouble`（枚举值 `8`）。

---

## Diff 测试用例位置

测试文件：`test/c10/core/DefaultDtypeTest.cpp`

### 测试用例原文

```cpp
TEST_F(DefaultDtypeTest, GetDefaultComplexDtype) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  auto dtype = c10::get_default_complex_dtype();
  // [DIFF] PyTorch输出: 9, PaddlePaddle输出: 8
  // file << std::to_string(dtype_to_int(dtype)) << " ";
  (void)dtype;
  file.saveFile();
}
```

---

## 输出对比

| 测试用例 | Paddle 输出 | Torch 输出 |
|---------|------------|------------|
| GetDefaultComplexDtype | `8` | `9` |

---

## 初步问题分析

Paddle 兼容层 `default_complex_dtype` 的初始值与 PyTorch 默认策略不一致，导致默认 complex dtype 语义差异。

---

# Device（`has_index`）

> Paddle 头文件：`c10/core/Device.h`
>
> 2026-03-29 复核：`has_index()` 默认 index 语义已与 PyTorch 对齐；测试已恢复实际值比对，本节保留历史差异说明。

## 当前结论

`DeviceTest.HasIndex` 当前 Paddle/Torch 一致输出 `0 1 0 1`，分别对应：

1. `c10::Device(c10::kCPU).has_index() == false`
2. `c10::Device(c10::kCPU, 0).has_index() == true`
3. `c10::Device(c10::kCUDA).has_index() == false`
4. `c10::Device(c10::kCUDA, 1).has_index() == true`

---

## 当前测试代码

```cpp
TEST_F(DeviceCompatTest, HasIndex) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();

  c10::Device cpu_default(c10::kCPU);
  c10::Device cpu_0(c10::kCPU, 0);
  c10::Device cuda_default(c10::kCUDA);
  c10::Device cuda_1(c10::kCUDA, 1);

  file << std::to_string(cpu_default.has_index() ? 1 : 0) << " ";
  file << std::to_string(cpu_0.has_index() ? 1 : 0) << " ";
  file << std::to_string(cuda_default.has_index() ? 1 : 0) << " ";
  file << std::to_string(cuda_1.has_index() ? 1 : 0) << " ";

  file.saveFile();
}
```

---

## 当前输出

| 测试用例 | Paddle 输出 | Torch 输出 |
|---------|------------|------------|
| HasIndex | `0 1 0 1` | `0 1 0 1` |

---

# Device（`str`）

> Paddle 头文件：`c10/core/Device.h`
>
> 2026-03-29 复核：`str()` 已输出 PyTorch 风格的 `cpu/cuda/privateuseone` 设备字符串；本节保留历史差异说明。

## 当前结论

`DeviceTest.DeviceStr` 当前 Paddle/Torch 一致输出 `cpu cpu:0 cuda:0 cuda:1`。

---

## 当前测试代码

```cpp
TEST_F(DeviceCompatTest, DeviceStr) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();

  c10::Device cpu_device(c10::kCPU);
  auto cpu_str = cpu_device.str();

  c10::Device cpu_device_0(c10::kCPU, 0);
  auto cpu_0_str = cpu_device_0.str();

  c10::Device cuda_device_0(c10::kCUDA, 0);
  auto cuda_0_str = cuda_device_0.str();

  c10::Device cuda_device_1(c10::kCUDA, 1);
  auto cuda_1_str = cuda_device_1.str();

  file << cpu_str << " ";
  file << cpu_0_str << " ";
  file << cuda_0_str << " ";
  file << cuda_1_str << " ";

  file.saveFile();
}
```

---

## 当前输出

| 测试用例 | Paddle 输出 | Torch 输出 |
|---------|------------|------------|
| DeviceStr | `cpu cpu:0 cuda:0 cuda:1` | `cpu cpu:0 cuda:0 cuda:1` |

---
