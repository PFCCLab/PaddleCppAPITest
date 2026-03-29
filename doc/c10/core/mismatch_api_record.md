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

---

## 当前对齐结果

| 测试用例 | Paddle/Torch 当前输出 |
|---------|----------------------|
| `DefaultConstructor` | `1 1 1` |
| `MutableGet` | `1 9.000000` |
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

## 差异点列表

### 1. 量化类型 `elementSize` 未实现

`c10::elementSize()` 对量化整型不支持：

| ScalarType | PyTorch 返回值 | Paddle 状态 |
|------------|--------------|------------|
| `QInt8`    | 1            | 未实现，编译报错 |
| `QUInt8`   | 1            | 未实现，编译报错 |
| `QInt32`   | 4            | 未实现，编译报错 |

### 2. Float8 扩展枚举值缺失

Paddle compat 的 `ScalarType` 枚举未定义以下两个值，`isFloat8Type` 实现中也将其注释掉：

- `ScalarType::Float8_e5m2fnuz`
- `ScalarType::Float8_e4m3fnuz`

PyTorch 完整支持这两个 Float8 变体，Paddle compat 仅保留了 `Float8_e5m2` 和 `Float8_e4m3fn`。

### 3. `ComplexHalf` 枚举值缺失

Paddle compat 的 `ScalarType` 枚举未包含 `ComplexHalf`，`isComplexType` 实现中对该分支也已注释掉。PyTorch 完整支持。

### 4. 以下 10 个函数/常量在 Paddle compat 中完全缺失

整块 `#ifndef USE_PADDLE_API` 保护了如下 10 个测试，Paddle 下全部跳过：

| 函数/常量 | 说明 |
|-----------|------|
| `c10::isQIntType()` | 判断量化整型 |
| `c10::isBitsType()` | 判断位类型 |
| `c10::isBarebonesUnsignedType()` | 判断裸无符号整型 |
| `c10::toQIntType()` | 转换为量化整型 |
| `c10::toUnderlying()` | 量化类型的底层类型 |
| `c10::isUnderlying()` | 判断底层类型关系 |
| `c10::toRealValueType()` | 复数类型转实数类型 |
| `c10::toComplexType()` | 实数类型转复数类型 |
| `c10::canCast()` | 类型间是否可转换 |
| `c10::NumScalarTypes` | ScalarType 枚举总数常量 |

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
| NumScalarTypes | 编译报错 | 正常输出 |

---

## 初步问题分析

1. **量化类型 elementSize**：Paddle 未实现 QInt8、QUInt8、QInt32 等量化类型的 elementSize，编译会报错。

2. **Float8 枚举值**：Paddle 缺少 Float8_e5m2fnuz 和 Float8_e4m3fnuz 枚举值。

3. **ComplexHalf 枚举值**：Paddle 缺少 ComplexHalf 枚举值。

4. **10个函数/常量缺失**：isQIntType、isBitsType、isBarebonesUnsignedType、toQIntType、toUnderlying、isUnderlying、toRealValueType、toComplexType、canCast、NumScalarTypes 在 Paddle 中完全缺失，需要补全。

---

## 修复方向

在 Paddle compat 的 `c10/core/ScalarType.h` 中逐一补全上述枚举值和函数实现，完成后将对应测试移出 `#ifndef USE_PADDLE_API` 块。

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
