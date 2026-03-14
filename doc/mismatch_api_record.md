##### 记录PaddleCPPAPITest仓库检测出来的接口不一致情况

# Allocator

## 差异点列表

1.  **构造函数参数默认值**
2.  **拷贝语义**
3.  **`get_deleter()` 在默认构造后的返回值**
4.  **`clear()` 后 `get_deleter()` 的行为**
5.  **Device 类型和方法**
6.  **`allocation()` 方法**

---

涉及到的 PR：https://github.com/PFCCLab/PaddleCppAPITest/pull/42/changes#diff

---

# Device

> Paddle 头文件：`c10\core\Device.h`

## 差异点列表

1.  **未指定 Index 时的默认行为**：PyTorch index = -1，has_index() = false；Paddle 强制默认为 0，has_index() = true
2.  **纯字符串解析行为**：PyTorch 保持无索引状态（如 `cpu`、`cuda`）；Paddle 自动补全为 0 号设备（如 `cpu:0`、`gpu:0`）
3.  **GPU/CUDA 字符串表示**：PyTorch 严格输出 `cuda` 或 `cuda:0`；Paddle 底层映射为 GPU，输出 `gpu:0` 或 `gpu:1`
4.  **底层类型枚举值（Enum ID）**：PyTorch CPU=0，CUDA=1；Paddle CPU=1，CUDA/GPU=2
5.  **默认 Tensor 所在设备**：PyTorch 处于无明确索引的 cpu 状态；Paddle 明确挂载在 cpu:0 设备上

---

提交的对齐 PR：https://github.com/PaddlePaddle/Paddle/pull/78066

---

# BFloat16

> Paddle 头文件：`c10\util\BFloat16.h`

## 差异点列表

1.  **BFloat16 ScalarType 枚举值**：PyTorch 为 **11**，Paddle 为 **15**
2.  **ComplexFloat ScalarType 枚举值**：PyTorch 为 **8**，Paddle 为 **9**

---

# DefaultDtype

> Paddle 头文件：`c10\core\DefaultDtype.h`

## 差异点列表

1.  **BFloat16 枚举值**：PyTorch 为 **11**，Paddle 为 **15**
2.  **ComplexFloat（复数类型）枚举值**：PyTorch 为 **8**，Paddle 为 **9**

---

# IValue

> Paddle 头文件：`ATen/core/ivalue.h`

## 差异点列表

1.  **命名空间**：PyTorch 为 `c10::IValue`；Paddle 为 `torch::IValue`（c10 命名空间中不存在 IValue）
2.  **方法命名风格**：PyTorch 使用 camelCase（如 `isNone()`、`toBool()`）；Paddle 使用 snake_case（如 `is_none()`、`to_bool()`）
3.  **`tagKind()` 方法**：PyTorch 存在；Paddle 中**不存在**
4.  **字符串提取方法**：PyTorch 为 `toStringRef()`；Paddle 为 `to_string()`

---

# SparseTensor

> Paddle 头文件：`ATen/ops/sparse_coo_tensor.h`、`ATen/ops/sparse_csr_tensor.h`

## 差异点列表

1.  **sparse_coo_tensor 无 size 推断行为**：PyTorch 能根据 indices 内容正确推断完整 size（如 `2 2 2`）；Paddle 推断结果第一个维度为 0（如 `0 2 2`）

---

# OptionalArrayRef

> Paddle 头文件：`c10\util\OptionalArrayRef.h`

## 差异点列表

1.  **运行时内存地址值**：两框架输出的内存地址不同（属正常运行时差异，不影响功能）
2.  **内部对象标识符**：两框架内部唯一标识符数值不同（属正常实现差异，不影响功能）

> 注：OptionalArrayRef 核心功能（has_value、size、元素访问、reset、swap、emplace、slice 等）在两个框架中完全兼容，仅运行时地址和标识符存在差异。

---

# at::indexing（Slice / EllipsisIndexType）

> Paddle 头文件：`ATen/TensorIndexing.h`

## 差异点列表

- [x] **头文件路径不同**：PyTorch 为 `ATen/TensorIndexing.h`；Paddle compat 为 `ATen/indexing.h`
- [ ] **`Tensor::operator[](Slice)` 不支持**：PyTorch 的 `Tensor::operator[]` 接受 `at::indexing::Slice`；Paddle compat 的 `operator[]` 仅重载 `int64_t`，传入 `Slice` 会编译报错
- [x] **多维 Slice 索引写法不同**：
    - PyTorch：`t.index({Slice(0,2), Slice(1,3)})` —— 接受 `std::initializer_list<TensorIndex>`
    - Paddle：`t.index(std::vector<at::indexing::Slice>{Slice(0,2), Slice(1,3)})` —— 仅重载 `std::vector<Slice>`
- [x] **`TensorIndex` 类不存在**：Paddle compat 的 `indexing.h` 未定义 `TensorIndex` 类，注释掉了 `index(ArrayRef<TensorIndex>)` 重载，仅保留 `index(const std::vector<Slice>&)`


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

## 修复方向

在 Paddle compat 的 `c10/core/ScalarType.h` 中逐一补全上述枚举值和函数实现，完成后将对应测试移出 `#ifndef USE_PADDLE_API` 块。

---

# TensorAccessor / GenericPackedTensorAccessor

> Paddle 头文件：`ATen/core/TensorAccessor.h`

## 差异点列表

### `GenericPackedTensorAccessorBase` / `GenericPackedTensorAccessor` 系列类缺失

Paddle compat 的 `ATen/core/TensorAccessor.h` 中**未实现**以下类和类型别名：

- `at::GenericPackedTensorAccessorBase<T, N, PtrTraits, index_t>`
- `at::GenericPackedTensorAccessor<T, N, PtrTraits, index_t>`
- `at::PackedTensorAccessor32<T, N, PtrTraits>`（`index_t = int32_t` 别名）
- `at::PackedTensorAccessor64<T, N, PtrTraits>`（`index_t = int64_t` 别名）

以及 `at::Tensor` 上的 `packed_accessor64<T,N>()` 方法（Paddle compat 仅有 `packed_accessor32`）。

libtorch 在同路径头文件中完整定义了上述类，供 CUDA kernel 使用。

## 修复方向

在 Paddle compat 的 `ATen/core/TensorAccessor.h` 中补充 `GenericPackedTensorAccessorBase`、`GenericPackedTensorAccessor` 完整实现及 `PackedTensorAccessor32/64` 类型别名；并在 `ATen/core/Tensor.h` 中补充 `packed_accessor64<T,N>()` 方法。

---

# Exception 宏（TORCH_CHECK_EQ / TORCH_CHECK_NE 失败语义差异）

> Paddle 头文件：`c10/util/Exception.h`

## 差异点列表

1. **`TORCH_CHECK_EQ` 失败行为**：PyTorch 调用 `abort()` 终止进程（测试用 `EXPECT_DEATH` 捕获）；Paddle 抛出 C++ 异常（测试用 try-catch 捕获）。
2. **`TORCH_CHECK_NE` 失败行为**：同上，两者失败行为不一致。

当前代码通过 `#if USE_PADDLE_API` 分叉两套检测逻辑以绕过差异，但这导致两个平台实际走不同测试路径，无法真正对比行为。

---

# CUDA Context（`at::cuda::getCurrentCUDAStream` 缺失）

> Paddle 头文件：`ATen/cuda/CUDAContext.h`（Paddle compat 中不存在）

## 差异点列表

1. **`at::cuda::getCurrentCUDAStream()` 不存在**：Paddle compat 未提供该函数，整个调用块被 `#ifndef USE_PADDLE_API` 保护，Paddle 下只输出固定字符串 `"stream_skipped_paddle"`，无法进行真实对比。

---

# CUDA 工具类（CUDAGuard / CUDAStream / PhiloxCudaState 全部缺失）

> Paddle 头文件：`c10/cuda/CUDAGuard.h`、`c10/cuda/CUDAStream.h`、`c10/cuda/PhiloxCudaState.h`（Paddle compat 中均不存在）
> 测试文件：`test/CUDATest2.cpp`

## 差异点列表

以下类和相关头文件在 Paddle compat 中**完全缺失**，对应测试被 `#ifndef USE_PADDLE_API` 整块保护跳过：

| 缺失类/结构 | 头文件 |
|-------------|--------|
| `c10::cuda::CUDAGuard` | `c10/cuda/CUDAGuard.h` |
| `c10::cuda::OptionalCUDAGuard` | `c10/cuda/CUDAGuard.h` |
| `c10::cuda::CUDAStream` | `c10/cuda/CUDAStream.h` |
| `c10::cuda::getCurrentCUDAStream()` | `c10/cuda/CUDAStream.h` |
| `c10::cuda::PhiloxCudaState` | `c10/cuda/PhiloxCudaState.h` |

---

# TensorOptions（`requires_grad` 传递问题）

> Paddle 头文件：`c10/core/TensorOptions.h`

## 差异点列表

1. **`at::empty()` 不支持含 `requires_grad` 的 `TensorOptions`**：Paddle 在通过 `at::empty({...}, opts)` 创建 tensor 时，若 `opts` 含有 `requires_grad(true)` 会抛出异常。PyTorch 完整支持。当前测试已绕过：将含 `requires_grad` 的 `opts` 与用于创建 tensor 的 `opts_for_dtype` 分离，单独测试 `requires_grad()` 的读取，但实际上 Paddle 无法通过 `TensorOptions` 在 tensor 创建时传递梯度需求。
2. **`device_index()` 对 CPU 设备的返回值不同**：Torch 对 CPU 设备返回 `-1`（无显式 index）；Paddle 会将 CPU 规范化为 `cpu:0`，因此返回 `0`。

---

## 详细记录

- 测试用例：DeviceIndex
- 字段：`c10::TensorOptions().device(c10::Device(c10::kCPU)).device_index()`
- 差异：
    - Paddle 输出：`0`
    - Torch 输出：`-1`
- 原因：Torch 将 CPU 设备视为无显式 index；Paddle 会将 CPU 设备规范化为 `cpu:0`。
- 处理：已在测试文件中注释掉该字段输出，并添加 `DIFF` 标注说明。

---

# Tensor::resize_（Paddle 不支持）

> Paddle 头文件：`ATen/core/Tensor.h`

## 差异点列表

1. **`resize_()` 不支持**：Paddle 调用 `tensor.resize_({...})` 会抛出异常，PyTorch 完整支持原地调整 tensor 形状。当前测试用 try-catch 捕获异常并输出 `"1 "` 表示异常发生，无法对比实际 resize 结果。

---

# Tensor::pin_memory / is_pinned（语义与组合矩阵差异）

> Paddle 相关头文件：`ATen/core/TensorBody.h`、`phi/common/place.h`
> PyTorch 相关实现：`aten/src/ATen/native/Memory.cpp`

## 差异点列表

1. **PyTorch `pin_memory` 仅支持 CPU Tensor**：`_pin_memory` 明确 `TORCH_CHECK(self.device().is_cpu())`，非 CPU Tensor 直接报错。
2. **PyTorch `device` 参数已弃用**：`pin_memory(device)` 与 `is_pinned(device)` 传参会触发 deprecation warning，官方建议不再传入。
3. **PyTorch `device` 语义**：即使保留该参数，也仅用于选择 pinned allocator 的 accelerator type，不改变“只能 pin CPU Tensor”的规则。
4. **Paddle 原生 `Tensor.pin_memory()` 语义是 copy 到 pinned place**：调用 `_copy_to(CUDAPinnedPlace/XPUPinnedPlace)`，更接近 place 迁移语义。
5. **Paddle 底层支持 `CPUPlace -> GPUPinnedPlace/XPUPinnedPlace`**：内存拷贝路径在 `memcpy.cc` 中有专门分支。
6. **当前 Paddle ATen compat 实现反向限制**：`ATen/core/TensorBody.h` 里 `pin_memory` 对 CPUPlace 抛异常、仅允许 GPU/XPU 转 pinned，和 PyTorch 语义不一致，也和 Paddle 原生 Python 语义不一致。

## `device` 可传入范围（`pin_memory` 语境）

### PyTorch

- 形式上：`optional<torch::Device>`。
- 实际建议：不传（参数已弃用）。
- 兼容旧行为时：应传 accelerator device type（如 `cuda`/`xpu` 等），`cpu` 不属于有效加速器目标语义。

### Paddle

- 原生 Python `Tensor.pin_memory()`：无 `device` 参数（仅 `blocking`）。
- Paddle ATen compat `Tensor::pin_memory(optional<Device>)`：当前签名保留了 `device`，但实现里几乎未使用该参数决定目标 place（存在语义偏差）。

## Tensor 类型组合行为对比（`pin_memory`）

| 框架/实现 | CPU Tensor | GPU Tensor | XPU Tensor | 备注 |
|---|---|---|---|---|
| PyTorch | 支持（返回 CPU pinned） | 不支持（报错） | 不支持（报错） | 仅 dense CPU 可 pin |
| Paddle 原生 Python | 支持（copy 到 pinned place） | 支持（copy 到 pinned place） | 支持（copy 到 pinned place） | 语义偏 place 迁移 |
| Paddle ATen compat（当前） | 不支持（抛异常） | 支持 | 支持 | 与上两者不一致 |

## 修复方向

1. 若目标是 **PyTorch 对齐**：将 compat `Tensor::pin_memory` 改为仅允许 CPU 输入，非 CPU 报错，`device` 仅作可选后端提示并保持弃用语义。
2. 若目标是 **Paddle 原生对齐**：允许 CPU/GPU/XPU 都走 copy-to-pinned-place，但需在文档中明确这不是 PyTorch 的严格语义。
3. 二选一后，同步更新 `is_pinned(device)`、`to(..., pin_memory=...)` 与相关算子工厂函数文档，避免行为和测试标准不一致。

---

# TensorFactoryTest

## 差异点列表

1. **ScalarType::Bool 枚举值不同**：Paddle 的 DataType::BOOL = 10，Torch 的 ScalarType::Bool = 11。

---

## 详细记录

- 测试用例：TensorFromBoolArrayRef
- 字段：scalar_type（write_tensor_info_to_file 输出的 static_cast<int>(t.scalar_type())）
- 差异：
    - Paddle 输出：10
    - Torch 输出：11
- 原因：Paddle 与 Torch 框架的 ScalarType::Bool 枚举值不同（Paddle=10，Torch=11），属于设计差异。
- 处理：已在测试文件中注释掉该字段输出，并添加 DIFF 标注说明。

---

# CUDADataTypeTest

## 差异点列表

1. **`ScalarTypeToCudaDataType(Bool)` 支持范围不同**：Paddle compat 不支持 `Bool` 转 `cudaDataType`，会抛出异常；Torch 侧接口支持范围更完整。当前测试已跳过 `Bool`。
2. **`empty_cuda` 结果依赖运行时/构建环境**：Torch CUDA 版通常可成功创建 CUDA Tensor；Paddle compat 在未编译 CUDA 或运行时不可用时会抛异常并进入不可用分支。该差异属于环境差异，不属于接口语义差异。

---

## 详细记录

- 测试用例：GetCudaDataType
- 字段：`Bool` 类型的 `ScalarTypeToCudaDataType` 转换
- 差异：
    - Paddle：抛出 `Cannot convert ScalarType Bool to cudaDataType`
    - Torch：可返回对应的 `cudaDataType`
- 原因：Paddle compat 的 `ATen/cuda/CUDADataType.h` 未实现 `Bool` 分支。
- 处理：已在测试文件中跳过 `Bool` 的输出，并添加 `DIFF` 注释说明。

- 测试用例：EmptyCUDA / EmptyCudaDifferentDtype
- 字段：结果字符串（`cuda_empty` / `cuda_empty_int` / `cuda_not_available`）
- 差异：
    - Paddle 输出：`cuda_not_available`
    - Torch 输出：`cuda_empty`、`cuda_empty_int`
- 原因：该结果依赖 Paddle 是否为 GPU 版以及当前 CUDA 运行时是否可用，属于运行时/构建环境差异，而非接口行为差异。
- 处理：已在测试文件中保留调用、注释掉结果字符串输出，并添加 `DIFF` 注释说明。
