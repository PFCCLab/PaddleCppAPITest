# PyTorch 与 PaddlePaddle API 差异对比

---

## Device

> Paddle 头文件：`c10\core\Device.h`

**总结差异**

核心差异在于设备索引（Index）的默认分配机制与底层命名规范。PyTorch 允许设备处于"仅有类型、未指定具体索引"的抽象状态（Index 为 -1），并严格保留 cuda 命名；而 PaddlePaddle 更加具象化，强制要求任何设备都必须有明确的索引（若未指定则自动兜底赋值为 0），并在输出时会自动将 cuda 概念映射转化回其原生的 gpu 命名，同时两套框架底层对设备类型（如 CPU/GPU）的枚举整型值也完全不同。

---

**具体差异对比**

| 差异点 | PyTorch (torch_DeviceTest) | PaddlePaddle (paddle_DeviceTest) |
|--------|----------------------------|-----------------------------------|
| 未指定 Index 时的默认行为 <br>(如 Device(kCPU)) | 无索引：index = -1，has_index() = false | 强制默认 0：index = 0，has_index() = true |
| 纯字符串解析 <br>(如传入 "cpu" 或 "cuda") | 保持无索引状态，解析为 cpu 或 cuda | 自动补全为 0 号设备，解析为 cpu:0 或 gpu:0 |
| GPU/CUDA 字符串表示 <br>(如 str() 方法输出) | 严格输出为 cuda 或 cuda:0 | 底层映射为 GPU，输出为 gpu:0 或 gpu:1 |
| 底层类型枚举值 (Enum ID) <br>(转为 int 打印) | CPU = 0, CUDA = 1 | CPU = 1, CUDA/GPU = 2 (枚举映射不同) |
| 默认 Tensor 所在设备 | 处于无明确索引的 cpu 状态 | 明确挂载在 cpu:0 设备上 |

---

## BFloat16

> Paddle 头文件：`c10\util\BFloat16.h`

**总结差异**

核心差异在于 BFloat16 的 ScalarType 枚举值定义不同。PyTorch 与 PaddlePaddle 对于 BFloat16 数据类型分配了不同的内部枚举整数值，这反映了两个框架在数据类型编码体系上的差异。

---

**具体差异对比**

| 差异点 | PyTorch (torch_HalfBFloat16Test) | PaddlePaddle (paddle_HalfBFloat16Test) |
|--------|-----------------------------------|------------------------------------------|
| BFloat16 枚举值 | **11** | **15** |
| ComplexFloat 枚举值 | **8** | **9** |

---

## DefaultDtype

> Paddle 头文件：`c10\core\DefaultDtype.h`

**总结差异**

核心差异在于默认数据类型（Default Dtype）的枚举值定义不同。PyTorch 与 PaddlePaddle 对于 BFloat16 和 ComplexFloat（复数类型）分配了不同的内部枚举整数值，这反映了两个框架在数据类型编码体系上的差异。

---

**具体差异对比**

| 差异点 | PyTorch (torch_DefaultDtypeTest) | PaddlePaddle (paddle_DefaultDtypeTest) |
|--------|----------------------------------|----------------------------------------|
| BFloat16 枚举值 | **11** | **15** |
| ComplexFloat 枚举值 | **8** | **9** |

---

## IValue

> Paddle 头文件：`ATen/core/ivalue.h`

**总结差异**

核心差异在于 IValue 类型所在的命名空间以及方法命名风格完全不同。PyTorch 的 IValue 位于 `c10` 命名空间，使用 camelCase 方法名；而 Paddle 的 IValue 位于 `torch` 命名空间，使用 snake_case 方法名。这导致无法写出同时兼容两个框架的代码，必须使用条件编译。

---

**具体差异对比**

| 差异点 | PyTorch (torch_IValueTest) | PaddlePaddle (paddle_IValueTest) |
|--------|----------------------------|-----------------------------------|
| IValue 类型命名空间 | `c10::IValue` | `torch::IValue` (c10 命名空间不存在) |
| 判断空值方法 | `isNone()` | `is_none()` |
| 判断布尔方法 | `isBool()` | `is_bool()` |
| 提取布尔方法 | `toBool()` | `to_bool()` |
| 判断整数方法 | `isInt()` | `is_int()` |
| 提取整数方法 | `toInt()` | `to_int()` |
| 判断浮点方法 | `isDouble()` | `is_double()` |
| 提取浮点方法 | `toDouble()` | `to_double()` |
| 判断字符串方法 | `isString()` | `is_string()` |
| 提取字符串方法 | `toStringRef()` | `to_string()` |
| 判断张量方法 | `isTensor()` | `is_tensor()` |
| 提取张量方法 | `toTensor()` | `to_tensor()` |
| 判断列表方法 | `isList()` | `is_list()` |
| 提取列表方法 | `toList()` | `to_list()` |
| 判断元组方法 | `isTuple()` | `is_tuple()` |
| 提取元组方法 | `toTuple()` | `to_tuple()` |
| 获取类型标签 | `tagKind()` | **不存在** |
| Optional 支持 | `c10::optional` | `paddle::optional` |

**兼容性结论**

由于 IValue 类型在两个框架中位于完全不同的命名空间（`c10` vs `torch`），且方法命名风格不同（camelCase vs snake_case），无法写出不使用条件编译的兼容代码。建议：

1. **跳过 IValue 测试** - 差异过大，不适合做对比测试
2. **使用条件编译** (`#if USE_PADDLE_API`) - 分别为两个框架编写代码

---

## SparseTensor

> Paddle 头文件：`ATen/ops/sparse_coo_tensor.h`、`ATen/ops/sparse_csr_tensor.h`

**总结差异**

核心差异在于 COO 稀疏张量在未指定 size 时的推断行为。PyTorch 能够根据 indices 的内容正确推断出完整的 size（例如从 indices 中的最大值推算出维度大小）；而 Paddle 在某些情况下推断出的 size 第一个维度为 0，导致 size 推断不完整。

---

**具体差异对比**

| 差异点 | PyTorch (torch_SparseTensorTest) | PaddlePaddle (paddle_SparseTensorTest) |
|--------|---------------------------------|----------------------------------------|
| sparse_coo_tensor 无 size 推断 | 正确推断：2 2 2 | 推断错误：0 2 2 (第一个维度为 0) |

---

## AccumulateType (Diff 2026-03-07)

> Paddle 头文件：`ATen/AccumulateType.h`

**Diff 信息**

```
Paddle (旧): Bool->10
Paddle (新): Bool->11
PyTorch: Bool->11
```

Paddle 已修复 Bool 类型的累加类型返回值，现与 PyTorch 一致（返回 11，即 Bool 类型）。

---

## AccumulateType (原始差异)

> Paddle 头文件：`ATen/AccumulateType.h`

**总结差异**

核心差异在于 `toAccumulateType` 函数对 `bool` 类型在 CPU 和 CUDA 设备上的返回值不同。Paddle 返回 `c10::complex<double>` (枚举值 10)，而 PyTorch 返回 `bool` (枚举值 11)。这可能是一个 Paddle 兼容层的 bug，因为头文件中声明的是 `CPU_ACC_TYPE(bool, bool)` 和 `CUDA_ACC_TYPE(bool, bool)`，期望返回 `bool` 类型。

---

**具体差异对比**

| 差异点 | PyTorch (torch_AccumulateTypeTest) | PaddlePaddle (paddle_AccumulateTypeTest) |
|--------|-----------------------------------|------------------------------------------|
| Bool CPU 累加类型 | **11** (Bool) | **10** (ComplexDouble) |
| Bool CUDA 累加类型 | **11** (Bool) | **10** (ComplexDouble) |

---

## OptionalArrayRef

> Paddle 头文件：`c10\util\OptionalArrayRef.h`

**总结差异**

核心差异在于运行时内存地址和对象标识符。`OptionalArrayRef` 类的核心功能（has_value、size、元素访问等）在两个框架中表现一致，但部分测试用例会输出运行时内存地址或内部对象指针，这些值在 Paddle 和 PyTorch 之间存在差异，属于正常的运行时差异，不影响功能正确性。

---

**具体差异对比**

| 差异点 | PyTorch (torch_OptionalArrayRefTest) | PaddlePaddle (paddle_OptionalArrayRefTest) |
|--------|--------------------------------------|---------------------------------------------|
| 内存地址值 (Position 25) | `140728760716544` | `140735674875072` |
| 内部标识符 (Positions 30-33) | `8092791957357752180 8232980759530400116 7301573481235833202 128060694099059` | `5719401567998992752 4714250036893545584 6081659751708586610 32783537689359205` |

**差异分析**

1. **内存地址差异**：位置 25 的数值差异是运行时内存地址，每次程序运行都可能不同，属于正常行为
2. **内部对象标识符差异**：位置 30-33 的数值差异可能来自内部对象的唯一标识符或指针，在不同框架实现中自然不同

**功能一致性**

以下核心功能在两个框架中表现一致：
- 默认构造：has_value() = false
- nullopt 构造：has_value() = false
- 单元素构造：has_value() = true，元素值正确
- 向量构造：has_value() = true，元素顺序和数量正确
- initializer_list 构造：功能一致
- has_value() 方法：行为一致
- value() 方法：行为一致
- reset() 方法：行为一致
- swap() 方法：行为一致
- emplace() 方法：行为一致
- slice() 方法：行为一致
- 相等运算符：行为一致

**结论**

OptionalArrayRef 在功能层面完全兼容，仅有运行时内存地址和内部标识符存在差异，这不影响正常使用。

---
