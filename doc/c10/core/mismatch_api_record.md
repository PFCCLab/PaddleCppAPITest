# Allocator

## 差异点列表

1.  **构造函数参数默认值**
2.  **拷贝语义**
3.  **`get_deleter()` 在默认构造后的返回值**
4.  **`clear()` 后 `get_deleter()` 的行为**
5.  **Device 类型和方法**
6.  **`allocation()` 方法**

---

## Diff 测试用例位置

测试文件：`test/c10/core/unmatch_AllocatorTest.cpp`

### 测试用例原文

#### 1. Diff_ConstructorDefaultDevice（构造函数参数默认值）

```cpp
TEST_F(AllocatorTest, Diff_ConstructorDefaultDevice) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();

#if USE_PADDLE_API
  // Paddle 支持不指定 device 的构造（使用默认 CPUPlace）
  c10::DataPtr ptr_default(static_cast<void*>(test_data_));
  file << "paddle_single_arg_ctor_supported ";
  file << std::to_string(ptr_default.get() == static_cast<void*>(test_data_))
       << " ";
#else
  // PyTorch 必须显式指定 device
  c10::DataPtr ptr_with_device(static_cast<void*>(test_data_),
                               c10::Device(c10::DeviceType::CPU));
  file << "torch_requires_device_arg ";
  file << std::to_string(ptr_with_device.get() ==
                         static_cast<void*>(test_data_))
       << " ";
#endif

  file.saveFile();
}
```

#### 2. Diff_CopySemantics（拷贝语义）

```cpp
TEST_F(AllocatorTest, Diff_CopySemantics) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();

#if USE_PADDLE_API
  // Paddle 支持拷贝构造
  c10::DataPtr original(static_cast<void*>(test_data_), phi::CPUPlace());
  c10::DataPtr copied(original);  // 拷贝构造
  c10::DataPtr assigned;
  assigned = original;  // 拷贝赋值

  file << "paddle_copy_supported ";
  // 拷贝后两个指针指向同一数据
  file << std::to_string(original.get() == copied.get()) << " ";
  file << std::to_string(original.get() == assigned.get()) << " ";
  // 原始对象仍然有效
  file << std::to_string(original.get() != nullptr) << " ";
#else
  // PyTorch 只支持移动，拷贝构造和拷贝赋值被删除
  c10::DataPtr original(static_cast<void*>(test_data_),
                        c10::Device(c10::DeviceType::CPU));
  c10::DataPtr moved(std::move(original));

  file << "torch_move_only ";
  file << std::to_string(moved.get() == static_cast<void*>(test_data_)) << " ";
  file << std::to_string(moved.get() != nullptr) << " ";
  file << std::to_string(true) << " ";  // 占位符保持输出长度一致
#endif

  file.saveFile();
}
```

#### 3. Diff_DefaultDeleter（get_deleter() 默认值）

```cpp
TEST_F(AllocatorTest, Diff_DefaultDeleter) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();

  c10::DataPtr default_ptr;

#if USE_PADDLE_API
  // Paddle: 默认 deleter 为 nullptr
  file << "paddle_default_deleter_null ";
  file << std::to_string(default_ptr.get_deleter() == nullptr) << " ";
#else
  // PyTorch: 默认 deleter 可能不为 nullptr
  file << "torch_default_deleter_may_exist ";
  bool has_deleter = (default_ptr.get_deleter() != nullptr);
  file << std::to_string(has_deleter || !has_deleter) << " ";  // 总是 true
#endif

  file.saveFile();
}
```

#### 4. Diff_ClearDeleterBehavior（clear() 后行为）

```cpp
TEST_F(AllocatorTest, Diff_ClearDeleterBehavior) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();

#if USE_PADDLE_API
  c10::DataPtr data_ptr(
      static_cast<void*>(test_data_), test_ctx_, test_deleter, phi::CPUPlace());
#else
  c10::DataPtr data_ptr(static_cast<void*>(test_data_),
                        test_ctx_,
                        test_deleter,
                        c10::Device(c10::DeviceType::CPU));
#endif

  // clear 前 deleter 应该正确设置
  file << std::to_string(data_ptr.get_deleter() == test_deleter) << " ";

  data_ptr.clear();

#if USE_PADDLE_API
  // Paddle: clear 后 deleter 被重置为 nullptr
  file << "paddle_clear_resets_deleter ";
  file << std::to_string(data_ptr.get_deleter() == nullptr) << " ";
#else
  // PyTorch: clear 后 deleter 可能仍然存在
  file << "torch_clear_keeps_deleter ";
  file << std::to_string(true) << " ";
#endif

  file.saveFile();
}
```

#### 5. Diff_DeviceType（Device 类型和方法）

```cpp
TEST_F(AllocatorTest, Diff_DeviceType) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();

#if USE_PADDLE_API
  c10::DataPtr data_ptr(static_cast<void*>(test_data_), phi::CPUPlace());
  // Paddle 使用 phi::Place，有 DebugString() 和 HashValue()
  std::string device_str = data_ptr.device().DebugString();
  size_t hash_value = data_ptr.device().HashValue();
  file << "paddle_phi_place ";
  file << std::to_string(!device_str.empty()) << " ";
  file << std::to_string(hash_value != 0 || hash_value == 0) << " ";
#else
  c10::DataPtr data_ptr(static_cast<void*>(test_data_),
                        c10::Device(c10::DeviceType::CPU));
  // PyTorch 使用 c10::Device，有 str() 方法
  std::string device_str = data_ptr.device().str();
  file << "torch_c10_device ";
  file << std::to_string(!device_str.empty()) << " ";
  file << std::to_string(device_str == "cpu") << " ";
#endif

  file.saveFile();
}
```

#### 6. Diff_AllocationMethod（allocation() 方法）

```cpp
TEST_F(AllocatorTest, Diff_AllocationMethod) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();

#if USE_PADDLE_API
  c10::DataPtr data_ptr(static_cast<void*>(test_data_), phi::CPUPlace());
  // Paddle 有 allocation() 方法
  auto alloc = data_ptr.allocation();
  file << "paddle_has_allocation_method ";
  file << std::to_string(alloc == nullptr) << " ";
#else
  c10::DataPtr data_ptr(static_cast<void*>(test_data_),
                        c10::Device(c10::DeviceType::CPU));
  // PyTorch 没有 allocation() 方法
  file << "torch_no_allocation_method ";
  file << std::to_string(true) << " ";
#endif

  file.saveFile();
}
```

---

## 输出对比

| 测试用例 | Paddle 输出 | Torch 输出 |
|---------|------------|------------|
| Diff_ConstructorDefaultDevice | `paddle_single_arg_ctor_supported 1` | `torch_requires_device_arg 1` |
| Diff_CopySemantics | `paddle_copy_supported 1 1 1` | `torch_move_only 1 1 1` |
| Diff_DefaultDeleter | `paddle_default_deleter_null 1` | `torch_default_deleter_may_exist 1` |
| Diff_ClearDeleterBehavior | `1 paddle_clear_resets_deleter 1` | `1 torch_clear_keeps_deleter 1` |
| Diff_DeviceType | `paddle_phi_place 1 1` | `torch_c10_device 1 1` |
| Diff_AllocationMethod | `paddle_has_allocation_method 1` | `torch_no_allocation_method 1` |

---

## 初步问题分析

1. **构造函数参数默认值**：PyTorch 的 DataPtr 构造函数要求显式传入 device 参数，而 Paddle 支持使用默认的 CPUPlace。

2. **拷贝语义**：PyTorch 的 DataPtr 删除了拷贝构造和拷贝赋值函数（仅支持移动语义），而 Paddle 支持完整的拷贝语义。

3. **get_deleter() 默认值**：PyTorch 默认构造的 DataPtr 的 deleter 可能不为 nullptr，而 Paddle 默认为 nullptr。

4. **clear() 后行为**：PyTorch 的 clear() 方法不会重置 deleter，而 Paddle 会将其重置为 nullptr。

5. **Device 类型**：PyTorch 使用 c10::Device（有 str() 方法），而 Paddle 使用 phi::Place（有 DebugString() 和 HashValue() 方法）。

6. **allocation() 方法**：Paddle 额外提供了 allocation() 方法返回底层 phi::Allocation 对象，PyTorch 没有此方法。

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

## Diff 测试用例位置

测试文件：`test/c10/core/DeviceTest.cpp`

### 测试用例原文

#### 1. IndexAndHasIndex（未指定 Index 时的默认行为）

```cpp
TEST_F(DeviceTest, IndexAndHasIndex) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();

  // CPU 设备
  c10::Device cpu_device(c10::kCPU);
  file << std::to_string(cpu_device.index()) << " ";
  file << (cpu_device.has_index() ? "1" : "0") << " ";

  // CUDA 设备 index=0
  c10::Device cuda_0(c10::kCUDA, 0);
  file << std::to_string(cuda_0.index()) << " ";
  file << (cuda_0.has_index() ? "1" : "0") << " ";

  // CUDA 设备 index=1
  c10::Device cuda_1(c10::kCUDA, 1);
  file << std::to_string(cuda_1.index()) << " ";
  file << (cuda_1.has_index() ? "1" : "0") << " ";

  // 字符串构造的设备
  c10::Device cpu_str("cpu");
  file << std::to_string(cpu_str.index()) << " ";
  file << (cpu_str.has_index() ? "1" : "0") << " ";

  c10::Device cuda0_str("cuda:0");
  file << std::to_string(cuda0_str.index()) << " ";
  file << (cuda0_str.has_index() ? "1" : "0") << " ";

  file.saveFile();
}
```

#### 2. ConstructorWithString（纯字符串解析行为）

```cpp
TEST_F(DeviceTest, ConstructorWithString) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();

  // "cpu" 字符串
  c10::Device cpu_str("cpu");
  write_device_result_to_file(&file, cpu_str);

  // "cpu:0" 字符串
  c10::Device cpu0_str("cpu:0");
  write_device_result_to_file(&file, cpu0_str);

  // "cuda" 字符串
  c10::Device cuda_str("cuda");
  write_device_result_to_file(&file, cuda_str);

  // "cuda:0" 字符串
  c10::Device cuda0_str("cuda:0");
  write_device_result_to_file(&file, cuda0_str);

  // "cuda:1" 字符串
  c10::Device cuda1_str("cuda:1");
  write_device_result_to_file(&file, cuda1_str);

  file.saveFile();
}
```

#### 3. ToString（GPU/CUDA 字符串表示）

```cpp
TEST_F(DeviceTest, ToString) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();

  c10::Device cpu_device(c10::kCPU);
  file << cpu_device.str() << " ";

  c10::Device cuda_0(c10::kCUDA, 0);
  file << cuda_0.str() << " ";

  c10::Device cuda_1(c10::kCUDA, 1);
  file << cuda_1.str() << " ";

  c10::Device cpu_str("cpu:0");
  file << cpu_str.str() << " ";

  c10::Device cuda_str("cuda:1");
  file << cuda_str.str() << " ";

  file.saveFile();
}
```

#### 4. DeviceType（底层类型枚举值）

```cpp
TEST_F(DeviceTest, DeviceType) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();

  c10::Device cpu_device(c10::kCPU);
  file << std::to_string(static_cast<int>(cpu_device.type())) << " ";

  c10::Device cuda_device(c10::kCUDA, 0);
  file << std::to_string(static_cast<int>(cuda_device.type())) << " ";

  // 从字符串解析
  c10::Device cpu_str("cpu");
  file << std::to_string(static_cast<int>(cpu_str.type())) << " ";

  c10::Device cuda_str("cuda:0");
  file << std::to_string(static_cast<int>(cuda_str.type())) << " ";

  file.saveFile();
}
```

#### 5. TensorDevice（默认 Tensor 所在设备）

```cpp
TEST_F(DeviceTest, TensorDevice) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();

  // 默认 CPU tensor
  at::Tensor cpu_tensor = at::zeros({2, 3});
  c10::Device cpu_dev = cpu_tensor.device();
  write_device_result_to_file(&file, cpu_dev);

  // 指定 CPU device 的 tensor
  at::Tensor cpu_tensor2 =
      at::zeros({2, 3}, at::TensorOptions().device(c10::kCPU));
  c10::Device cpu_dev2 = cpu_tensor2.device();
  write_device_result_to_file(&file, cpu_dev2);

  // 使用 TensorOptions 构造
  at::Tensor cpu_tensor3 =
      at::zeros({2, 3}, at::TensorOptions().device(c10::Device(c10::kCPU)));
  c10::Device cpu_dev3 = cpu_tensor3.device();
  write_device_result_to_file(&file, cpu_dev3);

  file.saveFile();
}
```

---

## 输出对比

| 测试用例 | Paddle 输出 | Torch 输出 |
|---------|------------|------------|
| IndexAndHasIndex (cpu_device) | `0 1` | `-1 0` |
| ConstructorWithString ("cpu") | `1 0 0 1 cpu:0` | `0 -1 0 0 cpu` |
| ConstructorWithString ("cuda") | `2 0 1 0 gpu:0` | `1 -1 1 0 cuda` |
| ToString (cuda_0) | `gpu:0` | `cuda:0` |
| DeviceType (cpu) | `1` | `0` |
| DeviceType (cuda) | `2` | `1` |

---

## 初步问题分析

1. **未指定 Index 默认行为**：PyTorch 使用 -1 表示无显式 index，has_index() 返回 false；Paddle 强制默认为 0，has_index() 返回 true。

2. **字符串解析行为**：PyTorch 解析 "cpu" 后保持无 index 状态；Paddle 自动补全为 "cpu:0"。

3. **GPU/CUDA 表示**：PyTorch 严格输出 "cuda"，Paddle 底层映射为 "gpu"。

4. **枚举值差异**：CPU 在 PyTorch=0, Paddle=1；CUDA 在 PyTorch=1, Paddle=2。

5. **默认 Tensor 设备**：PyTorch 默认 tensor 所在设备的 index 为 -1（无显式），Paddle 为 0。

---

提交的对齐 PR：https://github.com/PaddlePaddle/Paddle/pull/78066

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

# TensorOptions（`requires_grad` 传递问题）

> Paddle 头文件：`c10/core/TensorOptions.h`

## 差异点列表

1. **`at::empty()` 不支持含 `requires_grad` 的 `TensorOptions`**：Paddle 在通过 `at::empty({...}, opts)` 创建 tensor 时，若 `opts` 含有 `requires_grad(true)` 会抛出异常。PyTorch 完整支持。当前测试已绕过：将含 `requires_grad` 的 `opts` 与用于创建 tensor 的 `opts_for_dtype` 分离，单独测试 `requires_grad()` 的读取，但实际上 Paddle 无法通过 `TensorOptions` 在 tensor 创建时传递梯度需求。
2. **`device_index()` 对 CPU 设备的返回值不同**：Torch 对 CPU 设备返回 `-1`（无显式 index）；Paddle 会将 CPU 规范化为 `cpu:0`，因此返回 `0`。

---

## Diff 测试用例位置

测试文件：`test/c10/core/TensorOptionsTest.cpp`

### 测试用例原文

```cpp
// 测试 device_index() 对 CPU 设备的返回值
// [DIFF] 对于 `c10::TensorOptions().device(c10::Device(c10::kCPU))`，
// Paddle 返回 0（因为会将 CPU 规范化为 cpu:0），Torch 返回 -1
TEST_F(TensorOptionsTest, DeviceIndex) {
  auto opts = c10::TensorOptions().device(c10::Device(c10::kCPU));

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();

  // file << std::to_string(opts.device_index()) << " "; // [DIFF] 已注释
  file.saveFile();
}

// 测试 requires_grad 传递（测试已绕过）
TEST_F(TensorOptionsTest, ChainedSetters) {
  auto opts = c10::TensorOptions()
      .dtype(at::kDouble)
      .requires_grad(true);  // Paddle 不支持通过 TensorOptions 传递 requires_grad

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();

  // Paddle: requires_grad 会抛出异常，但此处单独测试 getter
  file << std::to_string(opts.requires_grad().value()) << " ";
  file.saveFile();
}
```

---

## 输出对比

| 测试用例 | Paddle 输出 | Torch 输出 |
|---------|------------|------------|
| DeviceIndex | 不序列化该字段（已注释） | 不序列化该字段（已注释） |
| ChainedSetters | `1` | `1` |

---

## 初步问题分析

1. **requires_grad 传递**：Paddle 不支持通过 TensorOptions 在创建 tensor 时传递 requires_grad 参数，会抛出异常。

2. **device_index() 返回值**：Paddle 将 CPU 设备规范化为 cpu:0，因此 device_index() 返回 0；PyTorch 返回 -1 表示无显式 index。

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

## 差异点列表

1. **默认 index 语义不一致**：PyTorch 默认为 `index = -1`（`has_index() = false`），Paddle 默认为 `index = 0`（`has_index() = true`）。

---

## Diff 测试用例位置

测试文件：`test/c10/core/DeviceTest.cpp`

### 测试用例原文

```cpp
TEST_F(DeviceCompatTest, HasIndex) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();

  c10::Device cpu_default(c10::kCPU);
  c10::Device cpu_0(c10::kCPU, 0);
  c10::Device cuda_default(c10::kCUDA);
  c10::Device cuda_1(c10::kCUDA, 1);

  bool cpu_default_has = cpu_default.has_index();
  bool cpu_0_has = cpu_0.has_index();
  bool cuda_default_has = cuda_default.has_index();
  bool cuda_1_has = cuda_1.has_index();

  // [DIFF] DeviceType::CPU 的默认 index 语义不同：Torch(-1, has_index=false) vs Paddle(0, has_index=true)
  // [DIFF] DeviceType::CUDA 的默认 index 语义不同：Torch(-1, has_index=false) vs Paddle(0, has_index=true)
  file << std::to_string(cpu_default_has || !cpu_default_has) << " ";
  file << std::to_string(cpu_0_has || !cpu_0_has) << " ";
  file << std::to_string(cuda_default_has || !cuda_default_has) << " ";
  file << std::to_string(cuda_1_has || !cuda_1_has) << " ";

  file.saveFile();
}
```

---

## 输出对比

| 测试用例 | Paddle 输出 | Torch 输出 |
|---------|------------|------------|
| HasIndex（原始） | `1 1 1 1` | `0 1 0 1` |

---

## 初步问题分析

Paddle 兼容层 `Device(DeviceType, DeviceIndex)` 的默认 index 设置为 `0`，而 PyTorch 默认为 `-1`（表示未显式指定设备索引），因此 `has_index()` 在默认构造路径上出现语义差异。

---

# Device（`str`）

> Paddle 头文件：`c10/core/Device.h`

## 差异点列表

1. **设备字符串规范不一致**：PyTorch 使用 `cpu/cuda`，Paddle 使用 `cpu:0/gpu` 风格。

---

## Diff 测试用例位置

测试文件：`test/c10/core/DeviceTest.cpp`

### 测试用例原文

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

  // [DIFF] PyTorch输出: cpu cpu:0 cuda:0 cuda:1
  // [DIFF] PaddlePaddle输出: cpu:0 cpu:0 gpu:0 gpu:1
  // file << cpu_str << " ";
  // file << cpu_0_str << " ";
  // file << cuda_0_str << " ";
  // file << cuda_1_str << " ";
  (void)cpu_str;
  (void)cpu_0_str;
  (void)cuda_0_str;
  (void)cuda_1_str;

  file.saveFile();
}
```

---

## 输出对比

| 测试用例 | Paddle 输出 | Torch 输出 |
|---------|------------|------------|
| DeviceStr | `cpu:0 cpu:0 gpu:0 gpu:1` | `cpu cpu:0 cuda:0 cuda:1` |

---

## 初步问题分析

Paddle 在设备命名（`gpu`）与 CPU 默认索引显式化（`cpu:0`）上的规范与 PyTorch（`cuda`、默认 `cpu`）不同，导致字符串层面的稳定差异。

---
