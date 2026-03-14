#### 记录PaddleCPPAPITest仓库检测出来的接口不一致情况，所有差异都记录了对应diff的测试文件和测试用例。在测试文件中也已标注，如有需要请直接搜索关键字 `[DIFF]` 定位到具体的差异点和测试用例。之前注释掉的可通过，但输出diff的测试现在已经打开了，运行./test/result_cmp.sh build可以看到全部diff。

## 按差异类型分组（便于 Review）

| 分类 | 测试 | 主要表现 |
|---|---|---|
| 语义差异（设计/规范不同） | `AccumulateTypeTest`、`DefaultDtypeTest`、`DeviceGuardTest`、`DeviceTest`、`HalfBFloat16Test`、`ScalarTypeTest`、`SparseTensorTest`、`StorageTest`、`TensorFactoryTest`、`TensorOptionsTest`、`TensorTest` | 默认值、枚举值、字符串规范或推断规则不同，且可稳定复现 |
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
| EqualTest | `... 0 0 1 ...` | `... 0 exception ...` |
| HalfBFloat16Test | `... 5 15` | `... 5 11` |
| ScalarTypeTest | `... QInt8 QUInt8 ...` | `... UNKNOWN_SCALAR UNKNOWN_SCALAR ...` |
| SelectTest | `... SelectNegativeDim 的真实结果 ...` | `known_crash_on_negative_dim` |
| SparseTensorTest | `... InferSize: 2 2 2 ...` | `... InferSize: 0 2 2 ...` |
| StorageTest | `... shared_alias=0 ...` | `... shared_alias=1 ...` |
| TensorFactoryTest | `... bool scalar_type=11 ...` | `... bool scalar_type=10 ...` |
| TensorOptionsTest | `... device_index=-1 ...` | `... device_index=0 ...` |
| TensorPtrTest | `const_ptr[0], mut_ptr[0]` | `typed_ptr_unavailable_on_paddle_compat ...` |
| TensorTest | `... device.type=0, get_device=-1 ...` | `... device.type=1, get_device=0 ...` |

## 建议跟踪优先级

1. **P0（稳定语义差异）**：`Device*`、`TensorOptionsTest`、`DefaultDtypeTest`、`HalfBFloat16Test`、`TensorFactoryTest`、`TensorTest`。
2. **P1（稀疏/存储一致性）**：`SparseTensorTest`、`StorageTest`、`AccumulateTypeTest`、`ScalarTypeTest`。
3. **P2（环境与实现缺口）**：`CUDADataTypeTest`、`EmptyOpsTest`、`EqualTest`、`SelectTest`、`TensorPtrTest`、`OptionalArrayRefTest`。

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

测试文件：`test/unmatch_AllocatorTest.cpp`

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

测试文件：`test/DeviceTest.cpp`

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

测试文件：`test/HalfBFloat16Test.cpp`

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

测试文件：`test/DefaultDtypeTest.cpp`

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

# IValue

> Paddle 头文件：`ATen/core/ivalue.h`

## 差异点列表

1.  **命名空间**：PyTorch 为 `c10::IValue`；Paddle 为 `torch::IValue`（c10 命名空间中不存在 IValue）
2.  **方法命名风格**：PyTorch 使用 camelCase（如 `isNone()`、`toBool()`）；Paddle 使用 snake_case（如 `is_none()`、`to_bool()`）
3.  **`tagKind()` 方法**：PyTorch 存在；Paddle 中**不存在**
4.  **字符串提取方法**：PyTorch 为 `toStringRef()`；Paddle 为 `to_string()`

---

## Diff 测试用例位置

测试文件：`test/unmatch_IValueTest.cpp`

### 测试用例原文

```cpp
// 测试 IValue 基本构造
TEST_F(IValueTest, None) {
  auto iv = c10::IValue();
  file << std::to_string(iv.isNone()) << " ";  // PyTorch: isNone()
  file.saveFile();
}

TEST_F(IValueTest, Bool) {
  auto iv_true = c10::IValue(true);
  auto iv_false = c10::IValue(false);
  file << std::to_string(iv_true.toBool()) << " ";  // PyTorch: toBool()
  file << std::to_string(iv_false.toBool()) << " ";
  file.saveFile();
}

TEST_F(IValueTest, String) {
  auto iv = c10::IValue(std::string("hello_world"));
  // PyTorch: toStringRef()
  // Paddle: to_string()
  file << iv.toStringRef() << " ";
  file.saveFile();
}
```

---

## 输出对比

| 测试用例 | Paddle 输出 | Torch 输出 |
|---------|------------|------------|
| None | 需使用 `is_none()` | `isNone()` |
| Bool | 需使用 `to_bool()` | `toBool()` |
| String | `to_string()` | `toStringRef()` |

---

## 初步问题分析

1. **命名空间差异**：Paddle 将 IValue 定义在 `torch` 命名空间，而 PyTorch 在 `c10` 命名空间，导致同时引用两库时出现符号冲突。

2. **方法命名风格**：PyTorch 使用 camelCase（如 isNone、toBool），Paddle 使用 snake_case（如 is_none、to_bool）。

3. **tagKind() 方法缺失**：Paddle 的 IValue 实现中没有 tagKind() 方法。

4. **字符串提取方法**：PyTorch 使用 toStringRef()，Paddle 使用 to_string()。

---

# SparseTensor

> Paddle 头文件：`ATen/ops/sparse_coo_tensor.h`、`ATen/ops/sparse_csr_tensor.h`

## 差异点列表

1.  **sparse_coo_tensor 无 size 推断行为**：PyTorch 能根据 indices 内容正确推断完整 size（如 `2 2 2`）；Paddle 推断结果第一个维度为 0（如 `0 2 2`）

---

## Diff 测试用例位置

测试文件：`test/ops/SparseTensorTest.cpp`

### 测试用例原文

```cpp
// COO 带推断 size
TEST_F(SparseTensorTest, SparseCOOInferSize) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();

  // indices: [2, 3] -> values: [3]
  at::Tensor indices = at::tensor({{0, 1, 2}, {0, 1, 2}}, at::kLong);
  at::Tensor values = at::tensor({1.0, 2.0, 3.0}, at::kFloat);

  // 不指定 size，让框架推断
  at::Tensor sparse = at::sparse_coo_tensor(indices, values);

  file << std::to_string(sparse.size(0)) << " ";
  file << std::to_string(sparse.size(1)) << " ";
  file << std::to_string(sparse.size(2)) << " ";

  file.saveFile();
}
```

---

## 输出对比

| 测试用例 | Paddle 输出 | Torch 输出 |
|---------|------------|------------|
| SparseCOOInferSize | `0 2 2` | `2 2 2` |

---

## 初步问题分析

Paddle 在使用 sparse_coo_tensor(indices, values) 不指定 size 参数时，无法正确推断第一个维度的大小，会返回 0；而 PyTorch 能正确推断为 2。

---

# OptionalArrayRef

> Paddle 头文件：`c10\util\OptionalArrayRef.h`

## 差异点列表

1.  **运行时内存地址值**：两框架输出的内存地址不同（属正常运行时差异，不影响功能）
2.  **内部对象标识符**：两框架内部唯一标识符数值不同（属正常实现差异，不影响功能）
3.  **FromOptionalArrayRef 临时对象悬空引用**：
    `std::optional<c10::ArrayRef<int64_t>>(std::vector<int64_t>{...})`
    让 `ArrayRef` 指向临时 vector，`front()` 输出随机内存值，Torch/Paddle diff。
    已按测试规范在 `OptionalArrayRefTest.cpp` 添加 `DIFF` 标注并注释该不稳定输出字段，仅保留 `has_value/size`。

> 注：OptionalArrayRef 核心功能（has_value、size、元素访问、reset、swap、emplace、slice 等）在两个框架中完全兼容，仅运行时地址和标识符存在差异。

---

## Diff 测试用例位置

测试文件：`test/OptionalArrayRefTest.cpp`

### 测试用例原文

```cpp
// DIFF: std::vector<int64_t>{1, 2, 3, 4, 5} 是临时对象，传入 OptionalArrayRef
// 在语句结束后被销毁， OptionalArrayRef 内部 ArrayRef
// 指向的内存已释放，继续访问会导致未定义行为
TEST_F(OptionalArrayRefTest, InPlaceConstruction) {
  c10::OptionalArrayRef<int64_t> arr(std::vector<int64_t>{1, 2, 3, 4, 5});

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();

  // [DIFF] 此处访问可能导致随机值或崩溃
  // file << std::to_string(arr.front()) << " ";  // 已注释
  file << std::to_string(arr.has_value()) << " ";
  file << std::to_string(arr->size()) << " ";

  file.saveFile();
}
```

---

## 输出对比

| 测试用例 | Paddle 输出 | Torch 输出 |
|---------|------------|------------|
| InPlaceConstruction | `1 5`（稳定字段） | `1 5`（稳定字段） |
| front() 值 | 不稳定（随机值） | 不稳定（随机值） |

---

## 初步问题分析

OptionalArrayRef 核心功能在两个框架中完全兼容。差异仅在于：
1. 运行时内存地址值不同（正常差异）
2. 内部对象标识符不同（正常差异）
3. 临时对象悬空引用问题：使用 std::vector 临时对象构造 OptionalArrayRef 时，ArrayRef 会指向已释放的内存，导致未定义行为。

---

# at::indexing（Slice / EllipsisIndexType）

> Paddle 头文件：`ATen/TensorIndexing.h`

## 差异点列表

- [x] **头文件路径不同**：PyTorch 为 `ATen/TensorIndexing.h`；Paddle compat 为 `ATen/indexing.h`
- [ ] **`Tensor::operator[](Slice)` 不支持**：PyTorch 的 `Tensor::operator[]` 接受 `at::indexing::Slice`；Paddle compat 的 `operator[]` 仅重载 `int64_t`，传入 `Slice` 会编译报错
- [x] **多维 Slice 索引写法不同**：
    - PyTorch：`t.index({Slice(0,2), Slice(1,3)})` —— 接受 `std::initializer_list<TensorIndex>`
    - Paddle：`t.index(std::vector<at::indexing::Slice>{Slice(0,2), Slice(1,3)})` —— 仅重载 `std::vector<Slice>`
4.  **`TensorIndex` 能力对齐状态**：Paddle compat 已提供 `TensorIndex`，但 `Tensor::operator[](Slice)` 仍不支持，实际使用仍需通过 `index(...)` 路径。

---

## Diff 测试用例位置

测试文件：`test/IndexingTest.cpp`

### 测试用例原文

```cpp
// Test using indexing with tensors
TEST_F(IndexingTest, TensorIndexing) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();

  at::Tensor t = at::arange(12, at::kInt).view({3, 4});

  // 【API 差异】Paddle compat 的 Tensor::operator[] 仅重载 int64_t，不支持传入
  // Slice；须改用 index(std::vector<at::indexing::Slice>) 接口。PyTorch 支持
  // operator[](Slice) 及 index({Slice, ...}) 两种写法。
#if USE_PADDLE_API
  at::Tensor result = t.index(std::vector<at::indexing::Slice>{
      at::indexing::Slice(), at::indexing::Slice()});
#else
  at::Tensor result = t.index({at::indexing::Slice(), at::indexing::Slice()});
#endif
  file << std::to_string(result.dim()) << " ";
  file << std::to_string(result.numel()) << " ";
  file.saveFile();
}

// Test Slice indexing
TEST_F(IndexingTest, SliceIndexing) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();

  at::Tensor t = at::arange(12, at::kInt).view({3, 4});

  // 【API 差异】同上：Paddle 不支持链式 operator[](Slice)，
  // 多维 Slice 须放入同一个 std::vector<Slice> 传给 index()；
  // PyTorch 可用 index({Slice(0,2), Slice(1,3)}) 的 initializer_list 写法。
#if USE_PADDLE_API
  at::Tensor result = t.index(std::vector<at::indexing::Slice>{
      at::indexing::Slice(0, 2), at::indexing::Slice(1, 3)});
#else
  at::Tensor result =
      t.index({at::indexing::Slice(0, 2), at::indexing::Slice(1, 3)});
#endif
  file << std::to_string(result.dim()) << " ";
  file << std::to_string(result.size(0)) << " ";
  file << std::to_string(result.size(1)) << " ";
  file.saveFile();
}
```

---

## 输出对比

| 测试用例 | Paddle 输出 | Torch 输出 |
|---------|------------|------------|
| TensorIndexing | `2 12` | `2 12` |
| SliceIndexing | `2 2 3` | `2 2 3` |

（注：输出相同，但调用方式不同）

---

## 初步问题分析

1. **头文件路径**：PyTorch 使用 `ATen/TensorIndexing.h`，Paddle 使用 `ATen/indexing.h`。

2. **operator[] 不支持 Slice**：Paddle 的 Tensor::operator[] 仅重载了 int64_t，不支持传入 Slice，需要使用 index() 方法。

3. **多维 Slice 写法**：PyTorch 支持 `t.index({Slice, Slice})` 写法（initializer_list），Paddle 只能使用 `t.index(std::vector<Slice>{})`。

4. **`TensorIndex` 已具备，但索引入口仍有差异**：`TensorIndex` 已在 compat 中实现；主要差异仍是 `operator[](Slice)` 与部分索引入口写法。

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

测试文件：`test/ScalarTypeTest.cpp`

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

# Exception 宏（TORCH_CHECK_EQ / TORCH_CHECK_NE 失败语义差异）

> Paddle 头文件：`c10/util/Exception.h`

## 差异点列表

1. **`TORCH_CHECK_EQ` 失败行为**：PyTorch 调用 `abort()` 终止进程（测试用 `EXPECT_DEATH` 捕获）；Paddle 抛出 C++ 异常（测试用 try-catch 捕获）。
2. **`TORCH_CHECK_NE` 失败行为**：同上，两者失败行为不一致。

当前代码通过 `#if USE_PADDLE_API` 分叉两套检测逻辑以绕过差异，但这导致两个平台实际走不同测试路径，无法真正对比行为。

---

## Diff 测试用例位置

测试文件：`test/ExceptionTest.cpp`

### 测试用例原文

```cpp
// 测试 TORCH_CHECK_EQ 失败行为
TEST_F(ExceptionTest, TorchCheckEqFailure) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();

#if USE_PADDLE_API
  // Paddle: 抛出异常
  try {
    TORCH_CHECK_EQ(1, 2, "Values should be equal");
    file << "no_exception ";
  } catch (const c10::Error& e) {
    file << "c10::Error ";
  }
#else
  // PyTorch: 调用 abort()，使用 EXPECT_DEATH 捕获
  // 在非 death test 中直接跳过
  file << "skipped ";
#endif
  file.saveFile();
}

// 测试 TORCH_CHECK_NE 失败行为
TEST_F(ExceptionTest, TorchCheckNe) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();

#if USE_PADDLE_API
  // Paddle: 抛出异常
  try {
    TORCH_CHECK_NE(1, 1, "Values should not be equal");
    file << "no_exception ";
  } catch (const c10::Error& e) {
    file << "c10::Error ";
  }
#else
  // PyTorch: 调用 abort()
  file << "skipped ";
#endif
  file.saveFile();
}
```

---

## 输出对比

| 测试用例 | Paddle 输出 | Torch 输出 |
|---------|------------|------------|
| TorchCheckEqFailure | `c10::Error` | `skipped`（需用 EXPECT_DEATH） |
| TorchCheckNe | `c10::Error` | `skipped`（需用 EXPECT_DEATH） |

---

## 初步问题分析

1. **TORCH_CHECK_EQ 失败行为**：PyTorch 调用 abort() 终止进程，Paddle 抛出 C++ 异常。
2. **TORCH_CHECK_NE 失败行为**：同上。

当前通过条件编译分叉两套测试逻辑，导致无法真正对比两框架的行为差异。

---

# CUDA Context（`at::cuda::getCurrentCUDAStream` 测试路径被跳过）

> Paddle 头文件：`ATen/cuda/CUDAContext.h`

## 差异点列表

1. **常规回归中该项未执行**：虽然 `unmatch_CUDAContextTest.cpp` 内已改为双端调用，但它不在常规构建集合，`result_cmp` 默认不会比较该项。

---

## Diff 测试用例位置

测试文件：`test/unmatch_CUDAContextTest.cpp`

### 测试用例原文

```cpp
// 测试 getCurrentCUDAStream
TEST_F(CUDAContextTest, GetCurrentCUDAStream) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();

try {
  auto stream = at::cuda::getCurrentCUDAStream();
  (void)stream;
  file << "stream_available ";
} catch (...) {
  file << "stream_not_available ";
}
  file.saveFile();
}
```

---

## 输出对比

| 测试用例 | Paddle 输出 | Torch 输出 |
|---------|------------|------------|
| GetCurrentCUDAStream | 不在常规 `result_cmp` 集合中 | 不在常规 `result_cmp` 集合中 |

---

## 初步问题分析

`ATen/cuda/CUDAContext.h` 已在 compat 提供。当前问题不是“接口缺失”，而是该节仍按 `unmatch` 管理，未纳入常规回归。

---

# CUDA 工具类（CUDAGuard / CUDAStream / PhiloxCudaState：接口形态有差异）

> Paddle 头文件：`c10/cuda/CUDAGuard.h`、`c10/cuda/CUDAStream.h`、`c10/cuda/PhiloxCudaState.h`
> 测试文件：`test/CUDATest2.cpp`

## 差异点列表

当前差异并非“文件缺失”，而是**接口形态与测试写法不一致**，导致对应测试被 `#ifndef USE_PADDLE_API` 整块保护跳过：

| 相关类/结构 | 头文件 | 现状 |
|-------------|--------|------|
| `c10::cuda::CUDAGuard` | `c10/cuda/CUDAGuard.h` | 可用，构造/成员形态与 Torch 有差异 |
| `c10::cuda::OptionalCUDAGuard` | `c10/cuda/CUDAGuard.h` | 可用 |
| `c10::cuda::CUDAStream` | `c10/cuda/CUDAStream.h` | 可用，默认构造与常量形态有差异 |
| `c10::cuda::getCurrentCUDAStream()` | `c10/cuda/CUDAStream.h` | 可用 |
| `PhiloxCudaState` | `c10/cuda/PhiloxCudaState.h` | 可用，但命名空间与 Torch 测试写法不同 |

---

## Diff 测试用例位置

测试文件：`test/CUDATest2.cpp`

### 测试用例原文

```cpp
// Paddle 路径仅占位，Torch 路径执行真实同步
TEST_F(CUDATest2, StreamSynchronize) {
#ifndef USE_PADDLE_API
  auto stream = c10::cuda::getCurrentCUDAStream();
  c10::cuda::stream_synchronize(stream.stream());
#else
  file << "stream_sync_placeholder ";
#endif
}

// 大量 CUDA 工具类用例仅在 Torch 路径编译
#ifndef USE_PADDLE_API
TEST_F(CUDATest2, CUDAGuardDefault) { /* ... */ }
TEST_F(CUDATest2, CUDAStreamDefault) { /* ... */ }
TEST_F(CUDATest2, PhiloxCudaStateDefault) { /* ... */ }
#endif
```

---

## 输出对比

| 测试用例 | Paddle 输出 | Torch 输出 |
|---------|------------|------------|
| CUDAGuardDefault | 未执行（`#ifndef USE_PADDLE_API` 下被编译排除） | 已执行 |
| CUDAStreamDefault | 未执行（`#ifndef USE_PADDLE_API` 下被编译排除） | 已执行 |
| PhiloxCudaStateDefault | 未执行（`#ifndef USE_PADDLE_API` 下被编译排除） | 已执行 |

---

## 初步问题分析

Paddle compat 已提供上述 CUDA 相关头文件与主要类/函数。当前差异的根因是：
- `test/CUDATest2.cpp` 中相关用例整体被 `#ifndef USE_PADDLE_API` 排除，Paddle 路径缺少同等覆盖；
- 兼容层接口形态与 Torch 侧测试写法不完全一致（例如构造方式、命名空间与成员能力）。

---

# TensorOptions（`requires_grad` 传递问题）

> Paddle 头文件：`c10/core/TensorOptions.h`

## 差异点列表

1. **`at::empty()` 不支持含 `requires_grad` 的 `TensorOptions`**：Paddle 在通过 `at::empty({...}, opts)` 创建 tensor 时，若 `opts` 含有 `requires_grad(true)` 会抛出异常。PyTorch 完整支持。当前测试已绕过：将含 `requires_grad` 的 `opts` 与用于创建 tensor 的 `opts_for_dtype` 分离，单独测试 `requires_grad()` 的读取，但实际上 Paddle 无法通过 `TensorOptions` 在 tensor 创建时传递梯度需求。
2. **`device_index()` 对 CPU 设备的返回值不同**：Torch 对 CPU 设备返回 `-1`（无显式 index）；Paddle 会将 CPU 规范化为 `cpu:0`，因此返回 `0`。

---

## Diff 测试用例位置

测试文件：`test/TensorOptionsTest.cpp`

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

# Tensor::resize_（Paddle 不支持）

> Paddle 头文件：`ATen/core/Tensor.h`

## Diff 测试用例位置

测试文件：`test/TensorTest.cpp`

### 测试用例原文

```cpp
// 测试 resize_ - Paddle不支持，会抛出异常
TEST_F(TensorTest, Resize) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();
  try {
    tensor.resize_({4, 5});
    file << "0 ";
  } catch (const std::exception& e) {
    file << "1 ";
  }
  file.saveFile();
}
```

---

## 输出对比

| 测试用例 | Paddle 输出 | Torch 输出 |
|---------|------------|------------|
| Resize | `1`（抛出异常） | `0`（成功） |

---

## 初步问题分析

Paddle 不支持 Tensor::resize_() 方法，调用时会抛出异常；PyTorch 完整支持原地调整 tensor 形状。

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

## Diff 测试用例位置

测试文件：`test/ops/TensorFactoryTest.cpp`

### 测试用例原文

```cpp
// 测试从 Bool 数组创建 Tensor
TEST_F(TensorFactoryTest, TensorFromBoolArrayRef) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();

  std::vector<bool> bool_data = {true, false, true};
  at::Tensor t = at::tensor(bool_data);

  // [DIFF] Paddle: scalar_type = 10 (DataType::BOOL)
  // Torch: scalar_type = 11 (ScalarType::Bool)
  // file << std::to_string(static_cast<int>(t.scalar_type())) << " "; // [DIFF]

  file << std::to_string(t.dim()) << " ";
  file << std::to_string(t.numel()) << " ";
  file.saveFile();
}
```

---

## 输出对比

| 测试用例 | Paddle 输出 | Torch 输出 |
|---------|------------|------------|
| TensorFromBoolArrayRef | `1 3` | `1 3` |

（注：scalar_type 字段已注释，仅对比其他字段）

---

## 初步问题分析

Paddle 与 PyTorch 的 ScalarType::Bool 枚举值不同：Paddle = 10，Torch = 11。这是两个框架设计上的差异。

---

# CUDADataTypeTest

## 差异点列表

1. **`ScalarTypeToCudaDataType(Bool)` 支持范围不同**：Paddle compat 不支持 `Bool` 转 `cudaDataType`，会抛出异常；Torch 侧接口支持范围更完整。当前测试已跳过 `Bool`。
2. **`empty_cuda` 结果依赖运行时/构建环境**：Torch CUDA 版通常可成功创建 CUDA Tensor；Paddle compat 在未编译 CUDA 或运行时不可用时会抛异常并进入不可用分支。该差异属于环境差异，不属于接口语义差异。

---

## Diff 测试用例位置

测试文件：`test/CUDADataTypeTest.cpp`

### 测试用例原文

```cpp
// 测试 ScalarTypeToCudaDataType 对 Bool 的支持
TEST_F(CUDADataTypeTest, GetCudaDataType) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();

  // 测试 Bool - [DIFF] Paddle 不支持，会抛出异常
  // file << std::to_string(
  //     at::cuda::ScalarTypeToCudaDataType(c10::ScalarType::Bool)) << " "; // [DIFF]

  file << "cuda_type_test ";
  file.saveFile();
}

// 测试 empty_cuda
TEST_F(CUDADataTypeTest, EmptyCUDA) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();

  try {
    at::Tensor t = at::empty_cuda({2, 3}, at::TensorOptions().dtype(at::kFloat));
    file << "cuda_empty ";
  } catch (const std::exception& e) {
    // Paddle 非 GPU 版或 CUDA 不可用时会抛异常
    file << "cuda_not_available ";
  }
  file.saveFile();
}
```

---

## 输出对比

| 测试用例 | Paddle 输出 | Torch 输出 |
|---------|------------|------------|
| GetCudaDataType | `cuda_type_test` | 正常输出（包含 Bool） |
| EmptyCUDA | `cuda_not_available` | `cuda_empty` |

---

## 初步问题分析

1. **ScalarTypeToCudaDataType(Bool)**：Paddle 未实现 Bool 到 cudaDataType 的转换，会抛出异常。

2. **empty_cuda**：属于运行时环境差异，取决于 Paddle 是否编译了 CUDA 支持。

---

# DeviceGuard

> Paddle 头文件：`ATen/DeviceGuard.h`

## 差异点列表

1.  **`device_of(tensor)` 的索引语义存在历史差异**：PyTorch 默认 CPU 设备常表现为 `index=-1`、`has_index=false`，Paddle 常规范化到 `cpu:0`。当前测试已避免直接序列化 `index/has_index` 字段。

---

## Diff 测试用例位置

测试文件：`test/DeviceGuardTest.cpp`

### 测试用例原文

```cpp
static void write_device_result_to_file(FileManerger* file,
                                        const std::optional<at::Device>& dev) {
  if (dev.has_value()) {
    // [DIFF] index/has_index 在两端语义不同，当前仅比较 type
    *file << dev->type() << " ";
  } else {
    *file << "nullopt ";
  }
}
```

---

## 输出对比

| 测试用例 | Paddle 输出 | Torch 输出 |
|---------|------------|------------|
| DeviceOfTensor | 仅比较 `type` 字段 | 仅比较 `type` 字段 |

---

## 初步问题分析

历史上该差异存在于 `index/has_index` 字段；当前测试策略已切换为只比较 `device.type()`，避免将设备表示差异放大为回归失败。

---

# Equal

> Paddle 头文件：`ATen/ops/equal.h`

## 差异点列表

1. **数据类型不同时的比对行为**：Torch在比对类型不一致的Tensor时会静默返回false，不触发任何错误；而Paddle在尝试比对时会在底层抛出类型检查不匹配（例如要求int32但接收到了float32）的C++异常甚至崩溃。

---

## Diff 测试用例位置

测试文件：`test/ops/EqualTest.cpp`

### 测试用例原文

```cpp
// [DIFF] Test paddle equal exception when comparing tensors of different types
// Torch returns false without checking specific data types, whereas Paddle throws:
// "The type of data we are trying to retrieve (int32) does not match the type of data (float32)..."
TEST_F(EqualTest, NotEqualDtype) {
  /*
  at::Tensor t1 = at::zeros({4}, at::kFloat);
  at::Tensor t2 = at::zeros({4}, at::kInt);

  bool result = t1.equal(t2);

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  write_bool_result_to_file(&file, result);
  file.saveFile();
  */
}
```

---

## 输出对比

| 测试用例 | Paddle 输出 | Torch 输出 |
|---------|------------|------------|
| NotEqualDtype | 历史观测：抛出异常 | 历史观测：`false` |

---

## 初步问题分析

该差异是历史实测结论；当前 case 已禁用，暂不参与常规回归对比。

---

# Select

> Paddle 头文件：`ATen/ops/select.h`

## 差异点列表

1. **支持负数维度的表现**：Torch支持传入负数维（如-1代表最后一维）进行选取；而Paddle在使用 -1 时可能会引发底层的 double free or corruption (out) 崩溃引发SIGABRT。

---

## Diff 测试用例位置

测试文件：`test/ops/SelectTest.cpp`

### 测试用例原文

```cpp
// [DIFF] Paddle select with negative dim causes double free or corruption SIGABRT
TEST_F(SelectTest, SelectNegativeDim) {
  /*
  at::Tensor t1 = at::zeros({3, 3}, at::kFloat);
  float* data = t1.data_ptr<float>();
  for (int i = 0; i < 9; ++i) {
    data[i] = static_cast<float>(i);
  }

  at::Tensor result = t1.select(-1, 0);

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  write_result_to_file(&file, result);
  file.saveFile();
  */
}
```

---

## 输出对比

| 测试用例 | Paddle 输出 | Torch 输出 |
|---------|------------|------------|
| SelectNegativeDim | 历史观测：崩溃 (SIGABRT) | 历史观测：正常返回 Tensor |

---

## 初步问题分析

该差异是历史复现结论；当前为保持回归稳定性已禁用该 case。

---

# Tensor 指针 API（`const_data_ptr<T>` / `mutable_data_ptr<T>`）

> Paddle 头文件：`ATen/core/TensorBody.h`

## 差异点列表

1. **模板版本的指针接口链接失败**：`tensor.const_data_ptr<float>()` 和 `tensor.mutable_data_ptr<float>()` 在 Paddle 侧出现 `undefined reference`。

---

## Diff 测试用例位置

测试文件：`test/ops/TensorPtrTest.cpp`

### 测试用例原文

```cpp
TEST(TensorBodyTest, PtrTest) {
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCPU);
  at::Tensor t = at::ones({2, 3}, options);

  // [DIFF] // const float* const_ptr = t.const_data_ptr<float>();
//   EXPECT_NE(const_ptr, nullptr);

  const void* void_const_ptr = t.const_data_ptr();
  EXPECT_NE(void_const_ptr, nullptr);

  // [DIFF] // float* mut_ptr = t.mutable_data_ptr<float>();
//   EXPECT_NE(mut_ptr, nullptr);

  void* void_mut_ptr = t.mutable_data_ptr();
  EXPECT_NE(void_mut_ptr, nullptr);

  auto file_name = g_custom_param.get();
  paddle_api_test::FileManerger file(file_name);
  file.openAppend();
  file.saveFile();
}
```

---

## 输出对比

| 测试用例 | Paddle 输出 | Torch 输出 |
|---------|------------|------------|
| PtrTest (`const_data_ptr<float>`) | 历史观测：链接报错（`undefined reference`） | 历史观测：正常返回指针 |
| PtrTest (`mutable_data_ptr<float>`) | 历史观测：链接报错（`undefined reference`） | 历史观测：正常返回指针 |

---

## 初步问题分析

Paddle 兼容层在 `ATen/core/TensorBody.h` 中声明了模板方法，但未提供对应定义或显式实例化；而 Torch 侧该模板接口可完整链接。

---

## 修复方向

在 Paddle compat 中补齐 `Tensor::const_data_ptr<T>()` 与 `Tensor::mutable_data_ptr<T>()` 的模板定义（或显式实例化），并与 `TensorBase` 的实现保持一致。

---

# Storage（`isSharedStorageAlias`）

> Paddle 头文件：`c10/core/Storage.h`

## 差异点列表

1. **共享存储别名判定口径不一致**：在 `slice` 场景下，Paddle 返回共享别名，Torch 返回非共享别名。

---

## Diff 测试用例位置

测试文件：`test/StorageTest.cpp`

测试用例：`StorageTest.StorageSetDataPtrNoswapAndTraitsProbe`

---

## 输出对比

| 测试用例 | Paddle 输出 | Torch 输出 |
|---------|------------|------------|
| StorageSetDataPtrNoswapAndTraitsProbe（原 `isSharedStorageAlias` 位置） | `1 0` | `0 0` |

---

## 初步问题分析

两端对“共享别名”的判定标准存在实现差异：在 `tensor.slice(...)` 产生的共享存储场景中，Paddle 判定为 true，Torch 判定为 false。

---

# DefaultDtype（`get_default_complex_dtype`）

> Paddle 头文件：`c10/core/DefaultDtype.h`

## 差异点列表

1. **默认复数类型不一致**：PyTorch 默认 `ComplexFloat`（枚举值 `9`），Paddle 默认 `ComplexDouble`（枚举值 `8`）。

---

## Diff 测试用例位置

测试文件：`test/DefaultDtypeTest.cpp`

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

测试文件：`test/DeviceTest.cpp`

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

测试文件：`test/DeviceTest.cpp`

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

# Empty（`at::empty` CUDA 场景）

> Paddle 头文件：`ATen/ops/empty.h`

## 差异点列表

1. **CUDA 结果受运行环境影响**：当前环境中 Paddle compat 可创建 CUDA Tensor，而 Torch 侧进入 `cuda_not_available` 分支。

---

## Diff 测试用例位置

测试文件：`test/EmptyOpsTest.cpp`

### 测试用例原文

```cpp
TEST_F(EmptyOpsTest, EmptyCUDA) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();

  // Try to create empty CUDA tensor
  try {
    at::Tensor t = at::empty({2, 3}, at::TensorOptions().device(at::kCUDA));
    // [DIFF] 当前环境下 Paddle compat 可成功创建 CUDA Tensor，而 Torch 侧进入不可用分支。
    // [DIFF] 该差异受运行时/构建环境影响，不属于 empty 接口的稳定语义差异，因此不参与结果序列化。
    (void)t;
  } catch (...) {
  }
  file.saveFile();
}
```

---

## 输出对比

| 测试用例 | Paddle 输出 | Torch 输出 |
|---------|------------|------------|
| EmptyCUDA（原始） | `cuda_tensor` | `cuda_not_available` |

---

## 初步问题分析

该差异与二进制构建方式和运行环境（CUDA 可用性）强相关，不属于 `at::empty` 的稳定接口语义差异。测试已保留调用路径，但不再序列化该字段以避免伪 diff。

---
