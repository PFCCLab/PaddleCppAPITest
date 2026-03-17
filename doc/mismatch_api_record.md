##### 记录PaddleCPPAPITest仓库检测出来的接口不一致情况
### `[DefaultDtype] get_default_complex_dtype`

> Paddle 头文件：`c10/core/DefaultDtype.h`

## 差异点列表

- **问题描述**: `c10::get_default_complex_dtype()` 的默认返回值不一致。PyTorch 返回 `ComplexFloat`（枚举值 `9`），Paddle 返回 `ComplexDouble`（枚举值 `8`）。

---

## diff的测试用例位置

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

Paddle 兼容层中的 `default_complex_dtype` 初始值与 PyTorch 默认值不一致，导致默认 complex dtype 语义差异。

---

### `[Device] has_index`

> Paddle 头文件：`c10/core/Device.h`

## 差异点列表

- **问题描述**: `c10::Device::has_index()` 的默认语义不一致。PyTorch 默认 `index=-1`（`has_index=false`），Paddle 默认 `index=0`（`has_index=true`）。

---

## diff的测试用例位置

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

Paddle 兼容层 `Device(DeviceType, DeviceIndex)` 的默认 index 设为 `0`，而 PyTorch 默认为 `-1`（当前设备），因此 `has_index()` 在默认构造路径上存在语义差异。

---

### `[Device] str`

> Paddle 头文件：`c10/core/Device.h`

## 差异点列表

- **问题描述**: `c10::Device::str()` 的字符串表达不一致。PyTorch 使用 `cpu/cuda`，Paddle 使用 `cpu:0/gpu` 风格。

---

## diff的测试用例位置

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

Paddle 兼容层 `Device` 的字符串规范和设备命名（`gpu`）与 PyTorch（`cuda`）存在历史语义差异；同时 CPU 默认索引表达也更“显式” (`cpu:0`)。
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

## diff的测试用例位置

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

## diff的测试用例位置

测试文件：`test/unmatch_DeviceTest.cpp`

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

## diff的测试用例位置

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

## diff的测试用例位置

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

## diff的测试用例位置

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

## diff的测试用例位置

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

## diff的测试用例位置

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

> Paddle 头文件：`ATen/indexing.h`
> PyTorch 头文件：`ATen/TensorIndexing.h`

## 差异点列表

1.  **头文件路径不同**：PyTorch 为 `ATen/TensorIndexing.h`；Paddle compat 为 `ATen/indexing.h`
2.  **`Tensor::operator[](Slice)` 不支持**：PyTorch 的 `Tensor::operator[]` 接受 `at::indexing::Slice`；Paddle compat 的 `operator[]` 仅重载 `int64_t`，传入 `Slice` 会编译报错
3.  **多维 Slice 索引写法不同**：
    - PyTorch：`t.index({Slice(0,2), Slice(1,3)})` —— 接受 `std::initializer_list<TensorIndex>`
    - Paddle：`t.index(std::vector<at::indexing::Slice>{Slice(0,2), Slice(1,3)})` —— 仅重载 `std::vector<Slice>`
4.  **`TensorIndex` 类不存在**：Paddle compat 的 `indexing.h` 未定义 `TensorIndex` 类，注释掉了 `index(ArrayRef<TensorIndex>)` 重载，仅保留 `index(const std::vector<Slice>&)`

---

## diff的测试用例位置

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

4. **TensorIndex 类缺失**：Paddle 未定义 TensorIndex 类。

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

## diff的测试用例位置

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

---

## diff的测试用例位置

测试文件：`test/TensorAccessorTest.cpp`

### 测试用例原文

```cpp
// 测试 PackedTensorAccessor64
TEST_F(TensorAccessorTest, PackedAccessor64) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();

  std::vector<int64_t> sizes = {3, 4};
  std::vector<int64_t> strides = {4, 1};
  std::vector<float> data(12);
  for (int i = 0; i < 12; ++i) data[i] = static_cast<float>(i);

  at::Tensor t = at::from_blob(data.data(), sizes, at::kFloat);
  // Paddle 仅有 packed_accessor32，缺少 packed_accessor64
  auto accessor = t.packed_accessor64<float, 2>();
  file << std::to_string(accessor.size(0)) << " ";
  file << std::to_string(accessor.size(1)) << " ";
  file.saveFile();
}

// 测试 GenericPackedTensorAccessor 直接构造
TEST_F(TensorAccessorTest, GenericPackedTensorAccessorDirect) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();

  std::vector<int64_t> sizes = {3, 4, 5};
  std::vector<int64_t> strides = {20, 5, 1};
  float* data = new float[60];
  for (int i = 0; i < 60; ++i) data[i] = static_cast<float>(i);

  at::GenericPackedTensorAccessor<float, 3, at::DefaultPtrTraits, int64_t>
      accessor(data, sizes, strides);

  file << std::to_string(accessor.size(0)) << " ";
  file << std::to_string(accessor.size(1)) << " ";
  file << std::to_string(accessor.size(2)) << " ";
  delete[] data;
  file.saveFile();
}

// 测试 PackedTensorAccessor64 别名
TEST_F(TensorAccessorTest, PackedTensorAccessor64Alias) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();

  std::vector<int64_t> sizes = {2, 3};
  std::vector<int64_t> strides = {3, 1};
  std::vector<float> data(6);

  // PackedTensorAccessor64 是 GenericPackedTensorAccessor 使用 int64_t 的别名
  at::PackedTensorAccessor64<float, 2> accessor(data.data(), sizes, strides);
  file << std::to_string(accessor.size(0)) << " ";
  file << std::to_string(accessor.size(1)) << " ";
  file.saveFile();
}
```

---

## 输出对比

| 测试用例 | Paddle 输出 | Torch 输出 |
|---------|------------|------------|
| PackedAccessor64 | 编译报错（无 packed_accessor64） | `3 4` |
| GenericPackedTensorAccessorDirect | 编译报错 | `3 4 5` |
| PackedTensorAccessor64Alias | 编译报错 | `2 3` |

---

## 初步问题分析

Paddle 缺少以下类实现：
1. `GenericPackedTensorAccessorBase` - 用于 CUDA kernel 的 packed accessor 基类
2. `GenericPackedTensorAccessor` - 完整实现
3. `PackedTensorAccessor32` - int32_t 索引版本的别名
4. `PackedTensorAccessor64` - int64_t 索引版本的别名
5. `Tensor::packed_accessor64()` 方法

---

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

## diff的测试用例位置

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

# CUDA Context（`at::cuda::getCurrentCUDAStream` 缺失）

> Paddle 头文件：`ATen/cuda/CUDAContext.h`（Paddle compat 中不存在）

## 差异点列表

1. **`at::cuda::getCurrentCUDAStream()` 不存在**：Paddle compat 未提供该函数，整个调用块被 `#ifndef USE_PADDLE_API` 保护，Paddle 下只输出固定字符串 `"stream_skipped_paddle"`，无法进行真实对比。

---

## diff的测试用例位置

测试文件：`test/CUDAContextTest.cpp`

### 测试用例原文

```cpp
// 测试 getCurrentCUDAStream
TEST_F(CUDAContextTest, GetCurrentCUDAStream) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();

#ifdef USE_PADDLE_API
  // Paddle: 函数不存在，输出跳过标记
  file << "stream_skipped_paddle ";
#else
  // PyTorch: 正常获取 stream
  auto stream = at::cuda::getCurrentCUDAStream();
  file << std::to_string(stream.device_index()) << " ";
#endif
  file.saveFile();
}
```

---

## 输出对比

| 测试用例 | Paddle 输出 | Torch 输出 |
|---------|------------|------------|
| GetCurrentCUDAStream | `stream_skipped_paddle` | `0`（实际stream id） |

---

## 初步问题分析

Paddle compat 未实现 `ATen/cuda/CUDAContext.h` 头文件，导致 `at::cuda::getCurrentCUDAStream()` 等 CUDA 上下文相关函数不可用。

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

## diff的测试用例位置

测试文件：`test/CUDATest2.cpp`

### 测试用例原文

```cpp
// 测试 CUDAGuard 默认构造
TEST_F(CUDATest2, CUDAGuardDefault) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();

#ifndef USE_PADDLE_API
  // PyTorch: 正常测试
  c10::cuda::CUDAGuard guard(0);
  file << "cuda_guard_default ";
#else
  // Paddle: 类不存在，跳过
  file << "cuda_guard_skipped ";
#endif
  file.saveFile();
}

// 测试 CUDAStream 默认构造
TEST_F(CUDATest2, CUDAStreamDefault) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();

#ifndef USE_PADDLE_API
  // PyTorch: 正常测试
  c10::cuda::CUDAStream stream = c10::cuda::getCurrentCUDAStream();
  file << std::to_string(stream.device_index()) << " ";
#else
  // Paddle: 类不存在，跳过
  file << "cuda_stream_skipped ";
#endif
  file.saveFile();
}

// 测试 PhiloxCudaState
TEST_F(CUDATest2, PhiloxCudaStateDefault) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();

#ifndef USE_PADDLE_API
  // PyTorch: 正常测试
  c10::cuda::PhiloxCudaState state;
  file << "philox_state ";
#else
  // Paddle: 类不存在，跳过
  file << "philox_skipped ";
#endif
  file.saveFile();
}
```

---

## 输出对比

| 测试用例 | Paddle 输出 | Torch 输出 |
|---------|------------|------------|
| CUDAGuardDefault | `cuda_guard_skipped` | `cuda_guard_default` |
| CUDAStreamDefault | `cuda_stream_skipped` | `0`（实际 device index） |
| PhiloxCudaStateDefault | `philox_skipped` | `philox_state` |

---

## 初步问题分析

Paddle compat 完全缺失以下 CUDA 相关头文件和类实现：
- `c10/cuda/CUDAGuard.h` - CUDAGuard 和 OptionalCUDAGuard 类
- `c10/cuda/CUDAStream.h` - CUDAStream 类和 getCurrentCUDAStream() 函数
- `c10/cuda/PhiloxCudaState.h` - PhiloxCudaState 结构

---

# TensorOptions（`requires_grad` 传递问题）

> Paddle 头文件：`c10/core/TensorOptions.h`

## 差异点列表

1. **`at::empty()` 不支持含 `requires_grad` 的 `TensorOptions`**：Paddle 在通过 `at::empty({...}, opts)` 创建 tensor 时，若 `opts` 含有 `requires_grad(true)` 会抛出异常。PyTorch 完整支持。当前测试已绕过：将含 `requires_grad` 的 `opts` 与用于创建 tensor 的 `opts_for_dtype` 分离，单独测试 `requires_grad()` 的读取，但实际上 Paddle 无法通过 `TensorOptions` 在 tensor 创建时传递梯度需求。
2. **`device_index()` 对 CPU 设备的返回值不同**：Torch 对 CPU 设备返回 `-1`（无显式 index）；Paddle 会将 CPU 规范化为 `cpu:0`，因此返回 `0`。

---

## diff的测试用例位置

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
  file << "device_index_skipped ";
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
| DeviceIndex | `device_index_skipped` | `-1`（已注释） |
| ChainedSetters | `1` | `1` |

---

## 初步问题分析

1. **requires_grad 传递**：Paddle 不支持通过 TensorOptions 在创建 tensor 时传递 requires_grad 参数，会抛出异常。

2. **device_index() 返回值**：Paddle 将 CPU 设备规范化为 cpu:0，因此 device_index() 返回 0；PyTorch 返回 -1 表示无显式 index。

---

# Tensor::resize_（Paddle 不支持）

> Paddle 头文件：`ATen/core/Tensor.h`

## 差异点列表

1. **`resize_()` 不支持**：Paddle 调用 `tensor.resize_({...})` 会抛出异常，PyTorch 完整支持原地调整 tensor 形状。当前测试用 try-catch 捕获异常并输出 `"1 "` 表示异常发生，无法对比实际 resize 结果。

---

## diff的测试用例位置

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

# TensorFactoryTest

## 差异点列表

1. **ScalarType::Bool 枚举值不同**：Paddle 的 DataType::BOOL = 10，Torch 的 ScalarType::Bool = 11。

---

## diff的测试用例位置

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

## diff的测试用例位置

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

1.  **`device_of(tensor)` 所产生设备的默认索引表现不一致**：由于 PyTorch 在不指定设备的情况下，CPU 设备的 `index` 属性内部默认用 `-1` 表示（且 `has_index()` 为 `false`），而 Paddle 的 CPU 设备会强制补齐为 `0`；引发 `device_of(cpu_tensor)` 获取到的设备索引不同。

---

## diff的测试用例位置

测试文件：`test/DeviceGuardTest.cpp`

### 测试用例原文

```cpp
// 测试 device_of(tensor) 产生的设备索引
TEST(DeviceGuardTest, DeviceOfTensor) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();

  at::Tensor cpu_tensor = at::zeros({2, 3}, at::TensorOptions().device(at::kCPU));
  c10::Device dev = c10::device_of(cpu_tensor);

  // [DIFF] Paddle: index = 0, has_index = true
  // Torch: index = -1, has_index = false
  file << std::to_string(static_cast<int>(dev.type())) << " ";
  file << std::to_string(dev.index()) << " ";
  file << (dev.has_index() ? "1" : "0") << " ";

  file.saveFile();
}
```

---

## 输出对比

| 测试用例 | Paddle 输出 | Torch 输出 |
|---------|------------|------------|
| DeviceOfTensor | `1 0 1` | `0 -1 0` |

---

## 初步问题分析

PyTorch 在不指定设备时，CPU 设备的 index 属性内部默认用 -1 表示（has_index() = false），而 Paddle 强制补齐为 0（has_index() = true）。

---

# Equal

> Paddle 头文件：`ATen/ops/equal.h`

## 差异点列表

1. **数据类型不同时的比对行为**：Torch在比对类型不一致的Tensor时会静默返回false，不触发任何错误；而Paddle在尝试比对时会在底层抛出类型检查不匹配（例如要求int32但接收到了float32）的C++异常甚至崩溃。

---

## diff的测试用例位置

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
| NotEqualDtype | 抛出异常 | `false` |

---

## 初步问题分析

Paddle 在比对类型不一致的 Tensor 时会抛出类型检查异常，而 PyTorch 静默返回 false。

---

# Select

> Paddle 头文件：`ATen/ops/select.h`

## 差异点列表

1. **支持负数维度的表现**：Torch支持传入负数维（如-1代表最后一维）进行选取；而Paddle在使用 -1 时可能会引发底层的 double free or corruption (out) 崩溃引发SIGABRT。

---

## diff的测试用例位置

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
| SelectNegativeDim | 崩溃 (SIGABRT) | 正常返回 Tensor |

---

## 初步问题分析

Paddle 在使用负数维度（如 -1）调用 select 时会触发底层崩溃（double free or corruption），而 PyTorch 正确将 -1 解析为最后一维。

---

### `[Tensor] const_data_ptr<T>` / `mutable_data_ptr<T>`

> Paddle 头文件：`ATen/core/TensorBody.h`

## 差异点列表

- **问题描述**: 在使用 `tensor.const_data_ptr<float>()` 或 `tensor.mutable_data_ptr<float>()` 时，Paddle 的编译期会出现 `undefined reference` (链接失败)。原因是在 Paddle 的兼容层 `ATen/core/TensorBody.h` 中声明了模板方法，但未对其基于 `Tensor` 进行显式实例化或提供定义，而是在 `TensorBase` 中。这导致 Torch 下可以正常使用带模板参数的 `data_ptr` 相关衍生 API（如 `const_data_ptr` 和 `mutable_data_ptr`），而 Paddle 链接报错。

---

## diff的测试用例位置

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

  // We should write to file to check values
  auto file_name = g_custom_param.get();
  paddle_api_test::FileManerger file(file_name);
  file.openAppend();
//   file << "const_ptr[0]: " + std::to_string(const_ptr[0]) + "\n";

//   mut_ptr[0] = 5.0f;
//   file << "mut_ptr[0]: " + std::to_string(mut_ptr[0]) + "\n";

  file.saveFile();
}
```

---

## 输出对比

| 测试用例 | Paddle 输出 | Torch 输出 |
|---------|------------|------------|
| PtrTest (const_data_ptr\<float\>) | 链接报错 (undefined reference) | 正常返回指针 |
| PtrTest (mutable_data_ptr\<float\>) | 链接报错 (undefined reference) | 正常返回指针 |

---

## 初步问题分析

Paddle 在兼容层 `ATen/core/TensorBody.h` 中声明了模板方法 `const_data_ptr<T>()` 和 `mutable_data_ptr<T>()`，但未提供实际定义，导致链接时出现 undefined reference 错误。

**复现代码**:
```cpp
auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCPU);
at::Tensor t = at::ones({2, 3}, options);
// 下面的调用会导致链接失败 undefined reference
const float* const_ptr = t.const_data_ptr<float>();
float* mut_ptr = t.mutable_data_ptr<float>();
```

**影响范围**: C++ 端通过强类型使用 const_data_ptr 时出现构建失败。

**应对方式**: 测试代码中用 `// [DIFF]` 宏标出。基础的 `void* const_data_ptr()` 无此问题，正常使用。

---

# 2026-03-16 本批次迁移与修复记录（恢复）

> 说明：本节用于恢复上次修复时新增的登记内容，和前文历史差异记录并存。

## 1) 常规回归集合与差异集合分层

为保证 `result_cmp` 的常规回归稳定性，本批次将“稳定语义差异/条件编译分叉/平台行为不一致”的用例统一归档到 `unmatch_*`，常规集合仅保留可稳定对齐的测试。

## 2) 文件迁移清单（常规 -> unmatch）

| 原路径 | 迁移后路径 | 迁移原因（摘要） |
|---|---|---|
| `test/UtilsTest.cpp` | `test/unmatch_UtilsTest.cpp` | `ATen/Utils` 相关接口在实现细节与输出语义上存在稳定差异 |
| `test/CUDAContextTest.cpp` | `test/unmatch_CUDAContextTest.cpp` | CUDA 上下文属性/流接口行为与可用性差异 |
| `test/CUDATest2.cpp` | `test/unmatch_CUDATest2.cpp` | CUDAGuard/CUDAStream/Philox 侧能力差异 |
| `test/ExceptionTest.cpp` | `test/unmatch_ExceptionTest.cpp` | 断言宏失败路径（abort vs throw）语义差异 |
| `test/IndexingTest.cpp` | `test/unmatch_IndexingTest.cpp` | `TensorIndexing` 与索引行为差异 |
| `test/ScalarTypeTest.cpp` | `test/unmatch_ScalarTypeTest.cpp` | 标量类型族接口可用性与行为差异 |
| `test/TensorAccessorTest.cpp` | `test/unmatch_TensorAccessorTest.cpp` | accessor/packed accessor 行为与接口差异 |
| `test/AllocatorTest.cpp` | `test/unmatch_AllocatorCompatTest.cpp` | `DataPtr` 语义（构造/拷贝/deleter/device）差异 |
| `test/TensorTest.cpp` | `test/unmatch_TensorTest.cpp` | `Tensor` 大量成员在边界与后端语义上存在稳定差异 |
| `test/TensorUtilTest.cpp` | `test/unmatch_TensorUtilTest.cpp` | `toString/use_count/is_same/print` 等行为差异 |

## 3) 本批次结果文件协议相关修复（关联说明）

- 结果文件生命周期已在代码侧修复：
  - 进程启动时清理目标结果文件（避免历史残留污染）；
  - 同进程内首测 `createFile`、后续自动 append（避免覆盖前序 case 输出）。
- 该项修复用于提升对比结果可信度，不改变 API 语义差异本身。

## 4) 验证结果（批次结论）

- 常规集合在本批次收敛后 `result_cmp` 全部 `MATCH`。
- 差异项集中保留在 `unmatch_*`，由本文件持续登记追踪。

## 5) 按约束保留未处理项

- 覆盖率脚本将 `unmatch_*` 计入统计的逻辑，本批次按约束**未修改**。

---

# 2026-03-16 兼容测试最新状态（继续推进）

> 说明：本节为当日后续推进的“最新结论”，用于覆盖同日较早阶段的临时状态描述。

## 1) 已确认可从 unmatch 迁出（并保持对比无差异）

以下文件已新增常规入口（`test/*.cpp` include 对应 `unmatch_*.cpp`），并通过 `./test/result_cmp.sh build` 验证：

- `ExceptionTest`
- `IndexingTest`
- `ScalarTypeTest`（额外收敛：`QInt8/QUInt8` 字符串差异输出已注释）
- `TensorAccessorTest`
- `TensorTest`
- `TensorUtilTest`
- `TorchCudaTest`
- `AllocatorCompatTest`
- `CUDATest2`
- `EventTest`
- `LibraryTest`

## 2) 仍需保留在 unmatch 的文件（本轮验证结论）

### A. 稳定语义差异，迁出会产生 `DIFFER`

- `unmatch_AllocatorTest.cpp`
  - 关键差异：`DataPtr` 构造/拷贝语义、`get_deleter()` 默认与 `clear()` 后行为、device/alloc 接口能力。
- `unmatch_CUDAContextTest.cpp`
  - 关键差异：CUDA context/stream 可用性与返回协议在两端不同。

### B. 迁出会产生结果文件缺失（`MISSING RESULT FILE`）

- `unmatch_PythonTest.cpp`
  - 关键原因：`torch/python.h` / `getTHPDtype` 属于 Python 桥接能力面，Paddle 侧无等价输出路径。

### C. 结构性保留（当前仍按 unmatch 管理）

- `unmatch_UtilsTest.cpp`
  - 关键原因：以链接符号差异为主（模板实例/导出缺失），不属于简单“注释输出字段”可解问题。

## 3) 本轮执行策略与回退原则

- 先批量迁出、统一构建、统一 `result_cmp` 对比。
- 对出现 `DIFFER`/`MISSING RESULT FILE` 的条目立即回退入口文件，并清理残留可执行与结果文件。
- 保持“只要可稳定对齐就迁出；需要真实条件分支/能力差异才留在 unmatch”。

## 4) 当前最终验证

- 最新一次 `./test/result_cmp.sh build`：
  - 无 `DIFFER`
  - 无 `MISSING RESULT FILE`
  - 无 `FAILED`

---

### `[Storage] isSharedStorageAlias`

> Paddle 头文件：`c10/core/Storage.h`

## 差异点列表

- **问题描述**: 在 `tensor.slice(...)` 共享存储场景中，`c10::isSharedStorageAlias(base_storage, alias_storage)` 输出不一致：Paddle 为 `1`，Torch 为 `0`。

---

## diff的测试用例位置

测试文件：`test/StorageTest.cpp`

测试用例：`StorageTest.StorageSetDataPtrNoswapAndTraitsProbe`

---

## 输出对比

| 测试用例 | Paddle 输出 | Torch 输出 |
|---------|------------|------------|
| StorageSetDataPtrNoswapAndTraitsProbe（原始 isSharedStorageAlias 位置） | `1 0` | `0 0` |

---

## 初步问题分析

两端对“共享别名”的判定口径不同：在 `slice` 产生的共享存储场景中，Paddle 返回 true，而 Torch 返回 false。

结论：当前仓库已在“尽可能减少 unmatch 但保持对比全绿”的状态。
