##### Stream.h 头文件 API 兼容情况


##### Stream.h 头文件 API 兼容性

✅ 表示已经支持
🚧 表示正在支持
❌ 表示不准备支持
🔧 表示部分支持（有功能限制）

**按照功能分类排序**

---

### 类型与结构

| torch API                    | paddle API 兼容性 | 测试用例状态 | 优先级 | 备注 |
|-----------------------------|------------------|------------|-------|------|
| `using StreamId = int64_t`  | ✅               | - [ ]       |   P0  | 类型别名一致 |
| `struct StreamData3`        | ✅               | - [ ]       |   P1  | 字段一致：`stream_id/device_index/device_type` |

---

### ABI 与导出宏

| torch API                    | paddle API 兼容性 | 测试用例状态 | 优先级 | 备注 |
|-----------------------------|------------------|------------|-------|------|
| `C10_API StreamData3`       | 🔧               | - [ ]       |   P2  | Paddle 版本未标注 `C10_API` 导出宏 |
| `C10_API Stream`            | 🔧               | - [ ]       |   P2  | 类型接口一致，但未显式使用 `C10_API` |
| `C10_API operator<<`        | 🔧               | - [ ]       |   P2  | 声明存在，但未标注 `C10_API` |

---

### 标签与构造

| torch API                                         | paddle API 兼容性 | 测试用例状态 | 优先级 | 备注 |
|--------------------------------------------------|------------------|------------|-------|------|
| `enum Unsafe { UNSAFE }`                         | ✅               | - [ ]       |   P0  | 一致 |
| `enum Default { DEFAULT }`                       | ✅               | - [ ]       |   P0  | 一致 |
| `Stream(Unsafe, Device, StreamId)`               | ✅               | - [ ]       |   P0  | 签名一致 |
| `Stream(Default, Device)`                        | ✅               | - [ ]       |   P0  | 默认流构造一致 |

---

### 比较运算符

| torch API                           | paddle API 兼容性 | 测试用例状态 | 优先级 | 备注 |
|------------------------------------|------------------|------------|-------|------|
| `operator==(const Stream&)`        | ✅               | - [ ]       |   P0  | 实现逻辑一致（设备+ID） |
| `operator!=(const Stream&)`        | ✅               | - [ ]       |   P0  | 一致 |

---

### 访问与元信息 API

| torch API               | paddle API 兼容性 | 测试用例状态 | 优先级 | 备注 |
|------------------------|------------------|------------|-------|------|
| `device()`             | ✅               | - [ ]       |   P0  | 一致 |
| `device_type()`        | ✅               | - [ ]       |   P0  | 一致 |
| `device_index()`       | ✅               | - [ ]       |   P0  | 一致 |
| `id()`                 | ✅               | - [ ]       |   P0  | 一致 |
| `native_handle()`      | ✅               | - [x]       |   P1  | 完整实现：在 `c10/core/Stream.cpp` 中定义，CUDA/HIP 返回 `cudaStream_t` 句柄，不支持的设备类型抛异常 |

---

### 同步与等待 API

| torch API                    | paddle API 兼容性 | 测试用例状态 | 优先级 | 备注 |
|-----------------------------|------------------|------------|-------|------|
| `wait(const T& event)`      | ✅               | - [ ]       |   P1  | 模板实现一致：调用 `event.block(*this)` |
| `query()`                   | ✅               | - [x]       |   P1  | 完整实现：在 `c10/core/Stream.cpp` 中定义，CUDA 调用 `cudaStreamQuery`，CPU 始终返回 true |
| `synchronize()`             | ✅               | - [x]       |   P1  | 完整实现：在 `c10/core/Stream.cpp` 中定义，CUDA 调用 `cudaStreamSynchronize`，CPU 为 no-op |

---

### 序列化与打包 API

| torch API                                       | paddle API 兼容性 | 测试用例状态 | 优先级 | 备注 |
|------------------------------------------------|------------------|------------|-------|------|
| `hash()`                                       | ✅               | - [ ]       |   P1  | 位打包策略一致 |
| `pack3()`                                      | ✅               | - [ ]       |   P1  | 返回结构一致 |
| `unpack3(StreamId, DeviceIndex, DeviceType)`  | ✅               | - [ ]       |   P1  | 逻辑一致；仅断言宏名不同（`TORCH_CHECK` vs `PD_CHECK`） |

---

### 标准输出与哈希特化

| torch API                          | paddle API 兼容性 | 测试用例状态 | 优先级 | 备注 |
|-----------------------------------|------------------|------------|-------|------|
| `operator<<(ostream&, Stream)`    | ✅               | - [ ]       |   P2  | 声明一致 |
| `std::hash<c10::Stream>`          | ✅               | - [ ]       |   P2  | 特化实现一致，基于 `hash()` |

---

### 兼容性统计

| 状态 | 数量 |
|------|------|
| ✅ 已完全支持 | 18 |
| 🚧 正在支持 | 0 |
| 🔧 部分支持 | 3 |
| ❌ 未实现 | 0 |

---

### 备注

1. **优先级说明**：
   - P0: 核心功能，必须支持
   - P1: 常用功能，高优先级
   - P2: 进阶功能，中优先级
   - P3: 边缘功能，低优先级

2. **对比范围说明**：
   - 本文档基于头文件声明及 `.cpp` 实现对比：
     - `paddle/phi/api/include/compat/c10/core/Stream.h`（声明）
     - `paddle/phi/api/include/compat/c10/core/Stream.cpp`（实现）
     - `/home/may/pytorch/c10/core/Stream.h` 及 `Stream.cpp`（PyTorch 参考）
   - `native_handle()`、`query()`、`synchronize()` 均在 `Stream.cpp` 中有完整实现，不再是仅声明。

3. **主要差异说明**：
   - `C10_API` 导出宏：PyTorch 在 `StreamData3`、`Stream`、`operator<<` 上使用 `C10_API`；Paddle 兼容头未显式标注。
   - 断言宏名不同：`unpack3()` 使用 `PD_CHECK` 替代 `TORCH_CHECK`，不影响接口调用方式。
