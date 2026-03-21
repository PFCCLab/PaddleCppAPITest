##### CUDAStream.h 头文件 API 兼容情况


##### CUDAStream.h 头文件 API 兼容性

✅ 表示已经支持
🚧 表示正在支持
❌ 表示不准备支持
🔧 表示部分支持（有功能限制）

**按照功能分类排序**

---

### 标签类型

| torch API                         | paddle API 兼容性 | 测试用例状态 | 优先级 | 备注 |
|-----------------------------------|------------------|------------|-------|------|
| `CUDAStream::UNCHECKED` (enum)    | ❌               | ❌          |   P2  | 无检查构造标签，Paddle 未提供 |

---

### 构造函数

| torch API                                     | paddle API 兼容性 | 测试用例状态 | 优先级 | 备注 |
|-----------------------------------------------|------------------|------------|-------|------|
| `CUDAStream(Stream stream)`                   | ✅               | - [ ]       |   P0  | 有检查构造，断言设备类型为 CUDA |
| `CUDAStream(Unchecked, Stream stream)`        | ❌               | ❌          |   P2  | 无检查构造，Paddle 未提供 |

---

### 比较运算符

| torch API                                     | paddle API 兼容性 | 测试用例状态 | 优先级 | 备注 |
|-----------------------------------------------|------------------|------------|-------|------|
| `operator==(const CUDAStream&)`               | ✅               | - [ ]       |   P1  | 相等比较 |
| `operator!=(const CUDAStream&)`               | ✅               | - [ ]       |   P1  | 不等比较 |

---

### 隐式转换运算符

| torch API                    | paddle API 兼容性 | 测试用例状态 | 优先级 | 备注 |
|------------------------------|------------------|------------|-------|------|
| `operator cudaStream_t()`    | ✅               | - [ ]       |   P0  | 隐式转换到 `cudaStream_t` |
| `operator Stream()`          | ✅               | - [ ]       |   P0  | 隐式转换到 `c10::Stream` |

---

### 设备信息 API

| torch API          | paddle API 兼容性 | 测试用例状态 | 优先级 | 备注 |
|--------------------|------------------|------------|-------|------|
| `device_type()`    | ✅               | - [ ]       |   P0  | 返回 `DeviceType::CUDA` |
| `device_index()`   | ✅               | - [ ]       |   P1  | 获取 CUDA 设备索引 |
| `device()`         | ✅               | - [ ]       |   P1  | 获取完整 `Device` 对象 |

---

### 流标识 API

| torch API  | paddle API 兼容性 | 测试用例状态 | 优先级 | 备注 |
|------------|------------------|------------|-------|------|
| `id()`     | ✅               | - [ ]       |   P0  | 返回 `StreamId`（`int64_t`） |
| `stream()` | ✅               | - [ ]       |   P0  | 显式转换到 `cudaStream_t` |
| `unwrap()` | ✅               | - [ ]       |   P0  | 显式转换到 `c10::Stream` |

---

### 同步与状态查询 API

| torch API        | paddle API 兼容性 | 测试用例状态 | 优先级 | 备注 |
|------------------|------------------|------------|-------|------|
| `query()`        | ✅               | - [x]       |   P1  | 通过 `unwrap()` 委托到 `c10::Stream::query()`，在 `c10/core/Stream.cpp` 中实现；CPU stream 始终返回 true |
| `synchronize()`  | ✅               | - [x]       |   P1  | 通过 `unwrap()` 委托到 `c10::Stream::synchronize()`，在 `c10/core/Stream.cpp` 中实现 |

---

### 优先级 API

| torch API                                                               | paddle API 兼容性 | 测试用例状态 | 优先级 | 备注 |
|-------------------------------------------------------------------------|------------------|------------|-------|------|
| `priority()`                                                            | ❌               | ❌          |   P2  | 获取流的优先级 |
| `priority_range()` (static)                                             | ❌               | ❌          |   P2  | 获取 PyTorch 支持的优先级范围 |

---

### 序列化 API

| torch API                                                               | paddle API 兼容性 | 测试用例状态 | 优先级 | 备注 |
|-------------------------------------------------------------------------|------------------|------------|-------|------|
| `pack3()`                                                               | ❌               | ❌          |   P3  | 打包为 `c10::StreamData3` |
| `unpack3(StreamId, DeviceIndex, DeviceType)` (static)                  | ❌               | ❌          |   P3  | 从三字段解包构造 `CUDAStream` |

---

### 全局流管理函数

| torch API                                              | paddle API 兼容性 | 测试用例状态 | 优先级 | 备注 |
|--------------------------------------------------------|------------------|------------|-------|------|
| `getCurrentCUDAStream(DeviceIndex)`                    | ✅               | - [x]       |   P0  | 返回 per-thread per-device 的当前流（TLS），未设置时回退到 Paddle 默认流 |
| `setCurrentCUDAStream(CUDAStream)`                     | ✅               | - [x]       |   P0  | 设置 per-thread per-device 当前流，仅修改线程本地状态（TLS），不影响其他线程或 GPUContext |
| `getDefaultCUDAStream(DeviceIndex)`                    | ✅               | - [x]       |   P1  | 返回设备固定默认流（null stream，`cudaStreamDefault`，id=0），独立于 TLS current stream |
| `getStreamFromPool(bool isHighPriority, DeviceIndex)`  | ✅               | - [x]       |   P1  | 真实流池实现：每设备 32 条低/高优先级流，懒初始化（`std::call_once`），round-robin 原子计数器分配 |
| `getStreamFromPool(int priority, DeviceIndex)`         | ❌               | ❌          |   P2  | 按数值优先级从池中获取流，Paddle 未提供 |
| `getStreamFromExternal(cudaStream_t, DeviceIndex)`     | ❌               | ❌          |   P1  | 从外部已分配流创建 `CUDAStream`，Paddle 未提供 |

---

### 输出运算符

| torch API                             | paddle API 兼容性 | 测试用例状态 | 优先级 | 备注 |
|---------------------------------------|------------------|------------|-------|------|
| `operator<<(ostream&, CUDAStream)`    | ❌               | ❌          |   P3  | 流的文本输出 |

---

### 标准库特化

| torch API                    | paddle API 兼容性 | 测试用例状态 | 优先级 | 备注 |
|------------------------------|------------------|------------|-------|------|
| `std::hash<CUDAStream>`      | ❌               | ❌          |   P2  | 哈希支持，用于 `unordered_map/set` |

---

### 兼容性统计

| 状态 | 数量 |
|------|------|
| ✅ 已完全支持 | 17 |
| 🚧 正在支持 | 0 |
| 🔧 部分支持 | 0 |
| ❌ 未实现 | 9 |

---

### 备注

1. **优先级说明**：
   - P0: 核心功能，必须支持
   - P1: 常用功能，高优先级
   - P2: 进阶功能，中优先级
   - P3: 边缘功能，低优先级

2. **实现说明**：
   - Paddle 兼容层的 `CUDAStream` 内部持有 `c10::Stream`，`stream()` 通过 `reinterpret_cast<cudaStream_t>(id())` 还原裸 `cudaStream_t` 句柄
   - `device_index()` 返回 `stream_.device_index()`；`device()` 返回 `Device(CUDA, device_index())`
   - `query()` 和 `synchronize()` 通过 `c10::Stream::query()`/`c10::Stream::synchronize()` 实现，在 `c10/core/Stream.cpp` 中定义

3. **流语义区分**：
   - `getCurrentCUDAStream()`：返回 per-thread per-device 的当前流（TLS），调用 `setCurrentCUDAStream()` 后可变
   - `getDefaultCUDAStream()`：始终返回设备的 null stream（`cudaStreamDefault`，handle=0），不受 TLS 影响
   - `getStreamFromPool()`：从预分配流池 round-robin 取出辅助流，与 current stream 不同，适合跨流异步操作

4. **缺失的功能**：
   - `std::hash<CUDAStream>` 缺失导致无法直接用于 `unordered_map`/`unordered_set`（可通过 `std::hash<c10::Stream>` 间接支持）
   - `getStreamFromExternal()` 缺失，限制了与第三方库（如 cuDNN、NCCL）自定义流的互操作
