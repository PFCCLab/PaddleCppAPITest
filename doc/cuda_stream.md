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
| `operator==(const CUDAStream&)`               | ❌               | ❌          |   P1  | 相等比较 |
| `operator!=(const CUDAStream&)`               | ❌               | ❌          |   P1  | 不等比较 |

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
| `device_index()`   | ❌               | ❌          |   P1  | 获取 CUDA 设备索引，Paddle 未提供 |
| `device()`         | ❌               | ❌          |   P1  | 获取完整 `Device` 对象，Paddle 未提供 |

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
| `query()`        | ❌               | ❌          |   P1  | 查询流上所有操作是否已完成 |
| `synchronize()`  | ❌               | ❌          |   P1  | 阻塞等待流上所有操作完成 |

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
| `getCurrentCUDAStream(DeviceIndex)`                    | ✅               | - [ ]       |   P0  | 获取当前 CUDA 流 |
| `setCurrentCUDAStream(CUDAStream)`                     | ✅               | - [ ]       |   P0  | 设置当前 CUDA 流，更新 Paddle GPUContext |
| `getDefaultCUDAStream(DeviceIndex)`                    | 🔧              | - [ ]       |   P1  | 通过 `#define` 宏转发到 `getCurrentCUDAStream`，非独立实现 |
| `getStreamFromPool(bool isHighPriority, DeviceIndex)`  | 🔧              | - [ ]       |   P1  | 未实现真正的流池，直接返回当前流，忽略优先级参数 |
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
| ✅ 已完全支持 | 9 |
| 🚧 正在支持 | 0 |
| 🔧 部分支持 | 2 |
| ❌ 未实现 | 15 |

---

### 备注

1. **优先级说明**：
   - P0: 核心功能，必须支持
   - P1: 常用功能，高优先级
   - P2: 进阶功能，中优先级
   - P3: 边缘功能，低优先级

2. **实现说明**：
   - Paddle 兼容层的 `CUDAStream` 基于 `phi::CUDAStream`（`paddle/phi/core/cuda_stream.h`）实现
   - `stream()` 通过 `reinterpret_cast<cudaStream_t>(stream_.id())` 将 `StreamId` 还原为裸 `cudaStream_t` 句柄
   - `device_index()` 和 `device()` 未在兼容层暴露，可通过 `unwrap().device_index()` 间接访问

3. **部分支持说明**：
   - `getDefaultCUDAStream`：以 `#define getDefaultCUDAStream getCurrentCUDAStream` 宏实现，行为上等价但不具备独立的"默认流"语义
   - `getStreamFromPool(bool, DeviceIndex)`：实现中未维护真正的流池，始终返回当前活跃流；`isHighPriority` 参数被忽略，TODO 注释标注待完善

4. **缺失的关键功能**：
   - 比较运算符（`==`/`!=`）导致 `CUDAStream` 无法直接用于容器或条件判断
   - `std::hash` 缺失导致无法用于 `unordered_map`/`unordered_set`
   - `query()` 和 `synchronize()` 缺失，流级别的同步操作需要绕开此接口直接调用 CUDA API
   - `getStreamFromExternal()` 缺失，限制了与第三方库（如 cuDNN、NCCL）自定义流的互操作
