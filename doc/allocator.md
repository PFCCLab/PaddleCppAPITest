##### Allocator.h 头文件 API 兼容情况


##### Allocator.h 头文件 API 兼容性

✅ 表示已经支持
🚧 表示正在支持
❌ 表示不准备支持
🔧 表示部分支持（有功能限制）

**按照功能分类排序**

---

### 类型定义

| torch API                    | paddle API 兼容性 | 测试用例状态 | 优先级 | 备注 |
|------------------------------|------------------|------------|-------|------|
| `DeleterFnPtr`               | ✅               | ✅          |   P0  | 删除器函数指针类型 |

---

### DataPtr 类

#### 构造与赋值

| torch API                    | paddle API 兼容性 | 测试用例状态 | 优先级 | 备注 |
|------------------------------|------------------|------------|-------|------|
| `DataPtr()`                  | ✅               | ✅          |   P0  | 默认构造函数 |
| `DataPtr(void*, Device)`     | ✅               | ✅          |   P0  | 数据指针+设备构造 |
| `DataPtr(void*, void*, DeleterFnPtr, Device)` | ✅ | ✅       |   P0  | 完整构造函数 |
| `DataPtr(const DataPtr&)`    | ✅               | ✅          |   P0  | 拷贝构造函数 |
| `DataPtr(DataPtr&&)`         | ✅               | ✅          |   P0  | 移动构造函数 |
| `operator=(const DataPtr&)`  | ✅               | ✅          |   P0  | 拷贝赋值运算符 |
| `operator=(DataPtr&&)`       | ✅               | ✅          |   P0  | 移动赋值运算符 |

#### 数据访问 API

| torch API                    | paddle API 兼容性 | 测试用例状态 | 优先级 | 备注 |
|------------------------------|------------------|------------|-------|------|
| `get()`                      | ✅               | ✅          |   P0  | 获取原始指针 |
| `operator->()`               | ✅               | ✅          |   P0  | 指针访问运算符 |
| `operator bool()`            | ✅               | ✅          |   P0  | 布尔转换 |
| `device()`                   | ✅               | ✅          |   P0  | 获取设备 |
| `get_deleter()`              | ✅               | ✅          |   P1  | 获取删除器 |
| `get_context()`              | ✅               | ✅          |   P1  | 获取上下文 |
| `clear()`                    | ✅               | ✅          |   P1  | 清空数据 |
| `mutable_get()`              | - [ ]            | - [ ]       |   P2  | 获取可变指针 |
| `release_context()`          | - [ ]            | - [ ]       |   P2  | 释放上下文 |
| `move_context()`             | - [ ]            | - [ ]       |   P2  | 移动上下文 |
| `compare_exchange_deleter()` | - [ ]            | - [ ]       |   P3  | 比较交换删除器 |
| `unsafe_set_device()`        | - [ ]            | - [ ]       |   P3  | 不安全设置设备 |

#### 比较运算符

| torch API                    | paddle API 兼容性 | 测试用例状态 | 优先级 | 备注 |
|------------------------------|------------------|------------|-------|------|
| `operator==(DataPtr, nullptr_t)` | ✅           | ✅          |   P0  |  |
| `operator==(nullptr_t, DataPtr)` | ✅           | ✅          |   P0  |  |
| `operator!=(DataPtr, nullptr_t)` | ✅           | ✅          |   P0  |  |
| `operator!=(nullptr_t, DataPtr)` | ✅           | ✅          |   P0  |  |

---

### Allocator 基类

| torch API                    | paddle API 兼容性 | 测试用例状态 | 优先级 | 备注 |
|------------------------------|------------------|------------|-------|------|
| `allocate(size_t)`           | - [ ]            | - [ ]       |   P2  | 分配内存 |
| `clone(void*, size_t)`       | - [ ]            | - [ ]       |   P3  | 克隆分配 |
| `is_simple_data_ptr()`       | - [ ]            | - [ ]       |   P3  |  |
| `raw_deleter()`              | - [ ]            | - [ ]       |   P2  | 获取原始删除器 |
| `raw_allocate(size_t)`       | - [ ]            | - [ ]       |   P2  | 原始分配 |
| `raw_deallocate(void*)`      | - [ ]            | - [ ]       |   P2  | 原始释放 |
| `copy_data()`                | - [ ]            | - [ ]       |   P3  | 复制数据 |

---

### 全局函数

| torch API                    | paddle API 兼容性 | 测试用例状态 | 优先级 | 备注 |
|------------------------------|------------------|------------|-------|------|
| `SetAllocator()`             | - [ ]            | - [ ]       |   P3  | 设置分配器 |
| `GetAllocator()`             | - [ ]            | - [ ]       |   P3  | 获取分配器 |

---

### InefficientStdFunctionContext

| torch API                    | paddle API 兼容性 | 测试用例状态 | 优先级 | 备注 |
|------------------------------|------------------|------------|-------|------|
| `makeDataPtr()`              | - [ ]            | - [ ]       |   P3  | 创建带自定义删除器的 DataPtr |

---

### Paddle 兼容层特有 API

| API                          | 说明 |
|------------------------------|------|
| `DataPtr(shared_ptr<phi::Allocation>)` | 从 Paddle Allocation 构造 |
| `allocation()`               | 获取底层 phi::Allocation |

---

### 兼容性统计

| 状态 | 数量 |
|------|------|
| ✅ 已完全支持 | 18 |
| 🚧 正在支持 | 0 |
| 🔧 部分支持 | 0 |
| - [ ] 未实现 | 12 |

---

### 备注

1. **优先级说明**：
   - P0: 核心功能，必须支持
   - P1: 常用功能，高优先级
   - P2: 进阶功能，中优先级
   - P3: 边缘功能，低优先级

2. **实现说明**：
   - Paddle 兼容层的 `DataPtr` 基于 `phi::Allocation` 实现
   - 设备类型使用 `phi::Place` 而非 `c10::Device`
   - `Allocator` 基类暂未实现，使用 Paddle 原生的 `phi::Allocator`

3. **命名空间**：
   - `c10::DataPtr` - 主命名空间
   - `at::DataPtr` - 别名
