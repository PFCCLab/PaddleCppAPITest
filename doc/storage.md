##### Storage.h 头文件 API 兼容性

对比文件：
- `/home/may/Paddle/paddle/phi/api/include/compat/c10/core/Storage.h`
- `/home/may/pytorch/c10/core/Storage.h`

状态说明：
- `✅` 已实现（接口存在且签名/语义基本一致）
- `🔧` 部分兼容（接口存在，但签名或实现语义有差异）
- `❌` 未实现（PyTorch 有，Paddle compat 头文件无）

---

### 全局与标签类型

| torch API | paddle API 兼容性 | 测试用例状态 | 优先级 | 备注 |
|-----------|------------------|------------|-------|------|
| `isSharedStorageAlias(const Storage&, const Storage&)` | ✅ | - [x] | P0 | 已实现，`StorageTest` 已覆盖别名判断路径 |
| `Storage::use_byte_size_t` | ✅ | - [ ] | P1 | 已实现 |
| `Storage::unsafe_borrow_t` | ✅ | - [ ] | P1 | 已实现 |

---

### 构造与初始化

| torch API | paddle API 兼容性 | 测试用例状态 | 优先级 | 备注 |
|-----------|------------------|------------|-------|------|
| `Storage()` | ✅ | - [x] | P0 | 已实现 |
| `Storage(intrusive_ptr<StorageImpl>)` | ❌ | - [ ] | P0 | Paddle 使用 `shared_ptr<StorageImpl>`，不接受 PyTorch `intrusive_ptr` |
| `Storage(use_byte_size_t, const SymInt&, Allocator*, bool)` | ❌ | - [ ] | P1 | 缺少 `SymInt` 版本 |
| `Storage(use_byte_size_t, size_t, DataPtr, Allocator*, bool)` | ✅ | - [ ] | P0 | 已实现 |
| `Storage(use_byte_size_t, SymInt, DataPtr, Allocator*, bool)` | ❌ | - [ ] | P1 | 缺少 `SymInt + DataPtr` 版本 |
| `Storage(const Storage&)` | ✅ | - [ ] | P0 | 已实现 |
| `Storage(Storage&&)` | ✅ | - [ ] | P0 | 已实现 |
| `operator=(const Storage&)` | ✅ | - [ ] | P0 | 已实现 |
| `operator=(Storage&&)` | ✅ | - [ ] | P0 | 已实现 |

---

### 生命周期与容量

| torch API | paddle API 兼容性 | 测试用例状态 | 优先级 | 备注 |
|-----------|------------------|------------|-------|------|
| `create_legacy(Device)` | ❌ | - [ ] | P2 | 缺失 |
| `reset_legacy()` | ❌ | - [ ] | P2 | 缺失 |
| `set_nbytes(size_t)` | ✅ | - [x] | P0 | 已实现，`StorageTest` 覆盖 |
| `set_nbytes(SymInt)` | ❌ | - [ ] | P1 | 缺少 `SymInt` 版本 |
| `resizable()` | ✅ | - [x] | P1 | 已实现，`StorageTest` 覆盖 |
| `nbytes()` | ✅ | - [x] | P0 | 已实现，`StorageTest` 覆盖 |
| `sym_nbytes()` | ❌ | - [ ] | P1 | 缺失 |

---

### 数据访问与替换

| torch API | paddle API 兼容性 | 测试用例状态 | 优先级 | 备注 |
|-----------|------------------|------------|-------|------|
| `data()` | ✅ | - [x] | P0 | 已实现，`StorageTest` 覆盖 |
| `mutable_data()` | ✅ | - [x] | P0 | 已实现，`StorageTest` 覆盖 |
| `mutable_data_ptr()` | ✅ | - [ ] | P1 | 返回 `DataPtr&`，语义与 PyTorch 一致 |
| `data_ptr()` | ✅ | - [x] | P0 | 返回 `const DataPtr&`，`StorageTest` 覆盖 |
| `set_data_ptr(DataPtr&&)` | ✅ | - [x] | P0 | 已实现，`StorageTest` 覆盖 |
| `set_data_ptr_noswap(DataPtr&&)` | ✅ | - [x] | P0 | 已实现，`StorageTest` 覆盖 |
| `swap_data_ptr(Storage&)` | ❌ | - [ ] | P2 | PyTorch 提供，Paddle 未提供 |

---

### 设备与分配器

| torch API | paddle API 兼容性 | 测试用例状态 | 优先级 | 备注 |
|-----------|------------------|------------|-------|------|
| `device_type()` | 🔧 | - [ ] | P1 | PyTorch 返回 `DeviceType`，Paddle 返回 `phi::AllocationType` |
| `allocator()` | 🔧 | - [x] | P1 | PyTorch 返回 `at::Allocator*`，Paddle 返回 `phi::Allocator*` |
| `device()` | 🔧 | - [ ] | P1 | PyTorch 返回 `at::Device`，Paddle 返回 `phi::Place` |

---

### 内部句柄与别名

| torch API | paddle API 兼容性 | 测试用例状态 | 优先级 | 备注 |
|-----------|------------------|------------|-------|------|
| `unsafeReleaseStorageImpl()` | ❌ | - [ ] | P1 | 缺失 |
| `unsafeGetStorageImpl()` | ❌ | - [ ] | P1 | 缺失 |
| `getWeakStorageImpl()` | ❌ | - [ ] | P1 | 缺失 |
| `operator bool()` | ✅ | - [x] | P1 | 已实现，`StorageTest` trait 路径有覆盖 |
| `use_count()` | ✅ | - [x] | P0 | 已实现，`StorageTest` 覆盖多 handle 计数变化 |
| `unique()` | ✅ | - [x] | P0 | 已实现，`StorageTest` 覆盖 |
| `is_alias_of(const Storage&)` | ✅ | - [x] | P0 | 已实现，`StorageTest` 覆盖 |

---

### 外部指针共享

| torch API | paddle API 兼容性 | 测试用例状态 | 优先级 | 备注 |
|-----------|------------------|------------|-------|------|
| `UniqueStorageShareExternalPointer(void*, size_t, DeleterFnPtr)` | ❌ | - [ ] | P2 | 缺失 |
| `UniqueStorageShareExternalPointer(DataPtr&&, size_t)` | ❌ | - [ ] | P2 | 缺失 |

---

### `MaybeOwnedTraits<c10::Storage>`

| torch API | paddle API 兼容性 | 测试用例状态 | 优先级 | 备注 |
|-----------|------------------|------------|-------|------|
| `createBorrow(const owned_type&)` | ✅ | - [x] | P1 | 已实现，`StorageTest` probe 覆盖 |
| `assignBorrow(...)` | ✅ | - [x] | P1 | 当前签名已与 PyTorch 对齐（引用参数） |
| `destroyBorrow(...)` | ✅ | - [x] | P1 | 已实现，`StorageTest` probe 覆盖 |
| `referenceFromBorrow(const borrow_type&)` | ✅ | - [x] | P1 | 已实现 |
| `pointerFromBorrow(const borrow_type&)` | ✅ | - [x] | P1 | 已实现 |
| `debugBorrowIsValid(const borrow_type&)` | ✅ | - [x] | P1 | 已实现 |

---

### `ExclusivelyOwnedTraits<c10::Storage>`

| torch API | paddle API 兼容性 | 测试用例状态 | 优先级 | 备注 |
|-----------|------------------|------------|-------|------|
| `nullRepr()` | ✅ | - [x] | P1 | 已实现，`StorageTest` probe 覆盖 |
| `createInPlace(...)` | ✅ | - [x] | P1 | 已实现 |
| `moveToRepr(Storage&&)` | ✅ | - [x] | P1 | 已实现 |
| `take(...)` | ✅ | - [x] | P1 | 已实现，签名已对齐 |
| `getImpl(repr_type&)` | ✅ | - [x] | P1 | 已实现 |
| `getImpl(const repr_type&)` | ✅ | - [x] | P1 | 已实现 |

---

### 兼容性统计

| 状态 | 数量 |
|---|---|
| ✅ 已实现 | 34 |
| 🔧 部分兼容 | 3 |
| ❌ 未实现 | 13 |

---

### 备注

1. **优先级说明**：
   - P0: 核心功能，必须支持
   - P1: 常用功能，高优先级
   - P2: 进阶功能，中优先级
   - P3: 边缘功能，低优先级

2. **对比范围说明**：
   - 本文档基于头文件声明对比：
     - `paddle/phi/api/include/compat/c10/core/Storage.h`
     - `/home/may/pytorch/c10/core/Storage.h`

3. **主要差异说明**：
   - Paddle `Storage` 的底层句柄体系使用 `shared_ptr<StorageImpl>`，与 PyTorch 的 `intrusive_ptr<StorageImpl>` 路径不同。
   - `SymInt` 构造与 `sym_nbytes()`、legacy/external-pointer 共享接口仍未覆盖。
   - `MaybeOwnedTraits` / `ExclusivelyOwnedTraits` 关键签名目前已对齐到 PyTorch 头文件口径。

4. **测试现状**：
   - `test/StorageTest.cpp` 已覆盖主要访问/计数/traits API。
   - 缺失接口（`SymInt`、legacy、weak impl、external pointer share）暂无直接测试。
