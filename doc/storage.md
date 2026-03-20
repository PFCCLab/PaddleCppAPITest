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

| torch API | paddle API 兼容性 | 备注 |
|---|---|---|
| `isSharedStorageAlias(const Storage&, const Storage&)` | ✅ | 已实现 |
| `Storage::use_byte_size_t` | ✅ | 已实现 |
| `Storage::unsafe_borrow_t` | ✅ | 已实现 |

---

### 构造与初始化

| torch API | paddle API 兼容性 | 备注 |
|---|---|---|
| `Storage()` | ✅ | 已实现 |
| `Storage(intrusive_ptr<StorageImpl>)` | ❌ | Paddle 使用 `shared_ptr<phi::Allocation>`，无 `StorageImpl` 构造 |
| `Storage(use_byte_size_t, const SymInt&, Allocator*, bool)` | ❌ | 缺少 `SymInt` 版本 |
| `Storage(use_byte_size_t, size_t, DataPtr, Allocator*, bool)` | ❌ | Paddle 该重载第三参为 `shared_ptr<phi::Allocation>`，非 `DataPtr` |
| `Storage(use_byte_size_t, SymInt, DataPtr, Allocator*, bool)` | ❌ | 缺少 `SymInt + DataPtr` 版本 |
| `Storage(const Storage&)` | ✅ | 已实现 |
| `Storage(Storage&&)` | ✅ | 已实现 |
| `operator=(const Storage&)` | ✅ | 已实现 |
| `operator=(Storage&&)` | ✅ | 已实现 |

---

### 生命周期与容量

| torch API | paddle API 兼容性 | 备注 |
|---|---|---|
| `create_legacy(Device)` | ❌ | 缺失 |
| `reset_legacy()` | ❌ | 缺失 |
| `set_nbytes(size_t)` | ✅ | 已实现 |
| `set_nbytes(SymInt)` | ❌ | 缺少 `SymInt` 版本 |
| `resizable()` | ✅ | 已实现 |
| `nbytes()` | ✅ | 已实现 |
| `sym_nbytes()` | ❌ | 缺失 |

---

### 数据访问与替换

| torch API | paddle API 兼容性 | 备注 |
|---|---|---|
| `data()` | ✅ | 已实现 |
| `mutable_data()` | ✅ | 已实现 |
| `mutable_data_ptr()` | ✅ | 返回 `DataPtr&`（引用），与 PyTorch 语义一致 |
| `data_ptr()` | ✅ | 返回 `const DataPtr&`（引用），与 PyTorch 语义一致 |
| `set_data_ptr(DataPtr&&)` | ✅ | **PR #78060 修复**：已重新实现，接受 `DataPtr&&`，返回旧 DataPtr，不再有不安全类型转换 |
| `set_data_ptr_noswap(DataPtr&&)` | ✅ | **PR #78060 修复**：已重新实现，接受 `DataPtr&&` |
| `set_data_ptr(shared_ptr<phi::Allocation>)` | ✅ | Paddle 特有，使用 shared_ptr 管理 phi::Allocation 生命周期 |
| `set_data_ptr_noswap(shared_ptr<phi::Allocation>)` | ✅ | Paddle 特有，直接使用 phi::Allocation |

---

### 设备与分配器

| torch API | paddle API 兼容性 | 备注 |
|---|---|---|
| `device_type()` | 🔧 | PyTorch 为 `DeviceType`，Paddle 为 `phi::AllocationType` |
| `allocator()` | 🔧 | PyTorch 返回 `at::Allocator*`，Paddle 返回 `phi::Allocator*` |
| `device()` | 🔧 | PyTorch 返回 `at::Device`，Paddle 返回 `phi::Place` |

---

### 内部句柄与别名

| torch API | paddle API 兼容性 | 备注 |
|---|---|---|
| `unsafeReleaseStorageImpl()` | ❌ | 缺失 |
| `unsafeGetStorageImpl()` | ❌ | 缺失 |
| `getWeakStorageImpl()` | ❌ | 缺失 |
| `operator bool()` | ✅ | 已实现 |
| `use_count()` | ✅ | 已实现 |
| `unique()` | ✅ | 已实现 |
| `is_alias_of(const Storage&)` | ✅ | 已实现 |

---

### 外部指针共享

| torch API | paddle API 兼容性 | 备注 |
|---|---|---|
| `UniqueStorageShareExternalPointer(void*, size_t, DeleterFnPtr)` | ❌ | 缺失 |
| `UniqueStorageShareExternalPointer(DataPtr&&, size_t)` | ❌ | 缺失 |

---

### `MaybeOwnedTraits<c10::Storage>`

| torch API | paddle API 兼容性 | 备注 |
|---|---|---|
| `createBorrow(const owned_type&)` | ✅ | 已实现 |
| `assignBorrow(...)` | 🔧 | PyTorch 签名为引用参数；Paddle 为指针参数 |
| `destroyBorrow(...)` | 🔧 | PyTorch 通过 `unsafeReleaseStorageImpl()` 处理；Paddle 直接重置为空 `Storage()` |
| `referenceFromBorrow(const borrow_type&)` | ✅ | 已实现 |
| `pointerFromBorrow(const borrow_type&)` | ✅ | 已实现 |
| `debugBorrowIsValid(const borrow_type&)` | ✅ | 已实现 |

---

### `ExclusivelyOwnedTraits<c10::Storage>`

| torch API | paddle API 兼容性 | 备注 |
|---|---|---|
| `nullRepr()` | ✅ | 已实现 |
| `createInPlace(...)` | ✅ | 已实现 |
| `moveToRepr(Storage&&)` | ✅ | 已实现 |
| `take(...)` | 🔧 | PyTorch 参数是 `Storage&`；Paddle 为 `Storage*` |
| `getImpl(repr_type&)` | ❌ | Paddle 无该重载 |
| `getImpl(const repr_type&)` | ✅ | 已实现 |

---

### Paddle compat 特有接口

| API | 说明 |
|---|---|
| `Storage(shared_ptr<phi::Allocation>, unique_ptr<phi::StorageProperties>)` | 直接从 `phi::Allocation` 构造 |
| `Storage(size_t, phi::Allocator*)` | 简化尺寸构造 |
| `valid()` | 检查是否持有 allocation |
| `allocation()` | 返回底层 `shared_ptr<phi::Allocation>` |
| `set_data_ptr_noswap(shared_ptr<phi::Allocation>)` | 直接替换 allocation |

---

### 兼容性统计（基于以上条目）

| 状态 | 数量 |
|---|---|
| ✅ 已实现 | 29 |
| 🔧 部分兼容 | 8 |
| ❌ 未实现 | 12 |

---

### 结论

- Paddle compat 的 `Storage` 已覆盖基础生命周期、数据访问与别名检查主路径。
- 与 PyTorch 的主要差距在 `StorageImpl` 体系相关接口（`intrusive_ptr/weak_ptr/legacy/external pointer/SymInt`）。
- **PR #78060 修复记录**:
  - `data_ptr()` 现在返回 `const DataPtr&`（引用），`mutable_data_ptr()` 返回 `DataPtr&`（引用），与 PyTorch 语义一致
  - 重新实现 `set_data_ptr(DataPtr&&)` 和 `set_data_ptr_noswap(DataPtr&&)`，不再有不安全类型转换
  - 保留 Paddle 特有的 `set_data_ptr(shared_ptr<phi::Allocation>)` 重载
  - Storage 内部使用 `std::shared_ptr<DataPtr> data_ptr_` 作为 PyTorch 兼容引用的持有者；拷贝 Storage 时共享同一个 `shared_ptr<DataPtr>`，确保外部 DataPtr 路径（`set_data_ptr(DataPtr&&)`）的 context/deleter 不会在拷贝后悬空
  - 对 phi::Allocation 原生路径，`*data_ptr_` 是非拥有性视图（仅含原始指针+device），不增加 allocation 的 refcount
  - `use_count()` 仍反映 `allocation_.use_count()`，行为与修复前一致
