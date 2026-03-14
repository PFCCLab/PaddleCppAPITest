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
| `mutable_data_ptr()` | 🔧 | PyTorch 返回 `DataPtr&`，Paddle 返回按值 `DataPtr` |
| `data_ptr()` | 🔧 | PyTorch 返回 `const DataPtr&`，Paddle 返回按值 `DataPtr` |
| `set_data_ptr(DataPtr&&)` | 🔧 | 已实现，但通过 `new_data_ptr.get()` 直接包装为 `phi::Allocation*`，语义与 `StorageImpl` 版不同 |
| `set_data_ptr_noswap(DataPtr&&)` | 🔧 | 已实现，语义同上存在差异 |

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
| ✅ 已实现 | 25 |
| 🔧 部分兼容 | 10 |
| ❌ 未实现 | 14 |

---

### 结论

- Paddle compat 的 `Storage` 已覆盖基础生命周期、数据访问与别名检查主路径。
- 与 PyTorch 的主要差距在 `StorageImpl` 体系相关接口（`intrusive_ptr/weak_ptr/legacy/external pointer/SymInt`）。
- 主要“部分兼容”来自底层模型差异：Paddle 以 `phi::Allocation` 和 `phi::Place` 为核心，导致 `data_ptr` 引用语义、设备类型与 traits 签名与上游不完全一致。
