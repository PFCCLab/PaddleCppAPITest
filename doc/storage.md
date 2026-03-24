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
| `Storage(intrusive_ptr<StorageImpl>)` | ❌ | Paddle 使用 `shared_ptr<StorageImpl>`，但 StorageImpl 是 compat 内部类型，不接受 PyTorch 的 `intrusive_ptr<c10::StorageImpl>` |
| `Storage(use_byte_size_t, const SymInt&, Allocator*, bool)` | ❌ | 缺少 `SymInt` 版本 |
| `Storage(use_byte_size_t, size_t, DataPtr, Allocator*, bool)` | ✅ | **PR #78060 修复**：已新增接受 `DataPtr` 的重载，语义与 PyTorch 一致 |
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
| `device_type()` | 🔧 | PyTorch 为 `DeviceType`，Paddle 为 `phi::AllocationType`；两条路径均已正确处理：allocation-backed 路径从 `allocation_->place()` 取值，external DataPtr 路径从 `data_ptr_->device()` 取值 |
| `allocator()` | 🔧 | PyTorch 返回 `at::Allocator*`，Paddle 返回 `phi::Allocator*` |
| `device()` | 🔧 | PyTorch 返回 `at::Device`，Paddle 返回 `phi::Place`；两条路径均已正确处理：allocation-backed 路径从 `allocation_->place()` 取值，external DataPtr 路径从 `data_ptr_->device()._PD_GetInner()` 取值（完整保留 device index） |

---

### 内部句柄与别名

| torch API | paddle API 兼容性 | 备注 |
|---|---|---|
| `unsafeReleaseStorageImpl()` | ❌ | 缺失 |
| `unsafeGetStorageImpl()` | ❌ | 缺失 |
| `getWeakStorageImpl()` | ❌ | 缺失 |
| `operator bool()` | ✅ | 已实现 |
| `use_count()` | ✅ | 已实现；对外暴露的计数等价于“共享同一 `StorageImpl` 的 `Storage` handle 数量”。内部 `StorageHolderView` 为了让 `DenseTensor::holder_` 反映 live storage 会暂持一个 bookkeeping 引用，但 `use_count()` 会将该内部引用扣除；空/无效 Storage 返回 0。`Storage storage = tensor.storage()` 后 `use_count == 2`（tensor wrapper 内部 handle + 外部复制出的 handle）。 |
| `unique()` | ✅ | 已实现；委托给 `use_count() == 1` |
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
| `valid()` | 检查是否持有 allocation 或有效 DataPtr |
| `allocation()` | 返回底层 `shared_ptr<phi::Allocation>` |
| `set_data_ptr(shared_ptr<phi::Allocation>)` | 替换 allocation，返回旧 allocation |
| `set_data_ptr_noswap(shared_ptr<phi::Allocation>)` | 直接替换 allocation，无返回值 |

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
  - **Reference semantics（本轮修复）**：引入 `StorageImpl` 共享状态结构体，`Storage` 改为持有 `shared_ptr<StorageImpl>`。拷贝 `Storage` 时共享同一个 `StorageImpl`，通过任意副本的 `set_data_ptr*()`/`set_nbytes()`/`mutable_data_ptr()` 写操作均对所有副本可见，与 PyTorch `Storage` 作为 `intrusive_ptr<StorageImpl>` handle 的语义一致。移除了旧的 CoW 机制（`ensureUniqueDataPtr()`）。
  - `data_ptr()` 返回 `const DataPtr&`（引用），`mutable_data_ptr()` 返回 `DataPtr&`（引用），均直接引用 `impl_->data_ptr_`，无 CoW 包装
  - `set_data_ptr(DataPtr&&)` 和 `set_data_ptr_noswap(DataPtr&&)` 直接修改 `impl_->data_ptr_`，对所有共享该 impl 的 Storage 副本可见
  - 保留 Paddle 特有的 `set_data_ptr(shared_ptr<phi::Allocation>)` 重载
  - `use_count()`：旧实现 allocation-backed 路径返回 allocation 的引用计数，会把 `DenseTensor::holder_` 等内部实现细节暴露出来；**本轮修复**后按 `StorageImpl` 共享语义计数，并扣除 `StorageHolderView` 的内部 bookkeeping 引用，因此对外结果与 PyTorch 的 `Storage` handle 计数口径保持一致
  - allocation-backed 路径中 `impl_->data_ptr_` 是对 `phi::Allocation` 的非拥有性视图（仅含原始指针+device，无 deleter），不增加 allocation 的 refcount
  - **TensorBase::storage() 跨 wrapper 共享（本轮修复）**：通过 `phi::DenseTensor::holder_` 恢复/创建共享 `StorageImpl`，并在 `TensorBase` 侧按 `StorageImpl*` 复用 canonical `Storage` 对象（静态弱引用缓存）。配合 `StorageHolderView` 回写 holder，可保证同一底层 `DenseTensor` 的多个 wrapper 观测到一致 storage 语义，`set_data_ptr_noswap()` 写入可立即反映到 `tensor.data_ptr()`。
