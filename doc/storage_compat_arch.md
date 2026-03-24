# Paddle compat 层兼容方式架构图

本文档说明 Paddle compat 层如何将 PyTorch 的 `c10::Storage` / `c10::DataPtr` 接口映射到 Paddle 内部实现。

> **Note**: 本文档已根据 PR #78060 的最新修复更新。主要变更：
> - 移除了 `TensorStorageRegistry` 全局注册表机制
> - `TensorBase::storage()` 改为通过 `DenseTensor::holder_` 上的 compat holder 复用同一个 `StorageImpl`
> - `storage()` 返回类型改为 `const c10::Storage&`（引用而非值）
> - `TensorBase` 内部改为共享 canonical `std::shared_ptr<c10::Storage>`，避免 alias wrapper 数量抬高 `Storage::use_count()`
> - `has_storage()` / `data_ptr()` 均基于 live holder 同步，而不是构造时快照
> - `use_count()` 以 `Storage` handle 视角计数，并扣除内部 `StorageHolderView` bookkeeping 引用

---

## 整体分层架构

```mermaid
flowchart TD
    subgraph L1["PyTorch LibTorch 兼容 API 层"]
        S["c10::Storage"]
        DP["c10::DataPtr"]
        A["c10::Allocator"]
        CUDA["at::cuda::*"]
    end

    subgraph L2["Paddle compat 实现层"]
        SH["Storage.h"]
        AH["Allocator.h"]
        CUDAL["CUDAContextLight.h/.cpp"]
    end

    subgraph L3["Paddle 原生实现层（phi）"]
        PHI["phi::Allocation / phi::Allocator"]
        PLACE["phi::Place"]
        GPU["phi::GPUContext"]
    end

    L1 -->|compat shim| L2
    L2 --> L3
```

---

## TensorBase::storage() 实现机制

PyTorch 中，所有 `TensorBase` wrapper 共享同一个 `TensorImpl`，因此 `TensorBase::storage()` 直接返回 `TensorImpl::storage_`。Paddle compat 层中，`at::TensorBase` 维护一个共享的 canonical `std::shared_ptr<c10::Storage>`，并通过 `phi::DenseTensor::holder_` 上挂接的 compat holder 复用同一个 `StorageImpl`。

```mermaid
flowchart TD
    subgraph WRAPPERS["at::TensorBase wrappers（value 语义）"]
        W1["TensorBase t1\nstorage_ (shared_ptr<c10::Storage>)"]
        W2["TensorBase t2 = t1\nstorage_ (same shared_ptr target)"]
    end

    subgraph STORAGE["c10::Storage handles"]
        S1["Storage s1 (shared_ptr&lt;StorageImpl&gt;)"]
        S2["Storage s2 (shared_ptr&lt;StorageImpl&gt;)"]
    end

    subgraph IMPL["c10::StorageImpl"]
        SI["data_allocation_ / nbytes_ / data_ptr_ / place_"]
        HV["StorageHolderView\nphi::Allocation adapter"]
    end

    subgraph PHI["Paddle 内部"]
        DT["phi::DenseTensor\nholder_ (shared_ptr&lt;phi::Allocation&gt;)"]
        ALLOC["phi::Allocation\nptr_ / place_ / size_"]
    end

    W1 -->|"storage() returns const ref"| S1
    W2 -->|"storage() returns const ref"| S2
    S1 --> SI
    S2 --> SI
    SI --> ALLOC
    SI --> HV
    DT --> HV
```

### 工作流程说明（PR #78060 修复后）

1. **首次同步**：`TensorBase::storage()` / `has_storage()` / `data_ptr()` 调用 `SyncStorageFromTensor()`，读取当前 `phi::DenseTensor::holder_`。
2. **holder 适配**：如果 holder 还是原始 `phi::Allocation`，compat 层会创建一个共享 `StorageImpl`，再生成 `StorageHolderView` 并回写到 `DenseTensor::holder_`。
3. **跨 wrapper 共享**：后续所有看到同一个 holder 的 `TensorBase` wrapper 都能恢复同一个 `StorageImpl`。对于 `t2 = t1` 这类 alias wrapper，二者共享同一个 canonical `Storage` 句柄，不会因为 wrapper 数量增加而额外抬高 `Storage::use_count()`。
4. **live 数据指针**：`Storage::set_data_ptr*()` 更新 `StorageImpl` 后，`StorageHolderView::ptr()` 立即反映新地址，`tensor.data_ptr()` 与 `tensor.storage()` 因此保持一致。

---

## c10::Storage 共享 StorageImpl 设计

Paddle compat 的 `Storage` 采用与 PyTorch 相同的 **shared handle** 设计：多个 `Storage` 副本共享同一个 `StorageImpl`，通过任意副本的 `set_data_ptr*()`/`set_nbytes()`/`mutable_data_ptr()` 写操作均对所有副本可见。

```mermaid
classDiagram
    class c10_Storage["c10::Storage"] {
        +shared_ptr~StorageImpl~ impl_
        +data_ptr() const DataPtr&
        +mutable_data_ptr() DataPtr&
        +set_data_ptr(DataPtr&&) DataPtr
        +set_data_ptr_noswap(DataPtr&&)
        +set_nbytes(size_t)
        +use_count() size_t
        +device() phi::Place
        +allocation() shared_ptr~phi::Allocation~
    }

    class StorageImpl["c10::StorageImpl (Paddle compat)"] {
        +shared_ptr~phi::Allocation~ data_allocation_
        +phi::Allocator* allocator_
        +size_t nbytes_
        +bool resizable_
        +phi::Place place_
        +DataPtr data_ptr_
        +weak_ptr~StorageHolderView~ tensor_holder_
    }

    class StorageHolderView["StorageHolderView"] {
        +ptr() void*
        +size() size_t
        +place() phi::Place
    }

    class DataPtr["c10::DataPtr"] {
        +UniqueVoidPtr ptr_
        +phi::Place device_
        +get() void*
        +get_deleter() DeleterFnPtr
        +get_context() void*
        +device() c10::Device
    }

    class phi_Allocation["phi::Allocation"] {
        +void* ptr_
        +phi::Place place_
        +size_t size_
    }

    c10_Storage --> "1" StorageImpl : shared_ptr (all copies share one)
    StorageImpl --> "0..1" phi_Allocation : allocation-backed path
    StorageImpl --> "1" DataPtr : direct member (non-owning view or external)
    StorageImpl --> "0..1" StorageHolderView : weak back-reference
    StorageHolderView --> "1" StorageImpl : shared owner
```

### 架构说明

| 属性 | PyTorch StorageImpl | Paddle compat StorageImpl |
|------|---------------------|---------------------------|
| Storage handle | `intrusive_ptr<StorageImpl>` | `shared_ptr<StorageImpl>` |
| 数据所有权 | `DataPtr data_ptr_`（直接成员） | `DataPtr data_ptr_`（直接成员，与 PyTorch 相同） |
| allocation-backed | 无（直接通过 DataPtr） | `shared_ptr<phi::Allocation>`（额外保存） |
| Tensor holder | `TensorImpl` 直接持有 `Storage` | `DenseTensor::holder_` 指向 `StorageHolderView`，借此恢复 shared `StorageImpl` |
| DataPtr 视图 | 由 Allocator 的 deleter 管理 | 对 `phi::Allocation`：非拥有性原始指针视图；外部 `DataPtr`：直接存储 |
| 设备信息来源 | `data_ptr_.device()` | `data_allocation_->place()` 或缓存的 `place_` |
| 引用计数来源 | `intrusive_ptr` 计数 | `impl_.use_count()` 减去 `StorageHolderView` 的 bookkeeping 引用 |
| copy-on-write | 无（single StorageImpl） | 无（已移除 CoW；共享 impl_ 直接传播写操作） |

### use_count() 计算依据（PR #78060 修复后）

```cpp
size_t use_count() const {
    if (!valid()) return 0;
    size_t count = impl_.use_count();
    if (!impl_->tensor_holder_.expired() && count > 0) {
        --count;
    }
    return count;
}
```

- **减去 holder bookkeeping 引用**：`StorageHolderView` 需要共享 `StorageImpl` 才能让 `DenseTensor` 的 `holder_` 始终反映 live storage，但它不应计入对外暴露的 `use_count()`
- **典型计数示例**：
    - 单 wrapper 场景，仅通过 `const auto& s = tensor.storage()` 借用引用：`use_count == 1`
    - 增加一个显式 `Storage` 复制句柄（如 `Storage s2 = tensor.storage()`）：`use_count == 2`
    - 额外复制 alias wrapper（`TensorBase alias = tensor`）不应单独再 +1
- **空/无效 Storage**：`valid()` 返回 false 时返回 0

### Reference Semantics：写操作传播示意

```mermaid
sequenceDiagram
    participant A as Storage a
    participant Impl as StorageImpl (shared)
    participant B as Storage b = a

    Note over A,B: Storage b = a 后，a 和 b 共享同一 impl_

    A->>Impl: set_data_ptr_noswap(new_ptr)
    Note over Impl: impl_->data_ptr_ = new_ptr

    B->>Impl: data() / data_ptr()
    Impl-->>B: new_ptr (可见)
```

---

## c10::DataPtr 与 phi::Place 的映射

```mermaid
flowchart LR
    subgraph DP["c10::DataPtr"]
        PTR["UniqueVoidPtr ptr_"]
        DEV["phi::Place device_"]
    end

    subgraph UV["c10::detail::UniqueVoidPtr"]
        RAW["void* raw_ptr"]
        CTX["void* context"]
        DEL["DeleterFnPtr deleter"]
    end

    DP --> PTR
    PTR --> UV

    C2["c10::Device\n\ninner_ = phi::Place\nindex() → GetDeviceId()\ntype() → GetType()"] -.->|"_PD_GetInner()"| DEV
```

---

## at::cuda 接口映射（CUDAContextLight）

```mermaid
flowchart TD
    subgraph ATEN["at::cuda 兼容层"]
        GETBLAS["getCurrentCUDABlasHandle()"]
        ISAVA["is_available()"]
        GETALLOC["getCUDADeviceAllocator()"]
    end

    subgraph ADAPTER["PaddleCUDAAllocatorAdapter (c10::Allocator)"]
        ALLOCATE["allocate(n)"]
        COPYDATA["copy_data(dst, src, n)"]
    end

    subgraph PHI["phi 层"]
        GPUCTX["phi::DeviceContextPool"]
        GPUINFO["phi::backends::gpu"]
        ALLOCFAC["AllocatorFacade"]
    end

    GETBLAS -->|getCurrentGPUContext| GPUCTX -->|cublas_handle| GETBLAS
    ISAVA -->|device_count| GPUINFO -->|GetGPUDeviceCount| ISAVA
    GETALLOC -->|static adapter| ADAPTER
    ALLOCATE -->|n>0: GetAllocator| ALLOCFAC
    ALLOCATE -->|n=0: 保留 CUDA device| ALLOCATE
    COPYDATA -->|cudaMemcpy D2D| COPYDATA
```

### at::cuda::getCUDADeviceAllocator()

提供 Paddle CUDA Allocator 的 `c10::Allocator` 适配：

```cpp
c10::Allocator* getCUDADeviceAllocator() {
    static PaddleCUDAAllocatorAdapter adapter;
    return &adapter;
}
```

`PaddleCUDAAllocatorAdapter` 将 `phi::AllocatorFacade` 的 GPU 分配器包装为 `c10::Allocator` 接口：

| 方法 | 行为 |
|------|------|
| `allocate(0)` | 返回 `DataPtr(nullptr, nullptr, nullptr, Device(CUDA, current_device_id))`，保留当前 CUDA 设备信息，不触发实际分配 |
| `allocate(n>0)` | 通过 `AllocatorFacade` 在当前 GPU 上分配，所有权通过 `deletePaddleCUDAAllocation` deleter 管理 |
| `copy_data(dst, src, n)` | 使用 `cudaMemcpy(dst, src, n, cudaMemcpyDeviceToDevice)` 实现 GPU-to-GPU 拷贝，兼容 `c10::Allocator::clone()` 语义 |
| `raw_deleter()` | 返回 `nullptr`，表示 raw API 不可用。`c10::Allocator` raw 契约要求 `allocate(n)` 返回的 DataPtr 满足 `get()==get_context()`，但本实现中 `data=device_ptr`、`context=phi::Allocation*`，两者不等，因此不能宣称 raw API 可用（PR #78060 当轮修复）。 |

---

## 注意事项

1. **StorageImpl 共享设计**：`Storage b = a` 后两者共享同一个 `StorageImpl`。任何通过 a 或 b 的写操作（`set_data_ptr*`、`set_nbytes`、`mutable_data_ptr` 返回引用后修改）立即对另一方可见。这与 PyTorch 中 `Storage` 作为 `intrusive_ptr<StorageImpl>` handle 的语义一致。

2. **独立 Storage 互不影响**：`Storage a(alloc1); Storage b(alloc2)` 各自持有独立的 `StorageImpl`，写操作不跨越 impl 边界。

   **TensorBase::storage() live 共享设计**（PR #78060 修复后）：每个 `TensorBase` wrapper 都持有自己的 `storage_` handle，但共享同一个底层 `StorageImpl`。不再使用全局 `TensorStorageRegistry`，而是把 shared impl 信息编码到 `DenseTensor::holder_` 上：

   ```cpp
   at::TensorBase t1 = paddle::ones({2, 3});
   at::TensorBase t2 = t1;                    // 同一底层 DenseTensor
   c10::Storage s1 = t1.storage();
    c10::Storage s2 = t2.storage();           // s1 / s2 共享同一 StorageImpl
   s1.set_data_ptr_noswap(new_alloc);
   // t2.data_ptr() / s2.data() 立即看到相同的新地址
   ```

3. **phi::Allocation DataPtr 视图**：allocation-backed 路径中，`impl_->data_ptr_` 是对 `phi::Allocation` 的非拥有性视图（只含原始指针 + device，无 deleter），真实所有权由 `impl_->data_allocation_` 维护。

4. **多卡 device index 保留**：`phi::GPUPlace(n)` 的 device id 为 `n`，通过 `phi::Place::GetDeviceId()` 可完整读回，因此 `DataPtr::device().index()` 在多卡场景下返回正确值。
