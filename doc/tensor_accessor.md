##### TensorAccessor.h 头文件 API 兼容性

对比文件：
- `/home/may/Paddle/paddle/phi/api/include/compat/ATen/core/TensorAccessor.h`
- `/home/may/pytorch/aten/src/ATen/core/TensorAccessor.h`

✅ 表示已经支持
🚧 表示正在支持
❌ 表示不准备支持
🔧 表示部分支持（有功能限制）

**按照功能分类排序**

---

### 指针 Traits 与基础类型

| torch API | paddle API 兼容性 | 测试用例状态 | 优先级 | 备注 |
|-----------|------------------|------------|-------|------|
| `DefaultPtrTraits<T>::PtrType` | ✅ | - [x] | P0 | 两端都提供默认指针 traits（`T*`） |
| `RestrictPtrTraits<T>::PtrType`（CUDA/HIP 条件导出） | ❌ | - [ ] | P0 | PyTorch 在 `__CUDACC__/__HIPCC__` 下导出；Paddle 头文件仅在注释中提及，未提供定义 |

---

### TensorAccessor 核心接口

| torch API | paddle API 兼容性 | 测试用例状态 | 优先级 | 备注 |
|-----------|------------------|------------|-------|------|
| `TensorAccessorBase<T, N, PtrTraits, index_t>` | ✅ | - [x] | P0 | 模板参数与核心成员接口一致：`sizes()/strides()/size()/stride()/data()` |
| `TensorAccessor<T, N, PtrTraits, index_t>` | ✅ | - [x] | P0 | 多维索引链式访问接口一致（`operator[]`） |
| `TensorAccessor<T, 1, ...>::operator[] -> T&` | ✅ | - [x] | P0 | 1 维特化存在且行为一致 |

---

### Packed Accessor 接口

| torch API | paddle API 兼容性 | 测试用例状态 | 优先级 | 备注 |
|-----------|------------------|------------|-------|------|
| `GenericPackedTensorAccessorBase<T, N, PtrTraits, index_t>` | ✅ | - [x] | P1 | 构造、`size/stride/data` 等访问接口可对齐 |
| `GenericPackedTensorAccessor<T, N, PtrTraits, index_t>` | ✅ | - [x] | P1 | 构造与基础访问对齐 |
| `GenericPackedTensorAccessor::transpose(dim1, dim2)` | 🔧 | - [x] | P1 | Paddle 接口存在；现有测试对该路径保守覆盖（存在实现差异注释） |
| `GenericPackedTensorAccessor<T, 1, ...>` 专门特化（1D `operator[] -> T&`） | 🔧 | - [x] | P0 | PyTorch 在 `torch/headeronly/core/TensorAccessor.h` 中有 1D 特化；Paddle 头文件未见对应显式特化 |

---

### 别名与向后兼容

| torch API | paddle API 兼容性 | 测试用例状态 | 优先级 | 备注 |
|-----------|------------------|------------|-------|------|
| `PackedTensorAccessor`（`C10_DEFINE_DEPRECATED_USING`） | 🔧 | - [x] | P1 | PyTorch 在本头文件保留 deprecated 类型别名；Paddle 未在本头文件声明同名别名，但在 `TensorBody.h` 提供 deprecated `packed_accessor()` 方法 |
| `PackedTensorAccessor32<T, N, PtrTraits>` | ✅ | - [x] | P1 | 别名存在且可用 |
| `PackedTensorAccessor64<T, N, PtrTraits>` | ✅ | - [x] | P1 | 别名存在且可用 |

---

### 兼容性统计

| 状态 | 数量 |
|------|------|
| ✅ 已完全支持 | 8 |
| 🚧 正在支持 | 0 |
| 🔧 部分支持 | 3 |
| ❌ 未实现 | 1 |

---

### 备注

1. **优先级说明**：
   - P0: 核心功能，必须支持
   - P1: 常用功能，高优先级
   - P2: 进阶功能，中优先级
   - P3: 边缘功能，低优先级

2. **对比范围说明**：
   - 本文档基于以下头文件对比：
     - `paddle/phi/api/include/compat/ATen/core/TensorAccessor.h`
     - `/home/may/pytorch/aten/src/ATen/core/TensorAccessor.h`
   - 由于 PyTorch 当前 `ATen/core/TensorAccessor.h` 主要通过 type alias 转发到 `torch/headeronly/core/TensorAccessor.h`，本文对涉及行为语义的条目参考了该转发目标实现。

3. **主要差异说明**：
   - `RestrictPtrTraits`：PyTorch 条件导出，Paddle 当前缺失显式定义。
   - 1D `GenericPackedTensorAccessor`：PyTorch 有显式特化；Paddle 头文件未见对应特化声明。
   - deprecated 类型别名：PyTorch 在 `TensorAccessor.h` 保留 `PackedTensorAccessor` 别名；Paddle 侧主要通过 `TensorBody.h` 的 deprecated `packed_accessor()` 方法维持兼容调用路径。
