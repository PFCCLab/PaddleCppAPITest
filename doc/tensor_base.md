##### TensorBase.h 头文件 API 兼容情况


##### TensorBase.h 头文件 API 兼容性

✅ 表示已经支持
🚧 表示正在支持
❌ 表示不准备支持
🔧 表示部分支持（有功能限制）

**按照功能分类排序**

---

### 构造与赋值

| torch API                    | paddle API 兼容性 | 测试用例状态 | 优先级 | 备注 |
|------------------------------|------------------|------------|-------|------|
| `TensorBase()`               | ✅               | ✅          |   P0  | 默认构造函数 |
| `TensorBase(const TensorBase&)` | ✅            | ✅          |   P0  | 拷贝构造函数 |
| `TensorBase(TensorBase&&)`   | ✅               | ✅          |   P0  | 移动构造函数 |
| `operator=(const TensorBase&)` | ✅             | ✅          |   P0  | 拷贝赋值运算符 |
| `operator=(TensorBase&&)`    | ✅               | ✅          |   P0  | 移动赋值运算符 |

---

### 数据访问 API

| torch API                    | paddle API 兼容性 | 测试用例状态 | 优先级 | 备注 |
|------------------------------|------------------|------------|-------|------|
| `data_ptr()`                 | ✅               | ✅          |   P0  | 返回 `void*` |
| `data_ptr<T>()`              | ✅               | ✅          |   P0  | 返回类型化指针 |
| `const_data_ptr()`           | ✅               | ✅          |   P0  | 返回 `const void*` |
| `const_data_ptr<T>()`        | ✅               | ✅          |   P0  | 返回类型化 const 指针 |
| `mutable_data_ptr()`         | ✅               | ✅          |   P0  | 返回可变 `void*` |
| `mutable_data_ptr<T>()`      | ✅               | ✅          |   P0  | 返回可变类型化指针 |
| `accessor<T, N>()`           | ✅               | ✅          |   P1  | TensorAccessor |
| `generic_packed_accessor<T, N>()` | ✅          | 🚧          |   P2  | CUDA PackedTensorAccessor |
| `packed_accessor32<T, N>()`  | ✅               | 🚧          |   P2  | 32位索引 PackedAccessor |
| `packed_accessor64<T, N>()`  | ✅               | 🚧          |   P2  | 64位索引 PackedAccessor |

---

### 维度与形状 API

| torch API                    | paddle API 兼容性 | 测试用例状态 | 优先级 | 备注 |
|------------------------------|------------------|------------|-------|------|
| `dim()`                      | ✅               | ✅          |   P0  | 返回维度数 |
| `ndimension()`               | ✅               | ✅          |   P0  | 同 `dim()` |
| `size(dim)`                  | ✅               | ✅          |   P0  | 支持负索引 |
| `sizes()`                    | ✅               | ✅          |   P0  | 返回 IntArrayRef |
| `stride(dim)`                | ✅               | ✅          |   P0  | 支持负索引 |
| `strides()`                  | ✅               | ✅          |   P0  | 返回 IntArrayRef |
| `numel()`                    | ✅               | ✅          |   P0  | 元素总数 |
| `storage_offset()`           | 🔧               | 🚧          |   P2  | 仅 DenseTensor 路径有意义，其他场景退化为 0 |
| `sym_size(dim)`              | 🔧               | 🚧          |   P3  | 由静态 `size` 包装，不支持真实符号表达 |
| `sym_stride(dim)`            | 🔧               | 🚧          |   P3  | 由静态 `stride` 包装，不支持真实符号表达 |
| `sym_sizes()`                | 🔧               | 🚧          |   P3  | 由静态 `sizes` 包装，不支持真实符号表达 |
| `sym_strides()`              | 🔧               | 🚧          |   P3  | 由静态 `strides` 包装，不支持真实符号表达 |
| `sym_numel()`                | 🔧               | 🚧          |   P3  | 由静态 `numel` 包装，不支持真实符号表达 |
| `sym_storage_offset()`       | 🔧               | 🚧          |   P3  | 由 `storage_offset` 包装，不支持真实符号表达 |

---

### 内存与字节 API

| torch API                    | paddle API 兼容性 | 测试用例状态 | 优先级 | 备注 |
|------------------------------|------------------|------------|-------|------|
| `nbytes()`                   | ✅               | ✅          |   P0  | 总字节数 |
| `sym_nbytes()`               | - [ ]            | - [ ]       |   P3  | PyTorch 提供，compat 暂未实现 |
| `itemsize()`                 | ✅               | ✅          |   P0  | 单元素字节数 |
| `element_size()`             | ✅               | ✅          |   P0  | 同 `itemsize()` |

---

### 类型与设备 API

| torch API                    | paddle API 兼容性 | 测试用例状态 | 优先级 | 备注 |
|------------------------------|------------------|------------|-------|------|
| `dtype()`                    | ✅               | ✅          |   P0  | 返回 ScalarType |
| `scalar_type()`              | ✅               | ✅          |   P0  | 同 `dtype()` |
| `device()`                   | ✅               | ✅          |   P0  | 返回 Device |
| `get_device()`               | ✅               | ✅          |   P0  | 返回设备索引 |
| `options()`                  | 🔧               | ✅          |   P0  | 当前仅设置 dtype/device，未完整保留 layout |
| `layout()`                   | ✅               | 🚧          |   P2  | 返回 Layout |
| `key_set()`                  | - [ ]            | - [ ]       |   P3  | DispatchKeySet |

---

### 设备类型检查 API

| torch API                    | paddle API 兼容性 | 测试用例状态 | 优先级 | 备注 |
|------------------------------|------------------|------------|-------|------|
| `is_cpu()`                   | ✅               | ✅          |   P0  | CPU 后端检查 |
| `is_cuda()`                  | ✅               | ✅          |   P0  | CUDA 后端检查 |
| `is_xpu()`                   | - [ ]            | - [ ]       |   P2  | XPU 后端检查 |
| `is_ipu()`                   | - [ ]            | - [ ]       |   P3  | IPU 后端检查 |
| `is_xla()`                   | - [ ]            | - [ ]       |   P3  | XLA 后端检查 |
| `is_mtia()`                  | - [ ]            | - [ ]       |   P3  | MTIA 后端检查 |
| `is_hpu()`                   | - [ ]            | - [ ]       |   P3  | HPU 后端检查 |
| `is_lazy()`                  | - [ ]            | - [ ]       |   P3  | Lazy 后端检查 |
| `is_hip()`                   | - [ ]            | - [ ]       |   P3  | HIP 后端检查 |
| `is_ve()`                    | - [ ]            | - [ ]       |   P3  | VE 后端检查 |
| `is_privateuseone()`         | - [ ]            | - [ ]       |   P3  | PrivateUse1 后端检查 |
| `is_meta()`                  | - [ ]            | - [ ]       |   P3  | Meta tensor 检查 |
| `is_maia()`                  | - [ ]            | - [ ]       |   P3  | Maia 后端检查 |

---

### 张量属性检查 API

| torch API                    | paddle API 兼容性 | 测试用例状态 | 优先级 | 备注 |
|------------------------------|------------------|------------|-------|------|
| `defined()`                  | ✅               | ✅          |   P0  | 是否已定义 |
| `is_contiguous()`            | 🔧               | ✅          |   P0  | 仅支持 `MemoryFormat::Contiguous` |
| `sym_is_contiguous()`        | - [ ]            | - [ ]       |   P3  | 符号化连续性检查 |
| `is_contiguous_or_false()`   | 🔧               | 🚧          |   P3  | 当前行为与 `is_contiguous()` 基本一致 |
| `is_non_overlapping_and_dense()` | 🔧           | 🚧          |   P3  | 自定义 stride 判定逻辑，与 PyTorch 细节可能有差异 |
| `is_complex()`               | ✅               | 🚧          |   P2  | 是否复数类型 |
| `is_floating_point()`        | ✅               | 🚧          |   P2  | 是否浮点类型 |
| `is_signed()`                | ✅               | 🚧          |   P2  | 是否有符号类型 |
| `is_sparse()`                | ✅               | 🚧          |   P3  | 是否稀疏张量 |
| `is_sparse_csr()`            | ✅               | 🚧          |   P3  | 是否 CSR 稀疏张量 |
| `is_mkldnn()`                | - [ ]            | - [ ]       |   P3  | 是否 MKLDNN 张量 |
| `is_mps()`                   | - [ ]            | - [ ]       |   P3  | 是否 MPS 张量 |
| `is_vulkan()`                | - [ ]            | - [ ]       |   P3  | 是否 Vulkan 张量 |
| `is_metal()`                 | - [ ]            | - [ ]       |   P3  | 是否 Metal 张量 |
| `is_quantized()`             | - [ ]            | - [ ]       |   P3  | 是否量化张量 |
| `is_inference()`             | - [ ]            | - [ ]       |   P3  | 是否推理张量 |
| `is_nested()`                | - [ ]            | - [ ]       |   P3  | 是否嵌套张量 |

---

### 内存格式与连续性 API

| torch API                    | paddle API 兼容性 | 测试用例状态 | 优先级 | 备注 |
|------------------------------|------------------|------------|-------|------|
| `contiguous()`               | 🔧               | ✅          |   P0  | 仅支持 Contiguous 格式 |
| `expect_contiguous()`        | - [ ]            | - [ ]       |   P2  | MaybeOwned 版本 |
| `suggest_memory_format()`    | - [ ]            | - [ ]       |   P2  |  |

---

### 就地修改操作 API

| torch API                    | paddle API 兼容性 | 测试用例状态 | 优先级 | 备注 |
|------------------------------|------------------|------------|-------|------|
| `fill_(scalar)`              | ✅               | ✅          |   P0  | 填充标量值 |
| `zero_()`                    | ✅               | ✅          |   P0  | 置零 |
| `copy_(src, non_blocking)`   | ✅               | ✅          |   P0  | 复制数据 |

---

### 形状变换 API

| torch API                    | paddle API 兼容性 | 测试用例状态 | 优先级 | 备注 |
|------------------------------|------------------|------------|-------|------|
| `reshape(shape)`             | ✅               | ✅          |   P0  |  |
| `view(size)`                 | ✅               | ✅          |   P0  | 形状 view |
| `view(dtype)`                | ✅               | ✅          |   P1  | 类型 view |

---

### 类型转换 API

| torch API                    | paddle API 兼容性 | 测试用例状态 | 优先级 | 备注 |
|------------------------------|------------------|------------|-------|------|
| `to(options, ...)`           | 🔧               | ✅          |   P1  | 仅支持 dtype 转换，不支持 device/memory_format |

---

### 存储相关 API

| torch API                    | paddle API 兼容性 | 测试用例状态 | 优先级 | 备注 |
|------------------------------|------------------|------------|-------|------|
| `has_storage()`              | 🔧               | 🚧          |   P2  | 当前等价于 `defined()`，语义弱于 PyTorch |
| `storage()`                  | 🔧               | 🚧          |   P2  | 依赖 DenseTensor Holder，不同 layout 下语义有限 |
| `is_alias_of(other)`         | 🔧               | 🚧          |   P2  | 基于 allocation 指针比较，语义近似 |
| `share_memory_()`            | - [ ]            | - [ ]       |   P3  |  |

---

### 内部状态 API

| torch API                    | paddle API 兼容性 | 测试用例状态 | 优先级 | 备注 |
|------------------------------|------------------|------------|-------|------|
| `is_same(other)`             | ✅               | 🚧          |   P2  | 是否同一张量 |
| `use_count()`                | ✅               | 🚧          |   P3  | 引用计数 |
| `weak_use_count()`           | 🔧               | 🚧          |   P3  | 当前固定返回 0 |
| `is_uniquely_owned()`        | - [ ]            | - [ ]       |   P3  | PyTorch 提供，compat 暂未实现 |
| `reset()`                    | ✅               | 🚧          |   P2  | 重置张量 |
| `_is_zerotensor()`           | - [ ]            | - [ ]       |   P3  |  |
| `_set_zero(zero)`            | - [ ]            | - [ ]       |   P3  |  |
| `is_conj()`                  | - [ ]            | - [ ]       |   P3  | 共轭标记 |
| `_set_conj(conjugate)`       | - [ ]            | - [ ]       |   P3  |  |
| `is_neg()`                   | - [ ]            | - [ ]       |   P3  | 负号标记 |
| `_set_neg(negative)`         | - [ ]            | - [ ]       |   P3  |  |

---

### TensorImpl 访问 API

| torch API                    | paddle API 兼容性 | 测试用例状态 | 优先级 | 备注 |
|------------------------------|------------------|------------|-------|------|
| `unsafeGetTensorImpl()`      | - [ ]            | - [ ]       |   P3  | 内部使用 |
| `unsafeReleaseTensorImpl()`  | - [ ]            | - [ ]       |   P3  | 内部使用 |
| `getIntrusivePtr()`          | - [ ]            | - [ ]       |   P3  | 内部使用 |
| `unsafeReleaseIntrusivePtr()` | - [ ]           | - [ ]       |   P3  | 内部使用 |

---

### 命名张量 API

| torch API                    | paddle API 兼容性 | 测试用例状态 | 优先级 | 备注 |
|------------------------------|------------------|------------|-------|------|
| `has_names()`                | 🔧               | 🚧          |   P3  | 已实现但当前固定返回 false |
| `opt_names()`                | - [ ]            | - [ ]       |   P3  |  |
| `names()`                    | - [ ]            | - [ ]       |   P3  |  |
| `get_named_tensor_meta()`    | - [ ]            | - [ ]       |   P3  |  |

---

### 自动求导 API

| torch API                    | paddle API 兼容性 | 测试用例状态 | 优先级 | 备注 |
|------------------------------|------------------|------------|-------|------|
| `requires_grad()`            | - [ ]            | - [ ]       |   P1  |  |
| `set_requires_grad(bool)`    | - [ ]            | - [ ]       |   P1  |  |
| `requires_grad_(bool)`       | - [ ]            | - [ ]       |   P1  |  |
| `is_leaf()`                  | - [ ]            | - [ ]       |   P2  |  |
| `grad_fn()`                  | - [ ]            | - [ ]       |   P2  |  |
| `retain_grad()`              | - [ ]            | - [ ]       |   P2  |  |
| `retains_grad()`             | - [ ]            | - [ ]       |   P2  |  |
| `_fw_grad(level)`            | - [ ]            | - [ ]       |   P3  | Forward AD |
| `_set_fw_grad(...)`          | - [ ]            | - [ ]       |   P3  | Forward AD |
| `register_hook(hook)`        | - [ ]            | - [ ]       |   P2  |  |
| `remove_hook(pos)`           | - [ ]            | - [ ]       |   P2  |  |
| `output_nr()`                | - [ ]            | - [ ]       |   P3  |  |
| `set_data(new_data)`         | - [ ]            | - [ ]       |   P2  |  |
| `data()`                     | - [ ]            | - [ ]       |   P2  |  |
| `_version()`                 | - [ ]            | - [ ]       |   P3  |  |
| `tensor_data()`              | - [ ]            | - [ ]       |   P3  |  |
| `variable_data()`            | - [ ]            | - [ ]       |   P3  |  |
| `grad_dtype()`               | - [ ]            | - [ ]       |   P3  |  |
| `set_grad_dtype(...)`        | - [ ]            | - [ ]       |   P3  |  |

---

### 视图相关 API

| torch API                    | paddle API 兼容性 | 测试用例状态 | 优先级 | 备注 |
|------------------------------|------------------|------------|-------|------|
| `is_view()`                  | - [ ]            | - [ ]       |   P2  |  |
| `_base()`                    | - [ ]            | - [ ]       |   P2  |  |

---

### 杂项 API

| torch API                    | paddle API 兼容性 | 测试用例状态 | 优先级 | 备注 |
|------------------------------|------------------|------------|-------|------|
| `toString()`                 | 🔧               | 🚧          |   P3  | 已实现简化版本（backend + dtype） |
| `print()`                    | 🔧               | 🚧          |   P3  | 已实现简化输出 |
| `name()`                     | - [ ]            | - [ ]       |   P3  |  |
| `quantizer()`                | - [ ]            | - [ ]       |   P3  | 量化器 |

---

### Paddle 兼容层特有 API

| API                          | 说明 |
|------------------------------|------|
| `_PD_GetInner()`             | 获取内部 PaddleTensor 引用 |

---

### 兼容性统计

| 状态 | 数量 |
|------|------|
| ✅ 已完全支持 | 47 |
| 🚧 正在支持 | 0 |
| 🔧 部分支持 | 20 |
| - [ ] 未实现 | 61 |

---

### 备注

1. **优先级说明**：
   - P0: 核心功能，必须支持
   - P1: 常用功能，高优先级
   - P2: 进阶功能，中优先级
   - P3: 边缘功能，低优先级

2. **部分支持说明**：
   - `contiguous()`: 目前仅支持 `MemoryFormat::Contiguous`，不支持 ChannelsLast 等格式
   - `to()`: 目前仅支持 dtype 转换，不支持 device 和 memory_format 选项

3. **符号化 API** (`sym_*`): 已提供接口，但当前主要是静态数值包装，尚未对齐 PyTorch 的完整符号语义

4. **自动求导 API**: 需要与 Paddle 的自动求导机制对接，待后续实现
