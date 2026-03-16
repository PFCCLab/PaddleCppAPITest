##### TypeMeta(typeid.h) 头文件 API 兼容情况

✅ 表示已经支持
🚧 表示正在支持
❌ 表示不准备支持
🔧 表示部分支持（有功能限制或行为差异）

**对比文件**
- Paddle: `/home/may/Paddle/paddle/phi/api/include/compat/c10/util/typeid.h`
- PyTorch: `/home/may/pytorch/c10/util/typeid.h`

---

### TypeIdentifier / DataType

| torch API | paddle API 兼容性 | 测试用例状态 | 优先级 | 备注 |
|---|---|---|---|---|
| `caffe2::TypeIdentifier::Get<T>()` | 🔧 | - [ ] | P1 | 接口一致；实现不同：Torch 基于 `c10::util::type_index`，Paddle 基于函数内静态地址 |
| `caffe2::TypeIdentifier::uninitialized()` | ✅ | - [ ] | P1 | 接口一致 |
| `TypeIdentifier::underlyingId()` | ✅ | - [ ] | P1 | 接口一致 |
| `operator< / == / != / <<` | ✅ | - [ ] | P1 | 接口一致 |
| `at::DataType` (`using DataType = TypeIdentifier`) | ✅ | - [ ] | P1 | 接口一致 |
| `std::hash<caffe2::TypeIdentifier>` | ✅ | - [ ] | P1 | 接口一致 |

---

### TypeMeta 核心接口

| torch API | paddle API 兼容性 | 测试用例状态 | 优先级 | 备注 |
|---|---|---|---|---|
| `TypeMeta()` / 拷贝移动构造与赋值 | ✅ | - [ ] | P0 | 接口一致 |
| `id()` | ✅ | - [ ] | P0 | 接口一致 |
| `isScalarType()` / `isScalarType(ScalarType)` | 🔧 | - [ ] | P1 | Paddle 使用 `index_ < ScalarType::Undefined`，Torch 使用 `index_ < NumScalarTypes` |
| `itemsize()` | 🔧 | - [ ] | P0 | 接口一致；量化类型路径存在差异（Paddle qint 槽位未完整填充） |
| `newFn()` / `placementNew()` / `copy()` / `placementDelete()` / `deleteFn()` | ✅ | - [ ] | P1 | 接口一致 |
| `name()` | 🔧 | - [ ] | P1 | Paddle 依赖 `typeid(T).name()`，Torch 倾向全限定名，字符串可读性/稳定性不同 |
| `Match<T>()` | ✅ | - [ ] | P1 | 接口一致 |
| `TypeMeta::Id<T>()` / `ItemSize<T>()` | ✅ | - [ ] | P1 | 接口一致 |
| `TypeMeta::TypeName<T>()` | 🔧 | - [ ] | P2 | Paddle 结果可能为编译器相关 mangled 名称 |
| `TypeMeta::Make<T>()` | ✅ | - [ ] | P0 | 接口一致 |
| `TypeMeta::fromScalarType()` | 🔧 | - [ ] | P0 | 边界检查条件与 Torch 不同（`<= Undefined` vs `< NumScalarTypes`） |
| `TypeMeta::toScalarType()` | 🔧 | - [ ] | P0 | 非标量类型错误路径不同：Paddle 直接抛 `InvalidArgument` |
| `operator== / != / <<` | ✅ | - [ ] | P1 | 接口一致 |

---

### 动态类型注册与宏

| torch API | paddle API 兼容性 | 测试用例状态 | 优先级 | 备注 |
|---|---|---|---|---|
| `addTypeMetaData<T>()` | 🔧 | - [ ] | P1 | 主流程一致；Torch 有 `__CUDACC__` 特殊分支，Paddle 无该分支 |
| `CAFFE_KNOWN_TYPE` | ✅ | - [ ] | P1 | 接口存在 |
| `CAFFE_DEFINE_KNOWN_TYPE` | ✅ | - [ ] | P1 | 接口存在 |
| `CAFFE_DECLARE_KNOWN_TYPE` | ✅ | - [ ] | P1 | 接口存在 |
| `CAFFE_KNOWN_TYPE_NOEXPORT` | ✅ | - [ ] | P2 | 接口存在 |
| 内置已声明类型（`std::string`、`char*`、`float*` 等） | ✅ | - [ ] | P1 | 两边均覆盖常见类型 |
| `long` 及 `std::vector<long>` guard 注册 | ❌ | - [ ] | P2 | Torch 通过 `_guard_long_unique` 兼容；Paddle 当前未提供对应声明 |

---

### 行为差异总结（重点）

1. **类型名字符串差异**
   - Torch 的 `TypeName<T>()` 使用全限定名工具，输出更稳定；Paddle 使用 `typeid(T).name()`，结果受编译器影响。

2. **量化类型 itemsize 差异风险**
   - Torch 在 `scalarTypeItemSizes` 覆盖量化类型；Paddle 注释中标注 qint C++ 类型未定义，对应槽位可能保持默认值。

3. **已知类型注册策略差异**
   - Torch 默认采用 `CAFFE_DECLARE_KNOWN_TYPE` + `.cpp` 定义；Paddle 在兼容头里使用 `CAFFE_KNOWN_TYPE_NOEXPORT` 做 header 内惰性注册。

4. **`long` 类型 guard 缺失**
   - Torch 专门处理 `long` 与 `int32_t/int64_t` 可能同构的问题；Paddle 当前未补齐该兼容分支。

---

### 兼容性统计（本次对比）

| 状态 | 数量 |
|---|---|
| ✅ 已完全支持 | 21 |
| 🚧 正在支持 | 0 |
| 🔧 部分支持 | 9 |
| ❌ 未支持 | 1 |

---

### 建议优先级

1. **P0**：补齐量化相关 `itemsize` 行为一致性（至少明确 qint 的兼容策略）。
2. **P1**：统一 `TypeName<T>()` 输出策略，减少跨编译器差异。
3. **P2**：补充 `long` / `std::vector<long>` 的 guard 类型注册分支。
