## typeid.h - TypeMeta 类 API 兼容性

对比文件：
- `/home/may/Paddle/paddle/phi/api/include/compat/c10/util/typeid.h`
- `/home/may/pytorch/c10/util/typeid.h`

状态说明：
- `✅` 已实现（接口存在且签名/语义基本一致）
- `🔧` 部分兼容（接口存在，但签名或实现语义有差异）
- `❌` 未实现（PyTorch 有，Paddle compat 头文件无）

---

### 类职责

`caffe2::TypeMeta` 是类型元信息句柄，核心作用是：

1. 用一个轻量索引 `index_` 指向全局 `TypeMetaData` 表。
2. 提供运行时类型查询（id/name/itemsize）与类型注册入口。
3. 在 `ScalarType` 与 `TypeMeta` 间建立双向映射。

---

### 构造与赋值

| torch API | paddle API 兼容性 | 测试用例状态 | 优先级 | 备注 |
|---|---|---|---|---|
| `TypeMeta() noexcept` | ✅ | - [ ] | P0 | 实现一致性：`✅`；语义：一致；实现：都通过 `_typeMetaData<detail::_Uninitialized>()` 初始化到 `Undefined`。 |
| `~TypeMeta() = default` | ✅ | - [ ] | P2 | 实现一致性：`✅`；语义：一致；实现：默认析构。 |
| `TypeMeta(const TypeMeta&)` | ✅ | - [ ] | P1 | 实现一致性：`✅`；语义：一致；实现：默认拷贝。 |
| `TypeMeta(TypeMeta&&)` | ✅ | - [ ] | P2 | 实现一致性：`✅`；语义：一致；实现：默认移动。 |
| `operator=(const TypeMeta&)` | ✅ | - [ ] | P1 | 实现一致性：`✅`；语义：一致；实现：默认拷贝赋值。 |
| `operator=(TypeMeta&&)` | ✅ | - [ ] | P2 | 实现一致性：`✅`；语义：一致；实现：默认移动赋值。 |
| `operator=(ScalarType)` | ✅ | - [ ] | P0 | 实现一致性：`✅`；语义：一致；实现：都将 `index_` 直接设为 `ScalarType` 的底层值。 |
| `TypeMeta(uint16_t)`（private） | ✅ | - [ ] | P1 | 实现一致性：`✅`；语义：一致；实现：都以 index 直接构造。 |

---

### 查询与访问 API

| torch API | paddle API 兼容性 | 测试用例状态 | 优先级 | 备注 |
|---|---|---|---|---|
| `id() const noexcept` | ✅ | - [ ] | P0 | 实现一致性：`✅`；语义：一致；实现：都返回 `data().id_`。 |
| `isScalarType() const noexcept` | ✅ | - [x] | P0 | 实现一致性：`✅`；语义：已对齐；实现：修复后使用 `index_ <= ScalarType::Undefined`，与上游一致。 |
| `isScalarType(ScalarType) const noexcept` | ✅ | - [x] | P0 | 实现一致性：`✅`；语义：一致；实现：都比较 `index_ == scalar_type`。 |
| `itemsize() const noexcept` | 🔧 | - [ ] | P0 | 实现一致性：`🔧`；语义：部分一致；实现：Torch 对标量走 `scalarTypeItemSizes` 快速路径，Paddle 统一返回 `data().itemsize_`。 |
| `newFn() const noexcept` | ✅ | - [ ] | P1 | 实现一致性：`✅`；语义：一致；实现：都转发 `data().new_`。 |
| `placementNew() const noexcept` | ✅ | - [ ] | P1 | 实现一致性：`✅`；语义：一致；实现：都转发 `data().placementNew_`。 |
| `copy() const noexcept` | ✅ | - [ ] | P1 | 实现一致性：`✅`；语义：一致；实现：都转发 `data().copy_`。 |
| `placementDelete() const noexcept` | ✅ | - [ ] | P1 | 实现一致性：`✅`；语义：一致；实现：都转发 `data().placementDelete_`。 |
| `deleteFn() const noexcept` | ✅ | - [ ] | P1 | 实现一致性：`✅`；语义：一致；实现：都转发 `data().delete_`。 |
| `name() const noexcept` | 🔧 | - [ ] | P1 | 实现一致性：`🔧`；语义：基本一致；实现：Paddle 注册名主要来自 `typeid(T).name()`，Torch 倾向全限定名。 |
| `template <typename T> Match() const noexcept` | ✅ | - [ ] | P1 | 实现一致性：`✅`；语义：一致；实现：都比较 `*this == Make<T>()`。 |

---

### 静态 helper API

| torch API | paddle API 兼容性 | 测试用例状态 | 优先级 | 备注 |
|---|---|---|---|---|
| `template <class T> static constexpr TypeIdentifier Id() noexcept` | ✅ | - [ ] | P0 | 实现一致性：`✅`；语义：一致；实现：都调用 `TypeIdentifier::Get<T>()`。 |
| `template <class T> static std::string_view TypeName() noexcept` | 🔧 | - [ ] | P1 | 实现一致性：`🔧`；语义：部分一致；实现：Torch 用 `get_fully_qualified_type_name<T>()`，Paddle 用 `typeid(T).name()`。 |
| `template <class T> static constexpr size_t ItemSize() noexcept` | ✅ | - [ ] | P1 | 实现一致性：`✅`；语义：一致；实现：都返回 `sizeof(T)`。 |
| `template <typename T> static TypeMeta Make()` | ✅ | - [ ] | P0 | 实现一致性：`🔧`；语义：一致；实现：都通过 `_typeMetaData<T>()` 构造，Torch 额外带编译器 warning 抑制。 |
| `static TypeMeta fromScalarType(ScalarType)` | ✅ | - [x] | P0 | 实现一致性：`✅`；语义：已对齐；实现：都断言 `index <= Undefined`（Paddle 修复后与 Torch 的 `index < NumScalarTypes` 语义一致）。 |
| `ScalarType toScalarType() const` | 🔧 | - [ ] | P0 | 实现一致性：`🔧`；语义：部分一致；实现：标量路径一致；非标量路径 Torch 调 `error_unsupported_typemeta`，Paddle 直接抛 `InvalidArgument`。 |
| `template <class T> static uint16_t addTypeMetaData()` | 🔧 | - [ ] | P0 | 实现一致性：`🔧`；语义：基本一致；实现：注册流程一致，但 Torch 对 `__CUDACC__` 有特殊分支，Paddle 无。 |

---

### 内部函数与注册索引 API

| torch API | paddle API 兼容性 | 测试用例状态 | 优先级 | 备注 |
|---|---|---|---|---|
| `getTypeMetaDatasLock()` | ✅ | - [ ] | P1 | 实现一致性：`✅`；语义：一致；实现：都用于保护 metadata 注册。 |
| `nextTypeIndex`（静态变量） | ✅ | - [ ] | P1 | 实现一致性：`✅`；语义：一致；实现：都记录下一个可分配类型槽位。 |
| `typeMetaDatas()` | ✅ | - [ ] | P1 | 实现一致性：`🔧`；语义：一致；实现：都返回全局元数据表；初始化细节由各自实现文件决定。 |
| `existingMetaDataIndexForType(TypeIdentifier)` | ✅ | - [ ] | P1 | 实现一致性：`✅`；语义：一致；实现：都在已注册区间线性查找。 |
| `template <class T> _typeMetaData() noexcept` | ✅ | - [ ] | P0 | 实现一致性：`✅`；语义：一致；实现：都通过特化将类型映射到 metadata index。 |
| `data() const` | ✅ | - [ ] | P1 | 实现一致性：`✅`；语义：一致；实现：都按 `index_` 返回表项引用。 |
| `error_unsupported_typemeta(TypeMeta)` | 🔧 | - [ ] | P1 | 实现一致性：`🔧`；语义：基本一致；实现：Torch 提供独立错误函数，Paddle 无该函数并在 `toScalarType()` 直接抛错。 |

---

### 非成员运算符

| torch API | paddle API 兼容性 | 测试用例状态 | 优先级 | 备注 |
|---|---|---|---|---|
| `operator==(const TypeMeta&, const TypeMeta&)` | ✅ | - [ ] | P0 | 实现一致性：`✅`；语义：一致；实现：都比较 `index_`。 |
| `operator!=(const TypeMeta&, const TypeMeta&)` | ✅ | - [ ] | P0 | 实现一致性：`✅`；语义：一致；实现：都基于 `==` 取反。 |
| `operator<<(std::ostream&, TypeMeta)` | ✅ | - [ ] | P2 | 实现一致性：`✅`；语义：一致；实现：都输出 `name()`。 |

---

### 兼容性统计

| 状态 | 数量 |
|---|---|
| ✅ 已实现 | 29 |
| 🔧 部分兼容 | 7 |
| ❌ 未实现 | 0 |

---

### 结论

1. `TypeMeta` 主体接口已完整兼容。
2. `isScalarType()` 语义已修复，与 PyTorch 上游对齐：现在包含 `ScalarType::Undefined`，使用 `index_ <= ScalarType::Undefined` 判断。
3. 需要重点关注的函数级差异集中在：`itemsize()`、`TypeName<T>()`、`fromScalarType()`、`toScalarType()`、`addTypeMetaData<T>()`。
4. 建议将上述 5 个函数作为 P0/P1 回归重点。
