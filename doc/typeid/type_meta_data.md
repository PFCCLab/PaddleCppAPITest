## typeid.h - detail::TypeMetaData 结构体 API 兼容性

对比文件：
- `/home/may/Paddle/paddle/phi/api/include/compat/c10/util/typeid.h`
- `/home/may/pytorch/c10/util/typeid.h`

状态说明：
- `✅` 已实现（接口存在且签名/语义基本一致）
- `🔧` 部分兼容（接口存在，但签名或实现语义有差异）
- `❌` 未实现（PyTorch 有，Paddle compat 头文件无）

---

### 类职责

`caffe2::detail::TypeMetaData` 是 `TypeMeta` 的底层元数据载体，记录某类型的大小、构造/拷贝/析构函数指针、类型 id 和类型名。

---

### 构造函数

| torch API | paddle API 兼容性 | 测试用例状态 | 优先级 | 备注 |
|---|---|---|---|---|
| `constexpr TypeMetaData() noexcept` | ✅ | - [ ] | P0 | 实现一致性：`✅`；语义：一致；实现：两边默认值均为 itemsize=0、函数指针为 `nullptr`、id 为 `uninitialized()`、name 为 `"nullptr (uninitialized)"`。 |
| `constexpr TypeMetaData(size_t, New*, PlacementNew*, Copy*, PlacementDelete*, Delete*, TypeIdentifier, std::string_view) noexcept` | ✅ | - [ ] | P0 | 实现一致性：`✅`；语义：一致；实现：都按参数原样填充所有字段，无附加逻辑。 |

---

### 成员类型与字段

| torch API | paddle API 兼容性 | 测试用例状态 | 优先级 | 备注 |
|---|---|---|---|---|
| `using New = void*()` | ✅ | - [ ] | P1 | 实现一致性：`✅`；语义：一致；实现：函数签名一致。 |
| `using PlacementNew = void(void*, size_t)` | ✅ | - [ ] | P1 | 实现一致性：`✅`；语义：一致；实现：函数签名一致。 |
| `using Copy = void(const void*, void*, size_t)` | ✅ | - [ ] | P1 | 实现一致性：`✅`；语义：一致；实现：函数签名一致。 |
| `using PlacementDelete = void(void*, size_t)` | ✅ | - [ ] | P1 | 实现一致性：`✅`；语义：一致；实现：函数签名一致。 |
| `using Delete = void(void*)` | ✅ | - [ ] | P1 | 实现一致性：`✅`；语义：一致；实现：函数签名一致。 |
| `itemsize_` | ✅ | - [ ] | P1 | 实现一致性：`✅`；语义：一致；实现：都表示单元素字节大小。 |
| `new_/placementNew_/copy_/placementDelete_/delete_` | ✅ | - [ ] | P1 | 实现一致性：`✅`；语义：一致；实现：都存储对应函数指针。 |
| `id_` | ✅ | - [ ] | P1 | 实现一致性：`✅`；语义：一致；实现：都持有 `TypeIdentifier`。 |
| `name_` | ✅ | - [ ] | P1 | 实现一致性：`✅`；语义：一致；实现：都使用 `std::string_view` 存名称。 |

---

### 兼容性统计

| 状态 | 数量 |
|---|---|
| ✅ 已实现 | 11 |
| 🔧 部分兼容 | 0 |
| ❌ 未实现 | 0 |

---

### 结论

1. `TypeMetaData` 本体（字段与构造）和 Torch 基本完全一致。
2. 差异主要来自“谁来填充这些字段”（即外围 helper 与 `TypeMeta::addTypeMetaData<T>()`），而不是该结构体本身。
