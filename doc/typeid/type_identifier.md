## typeid.h - TypeIdentifier 类 API 兼容性

对比文件：
- `/home/may/Paddle/paddle/phi/api/include/compat/c10/util/typeid.h`
- `/home/may/pytorch/c10/util/typeid.h`

状态说明：
- `✅` 已实现（接口存在且签名/语义基本一致）
- `🔧` 部分兼容（接口存在，但签名或实现语义有差异）
- `❌` 未实现（PyTorch 有，Paddle compat 头文件无）

---

### 类职责

`caffe2::TypeIdentifier` 用于表示 C++ 类型的运行时唯一标识，主要服务于 `TypeMeta` 的类型注册与比对。

---

### 构造与赋值

| torch API | paddle API 兼容性 | 测试用例状态 | 优先级 | 备注 |
|---|---|---|---|---|
| `TypeIdentifier(c10::util::type_index)`（private） | ✅ | - [ ] | P1 | 实现一致性：`🔧`（Torch 继承 `IdWrapper`，Paddle 组合持有 `type_index`）；语义：一致；实现：都以 `type_index` 为底层 id。 |
| 拷贝/移动构造与赋值（隐式） | ✅ | - [ ] | P2 | 实现一致性：`✅`；语义：一致；实现：均为 trivially copyable 风格轻量对象。 |

---

### 核心 API

| torch API | paddle API 兼容性 | 测试用例状态 | 优先级 | 备注 |
|---|---|---|---|---|
| `template <typename T> static constexpr TypeIdentifier Get() noexcept` | ✅ | - [ ] | P0 | 实现一致性：`✅`；语义：一致；实现：都调用 `c10::util::get_type_index<T>()` 生成 id。 |
| `static constexpr TypeIdentifier uninitialized()` | ✅ | - [ ] | P0 | 实现一致性：`✅`；语义：一致；实现：都返回 `type_index{0}`。 |
| `underlyingId()` | ✅ | - [ ] | P0 | 实现一致性：`🔧`（Torch 来自 `IdWrapper`，Paddle 显式转发）；语义：一致；实现：都返回底层 `uint64_t` id。 |
| `operator==` | ✅ | - [ ] | P0 | 实现一致性：`✅`；语义：一致；实现：都比较底层 id 是否相等。 |
| `operator!=` | ✅ | - [ ] | P0 | 实现一致性：`✅`；语义：一致；实现：都基于 `==` 的逻辑取反。 |
| `friend/operator<` | ✅ | - [ ] | P1 | 实现一致性：`✅`；语义：一致；实现：都按底层 id 的数值顺序比较。 |
| `friend/operator<<` | ✅ | - [ ] | P2 | 实现一致性：`✅`；语义：一致；实现：都输出 `underlyingId()`。 |

---

### 关联对象

| torch API | paddle API 兼容性 | 测试用例状态 | 优先级 | 备注 |
|---|---|---|---|---|
| `at::DataType`（`using DataType = TypeIdentifier`） | ✅ | - [ ] | P2 | 实现一致性：`✅`；语义：一致；实现：两边都是别名映射。 |
| `std::hash<caffe2::TypeIdentifier>` | ✅ | - [ ] | P1 | 实现一致性：`🔧`（Torch 宏生成，Paddle 手写特化）；语义：一致；实现：都基于底层 id 哈希。 |

---

### 结论

1. `TypeIdentifier` 现阶段已实现高一致性兼容。
2. 主要差异仅在代码组织方式（继承 vs 组合、宏生成 vs 手写），外部语义保持一致。
3. 建议优先补充 `Get<T>()` 与比较运算符的回归测试，覆盖跨 TU 场景。
