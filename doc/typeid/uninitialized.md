## typeid.h - detail::_Uninitialized 类 API 兼容性

对比文件：
- `/home/may/Paddle/paddle/phi/api/include/compat/c10/util/typeid.h`
- `/home/may/pytorch/c10/util/typeid.h`

状态说明：
- `✅` 已实现（接口存在且签名/语义基本一致）
- `🔧` 部分兼容（接口存在，但签名或实现语义有差异）
- `❌` 未实现（PyTorch 有，Paddle compat 头文件无）

---

### 类职责

`caffe2::detail::_Uninitialized` 是 `TypeMeta` 默认状态的哨兵类型，用于将默认构造的 `TypeMeta` 绑定到 `ScalarType::Undefined` 槽位。

---

### API 明细

| torch API | paddle API 兼容性 | 测试用例状态 | 优先级 | 备注 |
|---|---|---|---|---|
| `class _Uninitialized final {}` | ✅ | - [ ] | P1 | 实现一致性：`✅`；语义：一致；实现：两边均为空类，仅承担模板特化标签作用。 |
| `TypeMeta::_typeMetaData<detail::_Uninitialized>()` 特化 | ✅ | - [ ] | P0 | 实现一致性：`✅`；语义：一致；实现：都返回 `ScalarType::Undefined` 对应索引。 |
| `TypeMeta::TypeMeta() noexcept` 对该特化的使用 | ✅ | - [ ] | P0 | 实现一致性：`✅`；语义：一致；实现：默认构造都初始化为 `_Uninitialized` 对应索引。 |

---

### 结论

1. 该类无行为逻辑，兼容性风险极低。
2. 真正需要关注的是 `_typeMetaData<_Uninitialized>()` 特化是否始终映射到 `Undefined`。
