## typeid.h - detail::_guard_long_unique_dummy 类兼容性

对比文件：
- `/home/may/Paddle/paddle/phi/api/include/compat/c10/util/typeid.h`
- `/home/may/pytorch/c10/util/typeid.h`

状态说明：
- `✅` 已实现（接口存在且签名/语义基本一致）
- `🔧` 部分兼容（接口存在，但签名或实现语义有差异）
- `❌` 未实现（PyTorch 有，Paddle compat 头文件无）

---

### 类职责

`caffe2::detail::_guard_long_unique_dummy<T>` 是 long 类型注册保护机制中的占位类型。

设计目的：当 `long` 与 `int32_t/int64_t` 同构时，避免把 `long` 重复注册到已占用的类型 id 上。

---

### 类与关联模板

| torch API | paddle API 兼容性 | 测试用例状态 | 优先级 | 备注 |
|---|---|---|---|---|
| `template <class T> class _guard_long_unique_dummy final {}` | ✅ | - [ ] | P1 | 实现一致性：`✅`；语义：一致；实现：两边均为空模板类，仅做类型占位。 |
| `template <class T> using _guard_long_unique = conditional_t<...>` | ✅ | - [ ] | P1 | 实现一致性：`✅`；语义：一致；实现：当 `long` 与 `int32_t/int64_t` 同构时回退到 dummy，否则使用原类型。 |
| `CAFFE_DECLARE_KNOWN_TYPE(detail::_guard_long_unique<long>, ...)` | ✅ | - [ ] | P1 | 实现一致性：`✅`；语义：一致；实现：两边都通过 guard 类型进行 long 注册。 |
| `CAFFE_DECLARE_KNOWN_TYPE(detail::_guard_long_unique<std::vector<long>>, ...)` | ✅ | - [ ] | P1 | 实现一致性：`✅`；语义：一致；实现：两边都覆盖 `std::vector<long>` 的同构防冲突路径。 |

---

### 结论

1. 该类及其 alias 在当前 Paddle 版本已与 Torch 对齐。
2. 它本身无成员函数，关键语义来自 `_guard_long_unique` alias 与 `CAFFE_DECLARE_KNOWN_TYPE` 的联动。
3. 建议在 gcc/clang 下分别补一组类型注册回归，验证 long 同构分支行为。
