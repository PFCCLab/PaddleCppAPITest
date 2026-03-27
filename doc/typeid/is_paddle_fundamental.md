## typeid.h - detail::is_paddle_fundamental 类型萃取兼容性

对比文件：
- `/home/may/Paddle/paddle/phi/api/include/compat/c10/util/typeid.h`
- `/home/may/pytorch/c10/util/typeid.h`

状态说明：
- `✅` 已实现（接口存在且签名/语义基本一致）
- `🔧` 部分兼容（接口存在，但签名或实现语义有差异）
- `❌` 未实现（PyTorch 有，Paddle compat 头文件无）

---

### 类职责

`caffe2::detail::is_paddle_fundamental<T>` 用于决定某类型是否按“fundamental 类型”处理，从而在 `_PickPlacementNew/_PickCopy/_PickPlacementDelete` 中跳过构造/析构/逐元素赋值。

Torch 对应能力来自 `c10::guts::is_fundamental<T>` 及其特化。

---

### 模板与特化

| torch API | paddle API 兼容性 | 测试用例状态 | 优先级 | 备注 |
|---|---|---|---|---|
| `c10::guts::is_fundamental<T>`（基模板） | `is_paddle_fundamental<T> : std::is_fundamental<T>` | ✅ | - [ ] | P1 | 实现一致性：`🔧`；语义：一致；实现：命名空间与命名不同，但都以 `std::is_fundamental` 为基线。 |
| `is_fundamental<at::Half>` 特化 | `is_paddle_fundamental<at::Half>` 特化 | ✅ | - [ ] | P0 | 实现一致性：`✅`；语义：一致；实现：都将 `at::Half` 视为 fundamental。 |
| `is_fundamental<at::BFloat16>` 特化 | `is_paddle_fundamental<at::BFloat16>` 特化 | ✅ | - [ ] | P1 | 实现一致性：`🔧`；语义：基本一致；实现：Paddle 显式特化，Torch 该能力通常来自其他 traits 组合。 |
| `Float8` 相关 fundamental 规则 | `is_paddle_fundamental<c10::Float8_e4m3fn/e5m2>` | 🔧 | - [ ] | P1 | 实现一致性：`🔧`；语义：基本一致；实现：Paddle 显式覆盖 Float8 类型，Torch 路径与版本相关。 |

---

### 结论

1. 该 traits 的“行为目标”与 Torch 一致：让低精度标量在 TypeMeta helper 中按 POD 路径处理。
2. 主要差异是命名与特化集合，属于实现组织差异。
3. 对兼容回归最关键的是验证：Half/BFloat16/Float8 在 `new/copy/delete` 指针选择上是否与预期一致。
