##### OpMathType.h 头文件 API 兼容性

对比文件：
- `/home/may/Paddle/paddle/phi/api/include/compat/ATen/OpMathType.h`
- `/home/may/pytorch/aten/src/ATen/OpMathType.h`

状态说明：
- `✅` 已实现（接口存在且行为一致）
- `🔧` 部分兼容（接口存在，但覆盖范围或作用域有差异）
- `❌` 未实现（PyTorch 有，Paddle compat 头文件无）

---

### 核心模板与别名

| torch API | paddle API 兼容性 | 备注 |
|---|---|---|
| `template <typename scalar_t> struct OpMathType` | ✅ | 主模板一致，默认 `type = scalar_t` |
| `OpMathType<at::Half>` | ✅ | 映射到 `float` |
| `OpMathType<at::BFloat16>` | ✅ | 映射到 `float` |
| `OpMathType<at::Float8_e5m2>` | ✅ | 映射到 `float` |
| `OpMathType<at::Float8_e4m3fn>` | ✅ | 映射到 `float` |
| `OpMathType<c10::complex<Half>>` | ✅ | 映射到 `c10::complex<float>` |
| `using opmath_type = typename OpMathType<T>::type` | ✅ | 类型别名一致 |

---

### Float8 扩展覆盖

| torch API | paddle API 兼容性 | 备注 |
|---|---|---|
| `OpMathType<at::Float8_e5m2fnuz>` | ❌ | Paddle compat 缺失 |
| `OpMathType<at::Float8_e4m3fnuz>` | ❌ | Paddle compat 缺失 |
| `OpMathType<at::Float8_e8m0fnu>` | ❌ | Paddle compat 缺失 |

---

### `toOpMathType` 映射函数

| torch API | paddle API 兼容性 | 备注 |
|---|---|---|
| `toOpMathType(c10::ScalarType)` | 🔧 | 两侧逻辑一致（`AT_FORALL_SCALAR_TYPES_WITH_COMPLEX`）；但 PyTorch 放在匿名命名空间，Paddle 放在 `at` 命名空间 |

---

### 头文件依赖差异

| 项目 | paddle API 兼容性 | 备注 |
|---|---|---|
| `#include <c10/util/Float8_e4m3fn.h>` | ✅ | 已包含 |
| `#include <c10/util/Float8_e5m2.h>` | ✅ | 已包含 |
| `#include <c10/util/Float8_e4m3fnuz.h>` | ❌ | PyTorch 包含，Paddle 未包含 |
| `#include <c10/util/Float8_e5m2fnuz.h>` | ❌ | PyTorch 包含，Paddle 未包含 |

---

### 兼容性统计（基于以上条目）

| 状态 | 数量 |
|---|---|
| ✅ 已实现 | 10 |
| 🔧 部分兼容 | 1 |
| ❌ 未实现 | 5 |

---

### 结论

- `OpMathType` 的核心行为在 Paddle compat 中已对齐：`Half/BFloat16/常见 Float8` 都能提升到 `float` 做内部计算。
- 主要缺口是 PyTorch 新增的 3 个 Float8 类型特化（`e5m2fnuz/e4m3fnuz/e8m0fnu`）及其头文件依赖。
- `toOpMathType` 的实现逻辑一致，但作用域组织不同：Paddle 暴露在 `at` 命名空间，PyTorch 则在匿名命名空间内做内部可见性约束。
