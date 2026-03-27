## Layout.h 头文件 API 兼容性

对比文件：
- `/home/may/Paddle/paddle/phi/api/include/compat/c10/core/Layout.h`
- `/home/may/pytorch/c10/core/Layout.h`

状态说明：
- `✅` 已实现（接口存在且签名/语义基本一致）
- `🔧` 部分兼容（接口存在，但签名或实现语义有差异）
- `❌` 未实现（PyTorch 有，Paddle compat 头文件无）

---

### Layout 枚举类型

| torch API                    | paddle API 兼容性 | 测试用例状态 | 优先级 | 备注 |
|------------------------------|------------------|------------|-------|------|
| `Layout::Strided`            | ✅               | ✅          |   P0  | 标准密集布局 |
| `Layout::Sparse`             | ✅               | ✅          |   P2  | COO 稀疏布局 |
| `Layout::SparseCsr`          | ✅               | ✅          |   P2  | CSR 稀疏布局 |
| `Layout::Mkldnn`             | ✅               | ✅          |   P3  | MKLDNN 布局 |
| `Layout::SparseCsc`          | ✅               | ✅          |   P3  | CSC 稀疏布局 |
| `Layout::SparseBsr`          | ✅               | ✅          |   P3  | BSR 稀疏布局 |
| `Layout::SparseBsc`          | ✅               | ✅          |   P3  | BSC 稀疏布局 |
| `Layout::Jagged`             | ✅               | ✅          |   P3  | 不规则布局 |
| `Layout::NumOptions`         | ✅               | ✅          |   P3  | 布局选项数量 |

---

### 布局常量

| torch API                    | paddle API 兼容性 | 测试用例状态 | 优先级 | 备注 |
|------------------------------|------------------|------------|-------|------|
| `kStrided`                   | ✅               | ✅          |   P0  | = Layout::Strided |
| `kSparse`                    | ✅               | ✅          |   P2  | = Layout::Sparse |
| `kSparseCsr`                 | ✅               | ✅          |   P2  | = Layout::SparseCsr |
| `kMkldnn`                    | ✅               | ✅          |   P3  | = Layout::Mkldnn |
| `kSparseCsc`                 | ✅               | ✅          |   P3  | = Layout::SparseCsc |
| `kSparseBsr`                 | ✅               | ✅          |   P3  | = Layout::SparseBsr |
| `kSparseBsc`                 | ✅               | ✅          |   P3  | = Layout::SparseBsc |
| `kJagged`                    | ✅               | ✅          |   P3  | = Layout::Jagged |

---

### 辅助函数

| torch API                    | paddle API 兼容性 | 测试用例状态 | 优先级 | 备注 |
|------------------------------|------------------|------------|-------|------|
| `operator<<(ostream, Layout)` | ✅              | ✅          |   P1  | 输出流运算符 |
| `layout_from_backend()`      | - [ ]            | - [ ]       |   P3  | 从后端推断布局 |

---

### 命名空间导出

| 命名空间 | 导出内容 | 状态 |
|---------|---------|------|
| `c10`   | Layout 枚举、所有常量 | ✅ |
| `at`    | Layout 枚举、所有常量 | ✅ |
| `torch` | Layout 枚举、所有常量 | ✅ |

---

### 兼容性统计

| 状态 | 数量 |
|---|---|
| ✅ 已实现 | 18 |
| 🔧 部分兼容 | 0 |
| ❌ 未实现 | 1 |

---

### 备注

1. **优先级说明**：
   - P0: 核心功能，必须支持
   - P1: 常用功能，高优先级
   - P2: 进阶功能，中优先级
   - P3: 边缘功能，低优先级

2. **实现说明**：
   - Layout 枚举与 LibTorch 完全一致
   - 输出流运算符支持所有布局类型的字符串表示
   - `layout_from_backend()` 未实现，因为 Paddle 使用不同的后端抽象

3. **使用建议**：
   - 大多数场景使用 `kStrided`（默认密集布局）
   - 稀疏张量相关布局需要配合稀疏张量 API 使用
