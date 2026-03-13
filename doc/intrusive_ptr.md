##### intrusive_ptr.h 头文件 API 兼容性

✅ 表示已经支持
🚧 表示正在支持
❌ 表示不准备支持
🔧 表示部分支持（有功能限制）

**按照功能分类排序**

---

### 核心类型

| torch API | paddle API 兼容性 | 测试用例状态 | 优先级 | 备注 |
|-----------|------------------|--------------|--------|------|
| `intrusive_ptr_target` | ❌ | - [ ] | P0 | Paddle 兼容头未提供 intrusive refcount 基类 |
| `intrusive_ptr<T, NullType>` | 🔧 | - [ ] | P0 | Paddle 仅提供 `intrusive_ptr<T>`（基于 `std::shared_ptr`） |
| `weak_intrusive_ptr<T, NullType>` | ❌ | - [ ] | P1 | 未提供弱引用 intrusive 接口 |
| `weak_intrusive_ptr_target` | ❌ | - [ ] | P2 | 未提供别名类型 |

---

### intrusive_ptr 构造与赋值

| torch API | paddle API 兼容性 | 测试用例状态 | 优先级 | 备注 |
|-----------|------------------|--------------|--------|------|
| `intrusive_ptr()` | ✅ | - [ ] | P0 | 支持 |
| `intrusive_ptr(nullptr_t)` | 🔧 | - [ ] | P1 | Paddle 未显式提供该重载，但默认构造可替代 |
| `intrusive_ptr(T*)` | ✅ | - [ ] | P0 | 支持 |
| `intrusive_ptr(std::shared_ptr<T>)` | 🔧 | - [ ] | P1 | Paddle 特有；PyTorch 原生接口无该构造 |
| 跨类型拷贝构造（`intrusive_ptr<U> -> intrusive_ptr<T>`） | ✅ | - [ ] | P1 | 支持可转换类型 |
| 移动构造 / 移动赋值 | 🔧 | - [ ] | P1 | Paddle 依赖编译器生成；无显式 NullType 变体 |
| 拷贝赋值 | ✅ | - [ ] | P1 | 支持 |

---

### intrusive_ptr 观察器与基础操作

| torch API | paddle API 兼容性 | 测试用例状态 | 优先级 | 备注 |
|-----------|------------------|--------------|--------|------|
| `get()` | ✅ | - [ ] | P0 | 支持 |
| `operator*()` / `operator->()` | ✅ | - [ ] | P0 | 支持 |
| `operator bool()` | ✅ | - [ ] | P1 | 支持（Paddle 为 `explicit`） |
| `defined()` | ✅ | - [ ] | P0 | 支持 |
| `use_count()` | 🔧 | - [ ] | P1 | Paddle 返回 `int64_t`；PyTorch 返回 `uint32_t` |
| `weak_use_count()` | ❌ | - [ ] | P1 | 未支持 |
| `unique()` | ❌ | - [ ] | P1 | 未支持 |
| `is_uniquely_owned()` | ❌ | - [ ] | P1 | 未支持 |
| `reset()` | ✅ | - [ ] | P0 | 支持 |
| `swap(intrusive_ptr&)` | ❌ | - [ ] | P2 | 未提供成员函数 |

---

### 所有权转移与不安全适配

| torch API | paddle API 兼容性 | 测试用例状态 | 优先级 | 备注 |
|-----------|------------------|--------------|--------|------|
| `release()` | 🔧 | - [ ] | P0 | Paddle 实现基于 `shared_ptr::reset`，语义与 PyTorch“保留 owning raw ptr”不一致 |
| `reclaim(T*)` | ❌ | - [ ] | P0 | 未支持 |
| `reclaim_copy(T*)` | ❌ | - [ ] | P1 | 未支持 |
| `unsafe_steal_from_new(T*)` | ❌ | - [ ] | P2 | 未支持 |
| `unsafe_adapt_non_heap_allocated(T*, uint32_t)` | ❌ | - [ ] | P2 | 未支持 |
| `unsafe_reclaim_from_nonowning(T*)` | ❌ | - [ ] | P2 | 未支持 |

---

### 工厂函数与辅助接口

| torch API | paddle API 兼容性 | 测试用例状态 | 优先级 | 备注 |
|-----------|------------------|--------------|--------|------|
| `intrusive_ptr::make(args...)` | ✅ | - [ ] | P1 | 支持 |
| `make_intrusive<T>(args...)` | ✅ | - [ ] | P0 | 支持 |
| `get_shared()` | 🔧 | - [ ] | P2 | Paddle 特有（用于 shared_ptr 互操作） |

---

### 全局运算符与容器支持

| torch API | paddle API 兼容性 | 测试用例状态 | 优先级 | 备注 |
|-----------|------------------|--------------|--------|------|
| `operator==/!= (intrusive_ptr, intrusive_ptr)` | ✅ | - [ ] | P1 | 支持 |
| `operator==/!= (intrusive_ptr, nullptr)` | ❌ | - [ ] | P2 | 未提供显式全局重载 |
| `operator< (intrusive_ptr, intrusive_ptr)` | ❌ | - [ ] | P2 | 未支持 |
| `swap(intrusive_ptr&, intrusive_ptr&)` | ❌ | - [ ] | P2 | 未支持 |
| `std::hash<intrusive_ptr<...>>` | ❌ | - [ ] | P3 | 未支持 |

---

### weak_intrusive_ptr 与 raw 命名空间工具

| torch API | paddle API 兼容性 | 测试用例状态 | 优先级 | 备注 |
|-----------|------------------|--------------|--------|------|
| `weak_intrusive_ptr` 全部接口（`lock/release/reclaim/...`） | ❌ | - [ ] | P1 | Paddle 兼容头未实现 |
| `raw::intrusive_ptr::incref/decref/make_weak/use_count` | ❌ | - [ ] | P1 | 未支持 |
| `raw::weak_intrusive_ptr::incref/decref/lock/use_count` | ❌ | - [ ] | P1 | 未支持 |

---

### Traits 与元编程接口

| torch API | paddle API 兼容性 | 测试用例状态 | 优先级 | 备注 |
|-----------|------------------|--------------|--------|------|
| `detail::TargetTraits` | ❌ | - [ ] | P3 | 未支持 |
| `MaybeOwnedTraits<c10::intrusive_ptr<T>>` | ❌ | - [ ] | P2 | 未支持 |
| `raw::DontIncreaseRefcount` | ❌ | - [ ] | P2 | 未支持 |

---

### 兼容性统计

| 状态 | 数量 |
|------|------|
| ✅ 已完全支持 | 12 |
| 🚧 正在支持 | 0 |
| 🔧 部分支持 | 8 |
| ❌ 未支持 | 23 |

---

### 备注

1. **对比文件**：
   - Paddle: `/home/may/Paddle/paddle/phi/api/include/compat/c10/util/intrusive_ptr.h`
   - PyTorch: `/home/may/pytorch/c10/util/intrusive_ptr.h`

2. **核心结论**：
   - Paddle 兼容层当前实现是 `std::shared_ptr` 包装版本，覆盖了最常用 `intrusive_ptr` 基本用法。
   - PyTorch 原生 `intrusive_ptr` 的核心优势（内嵌原子引用计数、`weak_intrusive_ptr`、`reclaim`/`unsafe_*`、`raw::*` 低层 API）在 Paddle 头文件中大多未提供。
   - 因实现模型不同，涉及“裸指针所有权转移”的接口语义不能直接等价映射。

3. **测试现状**：
   - 在当前仓库中未检索到 `intrusive_ptr`/`weak_intrusive_ptr` 相关测试文件或直接调用用例，测试状态暂标记为 `- [ ]`。
