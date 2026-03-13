##### DispatchKeySet.h 头文件 API 兼容性

✅ 表示已经支持
🚧 表示正在支持
❌ 表示不准备支持
🔧 表示部分支持（有功能限制）

**按照功能分类排序**

---

### 结构体与偏移初始化

| torch API | paddle API 兼容性 | 测试用例状态 | 优先级 | 备注 |
|-----------|------------------|--------------|--------|------|
| `FunctionalityOffsetAndMask` | ✅ | - [ ] | P1 | 结构体定义一致 |
| `initializeFunctionalityOffsetsAndMasks()` | ✅ | - [ ] | P1 | 均提供声明（实现位于 `.cpp`） |
| `offsetsAndMasks()` | ✅ | - [ ] | P1 | 均为静态缓存访问接口 |

---

### DispatchKeySet 构造与表示

| torch API | paddle API 兼容性 | 测试用例状态 | 优先级 | 备注 |
|-----------|------------------|--------------|--------|------|
| `DispatchKeySet()` | ✅ | - [ ] | P0 | 默认空集合 |
| `DispatchKeySet(Full)` | ✅ | - [ ] | P1 | 全功能 key 集构造 |
| `DispatchKeySet(FullAfter, DispatchKey)` | ✅ | - [ ] | P1 | 语义一致 |
| `DispatchKeySet(Raw, uint64_t)` | ✅ | - [ ] | P1 | 原始位表示构造 |
| `DispatchKeySet(BackendComponent)` | ✅ | - [ ] | P1 | 后端位构造 |
| `DispatchKeySet(DispatchKey)` | ✅ | - [ ] | P0 | 运行时 key/功能 key 转位集 |
| `DispatchKeySet(initializer_list<DispatchKey>)` | ✅ | - [ ] | P1 | 列表构造 |
| `DispatchKeySet(initializer_list<BackendComponent>)` | ✅ | - [ ] | P2 | 列表构造 |
| `raw_repr()` | ✅ | - [ ] | P1 | 返回底层位表示 |
| `from_raw_repr(uint64_t)` | ✅ | - [ ] | P1 | 从位表示恢复 |

---

### 成员查询与集合操作

| torch API | paddle API 兼容性 | 测试用例状态 | 优先级 | 备注 |
|-----------|------------------|--------------|--------|------|
| `has(DispatchKey)` | ✅ | - [ ] | P0 | 均支持；断言宏不同 |
| `has_backend(BackendComponent)` | ✅ | - [ ] | P1 | 一致 |
| `has_all(DispatchKeySet)` | ✅ | - [ ] | P0 | 一致 |
| `has_any(DispatchKeySet)` | ✅ | - [ ] | P0 | 一致；内部断言宏不同 |
| `isSupersetOf(DispatchKeySet)` | ✅ | - [ ] | P1 | 一致 |
| `operator|` | ✅ | - [ ] | P0 | 并集 |
| `operator&` | ✅ | - [ ] | P0 | 交集 |
| `operator-` | ✅ | - [ ] | P0 | 差集（仅移除功能位） |
| `operator^` | ✅ | - [ ] | P1 | 异或 |
| `operator==` | ✅ | - [ ] | P1 | 一致 |
| `operator!=` | ✅ | - [ ] | P1 | 一致 |
| `add(DispatchKey)` | ✅ | - [ ] | P1 | 一致 |
| `add(DispatchKeySet)` | ✅ | - [ ] | P1 | 一致 |
| `remove(DispatchKey)` | ✅ | - [ ] | P1 | 一致 |
| `remove_backend(BackendComponent)` | ✅ | - [ ] | P1 | 一致 |
| `empty()` | ✅ | - [ ] | P0 | 一致 |

---

### 优先级与索引计算

| torch API | paddle API 兼容性 | 测试用例状态 | 优先级 | 备注 |
|-----------|------------------|--------------|--------|------|
| `highestFunctionalityKey()` | ✅ | - [ ] | P0 | 一致 |
| `highestBackendKey()` | ✅ | - [ ] | P0 | 一致 |
| `highestPriorityTypeId()` | ✅ | - [ ] | P0 | 一致 |
| `indexOfHighestBit()` | 🔧 | - [ ] | P1 | Paddle 使用编译器内建/平台分支，PyTorch 使用 `llvm::countLeadingZeros` |
| `getDispatchTableIndexForDispatchKeySet()` | ✅ | - [ ] | P0 | 移动端/非移动端逻辑均保留 |
| `getBackendIndex()` | ✅ | - [ ] | P1 | 一致 |
| `getDispatchTableIndexForDispatchKey(DispatchKey)` | ✅ | - [ ] | P1 | 一致 |

---

### 迭代器接口

| torch API | paddle API 兼容性 | 测试用例状态 | 优先级 | 备注 |
|-----------|------------------|--------------|--------|------|
| `DispatchKeySet::iterator` | ✅ | - [ ] | P1 | 结构与语义一致 |
| `iterator::operator++()` | ✅ | - [ ] | P1 | 均仅声明，定义在源文件 |
| `iterator::operator*()` | ✅ | - [ ] | P1 | 一致 |
| `begin()` | ✅ | - [ ] | P1 | 一致 |
| `end()` | ✅ | - [ ] | P1 | 一致 |

---

### 字符串与流输出

| torch API | paddle API 兼容性 | 测试用例状态 | 优先级 | 备注 |
|-----------|------------------|--------------|--------|------|
| `toString(DispatchKeySet)` | ✅ | - [ ] | P2 | 均提供声明 |
| `operator<<(ostream&, DispatchKeySet)` | ✅ | - [ ] | P2 | 均提供声明 |

---

### 预定义 keyset 常量

| torch API | paddle API 兼容性 | 测试用例状态 | 优先级 | 备注 |
|-----------|------------------|--------------|--------|------|
| `autograd_dispatch_keyset` | ✅ | - [ ] | P1 | 一致 |
| `autocast_dispatch_keyset` | ✅ | - [ ] | P1 | 一致 |
| `default_included_set` | ✅ | - [ ] | P1 | 一致 |
| `default_excluded_set` | ✅ | - [ ] | P1 | 一致 |
| `autograd_dispatch_keyset_with_ADInplaceOrView` | ✅ | - [ ] | P1 | 一致 |
| `python_ks` | ✅ | - [ ] | P2 | 一致 |
| `sparse_ks` | ✅ | - [ ] | P2 | 一致 |
| `sparse_csr_ks` | ✅ | - [ ] | P2 | 一致 |
| `mkldnn_ks` | ✅ | - [ ] | P2 | 一致 |

---

### PyTorch 存在但 Paddle 头文件未提供的全局 API

| torch API | paddle API 兼容性 | 测试用例状态 | 优先级 | 备注 |
|-----------|------------------|--------------|--------|------|
| `autogradother_backends` | ❌ | - [ ] | P2 | Paddle 头文件未声明 |
| `after_autograd_keyset` | ❌ | - [ ] | P2 | 未声明 |
| `after_ADInplaceOrView_keyset` | ❌ | - [ ] | P2 | 未声明 |
| `after_func_keyset` | ❌ | - [ ] | P2 | 未声明 |
| `backend_bitset_mask` | ❌ | - [ ] | P2 | 未声明 |
| `inplace_or_view_ks` 等 `autograd_*_ks` 常量 | ❌ | - [ ] | P2 | 未声明 |
| `functorch_transforms_ks` / `functorch_batched_ks` | ❌ | - [ ] | P3 | 未声明 |
| `backend_functionality_keys` | ❌ | - [ ] | P2 | 未声明 |
| `OpTableOffsetAndMask` | ❌ | - [ ] | P3 | 未声明 |
| `isBackendDispatchKey(DispatchKey)` | ❌ | - [ ] | P1 | 未声明 |
| `getRuntimeDispatchKeySet(DispatchKey)` | ❌ | - [ ] | P1 | 未声明 |
| `runtimeDispatchKeySetHas(DispatchKey, DispatchKey)` | ❌ | - [ ] | P1 | 未声明 |
| `getBackendKeySetFromAutograd(DispatchKey)` | ❌ | - [ ] | P1 | 未声明 |
| `getAutogradRelatedKeySetFromBackend(BackendComponent)` | ❌ | - [ ] | P1 | 未声明 |
| `getAutocastRelatedKeySetFromBackend(BackendComponent)` | ❌ | - [ ] | P1 | 未声明 |
| `highestPriorityBackendTypeId(DispatchKeySet)` | ❌ | - [ ] | P1 | 未声明 |
| `isIncludedInAlias(DispatchKey, DispatchKey)` | ❌ | - [ ] | P2 | 未声明 |
| `legacyExtractDispatchKey(DispatchKeySet)` | ❌ | - [ ] | P1 | 未声明 |
| `is_not_DispatchKeySet` | ❌ | - [ ] | P3 | 未声明 |
| `remove_DispatchKeySet_arg_from_func` | ❌ | - [ ] | P3 | 未声明 |

---

### 兼容性统计

| 状态 | 数量 |
|------|------|
| ✅ 已完全支持 | 53 |
| 🚧 正在支持 | 0 |
| 🔧 部分支持 | 1 |
| ❌ 未支持 | 20 |

---

### 备注

1. **对比文件**：
   - Paddle: `/home/may/Paddle/paddle/phi/api/include/compat/c10/core/DispatchKeySet.h`
   - PyTorch: `/home/may/pytorch/c10/core/DispatchKeySet.h`

2. **核心结论**：
   - `DispatchKeySet` 核心类（构造、集合操作、优先级查询、迭代器）在 Paddle 兼容层中整体保持一致。
   - 差异主要集中在 PyTorch 文件尾部的一批全局常量、alias 解析辅助函数、以及模板元编程工具类型，Paddle 头文件当前未暴露。
   - `indexOfHighestBit()` 实现策略不同，但接口与语义兼容。

3. **测试现状**：
   - 当前仓库未检索到 `DispatchKeySet` 相关独立测试文件；表中测试状态暂标记为 `- [ ]`。
