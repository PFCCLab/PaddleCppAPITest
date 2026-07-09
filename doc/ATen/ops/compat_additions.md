# ATen Ops 兼容接口新增记录

## 对齐迭代记录（2026-05-28）

### 1) 接口变更
- 接口名：`at::column_stack`
- 变更类型：新增
- Paddle 兼容层位置：`paddle/phi/api/include/compat/ATen/ops/column_stack.h`
- 参考 PyTorch 位置：`aten/src/ATen/native/TensorShape.cpp`

### 2) 测试覆盖
- 测试文件：`test/ATen/ops/ColumnStackTest.cpp`
- Paddle 回归测试：`test/cpp/compat/ATen_column_stack_test.cc`
- 新增/修改用例：
  - 基础功能：Basic1D, Basic2D, Mixed1DAnd2D, ScalarTensors, SingleTensor
  - Shape 覆盖：LargeShape, ZeroDim, AllOneShape
  - Dtype 覆盖：DtypeFloat, DtypeDouble, DtypeInt, DtypeLong
  - 异常行为：EmptyList, MismatchedRows

### 3) 新增接口验证结果
- 新增前状态（缺失）：Paddle compat 层无 `at::column_stack` 实现
- 新增后验证结果：Paddle 与 Torch 输出完全一致（MATCH）
- 关键行为说明：
  - 1D tensor 自动 reshape 为 (n, 1) 后按 dim=1 拼接
  - 0D (scalar) tensor reshape 为 (1, 1)
  - 2D+ tensor 直接按 dim=1 拼接
  - 空列表抛出异常

### 4) 构建与回归结果
- Paddle 编译：通过
- ctest (ATen|c10|torch)：69/69 通过
- result_cmp：ColumnStackTest MATCH，无新增差异

### 5) 未完成项与下一轮计划
- 未完成接口：无
- 下一轮优先级：无

---

## 兼容性统计

| 状态 | 数量 |
|---|---|
| ✅ 已实现 | 1 |
| 🔧 部分兼容 | 0 |
| ❌ 未实现 | 0 |

---

## API 对比表

| torch API | paddle API 兼容性 | 测试用例状态 | 优先级 | 备注 |
|-----------|------------------|------------|-------|------|
| `at::column_stack(TensorList)` | ✅ | - [x] | P0 | 与 PyTorch 语义一致 |

---

## 关键差异说明

无差异。`at::column_stack` 在 Paddle compat 层的行为与 PyTorch 完全一致。
