# compatibility-testing

PaddlePaddle 与 PyTorch C++ API 兼容性测试开发规范。

## 触发条件

适用场景：
- 编写或扩展 `PaddleCppAPITest\test` 下的兼容性测试
- 验证 Paddle 兼容层与 PyTorch 对同一 API 的行为一致性
- 定位某个接口在两个框架间的输出差异

## 测试目标

**测试范围**：覆盖 `Paddle\paddle\phi\api\include\compat` 目录下**所有**接口，包括但不限于：

| 目录 | 接口类型 | 示例 |
|------|---------|------|
| `ATen/ops/` | ATen 算子 | `abs.h`, `sum.h`, `reshape.h`, `zeros.h` ... |
| `ATen/core/` | ATen 核心类型 | `Tensor.h`, `TensorBody.h`, `TensorAccessor.h` ... |
| `ATen/` | ATen 基础 | `Tensor.h`, `Device.h`, `DeviceGuard.h` ... |
| `c10/core/` | C10 核心 | `ScalarType.h`, `TensorOptions.h`, `Storage.h` ... |
| `c10/util/` | C10 工具 | `Optional.h`, `ArrayRef.h`, `Half.h` ... |
| `c10/cuda/` | C10 CUDA | `CUDAStream.h`, `CUDAGuard.h`, `CUDAException.h` ... |
| `torch/` | Torch 包装 | `all.h`, `cuda.h`, `extension.h` ... |
| `utils/` | 工具函数 | `scalar_type_conversion.h`, `int_array_ref_conversion.h` ... |

> `AbsTest.cpp`（位于 `test/ops/` 仅为示例）仅作为**参考**，展示测试文件结构和输出格式。

## 项目约定

- 构建系统通过 `CMakeLists.txt` 中的 `create_paddle_tests()` 函数同时生成 `torch_*` 和 `paddle_*` 两套可执行文件
- 测试二进制运行时自动以自身文件名命名输出文件（如 `torch_AbsTest.txt`），由 `main.cpp` 中的 `g_custom_param` 传递
- 结果对比依赖文本 diff，因此输出格式的确定性至关重要

## 测试文件结构

### 文件头与命名空间

测试文件统一位于 `PaddleCppAPITest\test`，与 compat 接口目录结构对应。参考以下结构（以 `AbsTest.cpp` 为示例）：

```cpp
#include <ATen/ATen.h>
#include <ATen/core/Tensor.h>
#include <ATen/ops/abs.h>          // 按需替换为目标算子头文件
#include <ATen/ops/zeros.h>        // 辅助构造用
#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "../../src/file_manager.h"

extern paddle_api_test::ThreadSafeParam g_custom_param;

namespace at {
namespace test {

using paddle_api_test::FileManerger;
using paddle_api_test::ThreadSafeParam;

class AbsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // 构造基准输入 tensor
  }
  at::Tensor test_tensor;
};

// 测试用例 ...

}  // namespace test
}  // namespace at
```

**关键约束**：
- 命名空间固定为 `at::test`，保证与 ATen 类型系统的直接可见性
- `g_custom_param` 是全局线程安全参数，存储当前运行的输出文件名，由 `main.cpp` 在 `RUN_ALL_TESTS()` 前注入
- 测试类命名格式 `<OpName>Test`，文件名与之一致

### 结果输出函数

每个测试文件包含一个静态输出函数，负责将 tensor 结果序列化到文件。该函数是跨框架对比的唯一数据源，格式必须确定且可复现：

```cpp
static void write_abs_result_to_file(FileManerger* file, const at::Tensor& result) {
  *file << std::to_string(result.dim()) << " ";
  *file << std::to_string(result.numel()) << " ";
  float* data = result.data_ptr<float>();
  for (int64_t i = 0; i < result.numel(); ++i) {
    *file << std::to_string(data[i]) << " ";
  }
}
```

注意：
- 第一个测试用例调用 `file.createFile()` 创建文件，后续用例调用 `file.openAppend()` 追加
- 对于多 dtype 支持的算子，需按 `result.scalar_type()` 分发到对应的 `data_ptr<T>()` 类型

## Shape 覆盖要求

测试 shape 的选择直接影响边界条件的暴露率。以下为四个必选维度区间，每个新算子测试须至少各取一例：

### 标量 (0-d tensor)
- `{}` — 零维标量，部分算子（如 `sum` 不指定 dim）的返回类型
- 注意：`{1}` 是 1-d tensor，**不是**标量

### 小 shape（元素数 ≤ 64）
- 典型值：`{4}`、`{2, 3}`、`{2, 3, 4}`
- 便于手工验证数值正确性

### 大 shape（元素数 ≥ 10000）
- 典型值：`{10000}`、`{100, 100}`、`{10, 20, 30, 40}`
- 主要暴露精度累积误差和内存布局差异

### 边界 shape
- 含零维度：`{0}`、`{2, 0}`、`{1, 0, 3}` — 验证空 tensor 语义
- 全一维度：`{1, 1, 1}` — 常触发 squeeze/broadcast 的特殊路径
- 经 `transpose()` / `as_strided()` 产生的非连续 tensor — 验证 stride 处理的正确性

## Dtype 覆盖要求

以下为 ATen 支持的标准标量类型，通过 `at::TensorOptions().dtype()` 或 shorthand 常量指定。新增测试至少需要覆盖 `kFloat`、`kDouble`、`kInt`、`kLong` 四种基础类型，其余按算子语义酌情补充：

| 标量类型 | ATen 常量 | C++ 对应类型 | 适用注意 |
|---------|-----------|-------------|---------|
| float32 | `at::kFloat` | `float` | 多数算子的默认 dtype |
| float64 | `at::kDouble` | `double` | 精度基准，常用于 reference 比较 |
| int32 | `at::kInt` | `int32_t` | 整型算子、索引 |
| int64 | `at::kLong` | `int64_t` | shape / dim 参数的底层类型 |
| int16 | `at::kShort` | `int16_t` | 较少使用，部分量化场景 |
| int8 | `at::kChar` | `int8_t` | 不要与 `kByte` (uint8) 混淆 |
| uint8 | `at::kByte` | `uint8_t` | 常见于图像数据 |
| bool | `at::kBool` | `bool` | 比较算子的返回类型 |

> Paddle 兼容层的 dtype 映射与 PyTorch 存在细微差异（例如默认 dtype 可能不同），输出对比时需关注此类隐式转换。

## 按算子类别的测试设计

下面按六大类说明每类接口的测试重点和必要覆盖项。

### 1. Creation Ops — `zeros`, `ones`, `empty`, `full`, `arange`, `linspace`, `from_blob` …

覆盖项：
- 默认 dtype vs 显式指定 dtype
- 各 shape 区间（标量 / 小 / 大 / 边界）
- `from_blob` 的外部内存生命周期（避免 dangling pointer）
- `empty` 不检查数据值，只验证 shape 和 dtype 一致

```cpp
TEST_F(CreationOpsTest, ZerosMultiShape) {
  at::Tensor scalar = at::zeros({}, at::kFloat);          // 0-d
  at::Tensor vec    = at::zeros({2, 3}, at::kFloat);      // small
  at::Tensor large  = at::zeros({1000, 1000}, at::kFloat);// large
  at::Tensor dbl    = at::zeros({3, 4}, at::kDouble);     // dtype
  at::Tensor itn    = at::zeros({2, 2}, at::kInt);        // int dtype
}
```

### 2. Math Ops — `abs`, `add`, `sub`, `mul`, `div`, `neg`, `pow`, `exp`, `log`, `sqrt` …

覆盖项：
- 值域分区：全正、全负、混合正负零、极大值 (`1e10`)、极小值 (`1e-10`)
- 特殊浮点值：`NaN`、`+Inf`、`-Inf`、`-0.0` 与 `+0.0` 的区分
- 原地变体：`at::abs_(t)` vs `at::abs(t)` — 原地操作的返回值是否为同一 storage
- `out=` 重载：`at::abs_out(output, input)` — 验证输出 tensor 被正确写入且 shape/dtype 匹配

```cpp
TEST_F(AbsTest, EdgeValues) {
  at::Tensor t = at::zeros({6}, at::kFloat);
  float* d = t.data_ptr<float>();
  d[0] = 1e10f;   d[1] = -1e10f;  // 极值
  d[2] = 1e-10f;  d[3] = -1e-10f; // 亚正常域附近
  d[4] = 0.0f;    d[5] = -0.0f;   // 正负零

  at::Tensor result = at::abs(t);
  // 写入文件 ...
}
```

### 3. Shape Ops — `reshape`, `view`, `flatten`, `transpose`, `squeeze`, `unsqueeze`, `narrow`, `expand` …

覆盖项：
- `view` 与 `reshape`：连续 tensor 上两者等价，非连续 tensor 上 `view` 抛异常而 `reshape` 不会 — 需验证此差异
- 负索引：`squeeze(-1)`、`narrow(0, -N, len)` 等
- `transpose` 后 tensor 的 `is_contiguous()` 返回 `false`，后续算子的行为可能变化
- `expand` 不分配新内存（stride 为 0），验证数据共享语义

### 4. Indexing Ops — `index`, `index_select`, `select`, `take`, `masked_select`, `gather`, `scatter` …

覆盖项：
- 整数索引、切片索引、布尔掩码索引三种路径
- 负索引：`select(0, -1)` 等价于最后一个元素
- 空索引（`masked_select` 所有元素不满足条件）→ 返回 0 元素 tensor
- 多维索引场景下 broadcast 规则的一致性

### 5. Comparison Ops — `eq`, `ne`, `gt`, `lt`, `ge`, `le`, `isfinite`, `isinf`, `isnan` …

覆盖项：
- 返回 dtype 固定为 `kBool`，无论输入 dtype
- `NaN` 的不等性：`NaN != NaN` 应为 `true`，`NaN == NaN` 应为 `false`
- `Inf` 与有限大数的比较
- 混合 dtype 输入（如 float vs double）时的隐式提升规则

### 6. Reduction Ops — `sum`, `mean`, `var`, `std`, `min`, `max`, `argmin`, `argmax`, `any`, `all` …

覆盖项：
- 全局归约（不指定 dim）vs 指定 dim 归约
- `keepdim=true/false` 对输出 shape 的影响
- `dtype` 参数：`at::sum(input, at::kDouble)` — 累加精度提升
- `out=` 重载：输出 tensor 预分配时的 shape 匹配要求
- 空 tensor 归约行为：`at::sum(at::zeros({0}, at::kFloat))` 应返回 `0`

```cpp
TEST_F(SumTest, DimVariants) {
  at::Tensor input = at::ones({3, 4, 5}, at::kFloat);

  at::Tensor all      = at::sum(input);                    // 标量 60.0
  at::Tensor dim0     = at::sum(input, {0}, false);        // shape {4, 5}
  at::Tensor dim01    = at::sum(input, {0, 1}, false);     // shape {5}
  at::Tensor keep     = at::sum(input, {0}, true);         // shape {1, 4, 5}
  at::Tensor as_dbl   = at::sum(input, at::kDouble);       // dtype 提升

  at::Tensor out = at::zeros({4, 5}, at::kFloat);
  at::sum_out(out, input, {0}, false);                     // out= 重载
}
```

## 异常行为测试

部分算子在非法输入下的异常行为可能在两个框架间存在差异（一个抛异常、另一个返回 NaN 或空 tensor）。此类差异需显式捕获并记录：

```cpp
TEST_F(SomeOpTest, InvalidInputHandling) {
  try {
    at::Tensor result = at::some_op(invalid_tensor);
    // 未抛异常 — 正常记录结果
    auto file_name = g_custom_param.get();
    FileManerger file(file_name);
    file.openAppend();
    write_someop_result_to_file(&file, result);
    file.saveFile();
  } catch (const c10::Error& e) {
    // ATen/c10 层异常
    auto file_name = g_custom_param.get();
    FileManerger file(file_name);
    file.openAppend();
    file << "c10::Error: " << e.what();
    file.saveFile();
  } catch (const std::exception& e) {
    auto file_name = g_custom_param.get();
    FileManerger file(file_name);
    file.openAppend();
    file << "exception: " << e.what();
    file.saveFile();
  }
}
```

> 捕获时优先匹配 `c10::Error`（ATen 的标准异常类型），再兜底 `std::exception`。异常信息写入输出文件后可直接 diff，两框架的异常消息不要求完全一致，但**是否抛异常**须一致。

## 输出格式

输出文件采用空格分隔的纯文本，按以下字段顺序逐 tensor 追加：

```
<ndim> <numel> [<size_0> <size_1> ...] <val_0> <val_1> ...
```

示例（一个 shape 为 `{2, 3}` 的 float tensor）：
```
2 6 2 3 1.000000 2.000000 3.000000 4.000000 5.000000 6.000000
```

注意事项：
- 浮点值通过 `std::to_string()` 序列化，精度为 6 位有效数字
- 不同测试用例的输出依次追加到同一文件中，以换行或空格分隔，顺序由 GTest 的用例注册顺序决定

## 调试手段

### 运行时 tensor 状态检查

```cpp
std::cout << "shape: " << result.sizes()
          << "  dtype: " << result.scalar_type()
          << "  contiguous: " << result.is_contiguous()
          << "  device: " << result.device() << std::endl;
```

### 逐元素打印

```cpp
auto* data = result.data_ptr<float>();
for (int64_t i = 0; i < result.numel(); ++i) {
  std::cout << "[" << i << "] " << data[i] << "\n";
}
```

### GTest 断言附加信息

```cpp
EXPECT_EQ(result.dim(), 2) << "Unexpected rank for input shape " << input.sizes();
EXPECT_TRUE(result.is_contiguous()) << "Non-contiguous result from reshape";
```

### 仅运行单个测试

```bash
./torch/torch_AbsTest --gtest_filter="AbsTest.EdgeValues"
```

## 已知跨框架差异点

以下是实践中高频出现的 Paddle 与 PyTorch 行为差异，编写测试时需针对性设计用例：

| 差异点 | 表现 | 测试建议 |
|--------|------|---------|
| **默认 dtype** | PyTorch 默认 `kFloat`，Paddle 在某些路径下可能为 `kFloat64` | 所有 creation op 显式指定 dtype |
| **broadcast 语义** | 两框架均遵循 NumPy 风格 broadcast，但边界情况（如 `{0}` 参与 broadcast）行为可能不同 | 补充含零维度的 broadcast 用例 |
| **浮点精度** | 底层 BLAS/cuBLAS 实现差异导致 `sum` 等归约操作在大 shape 上存在 ULP 级别偏差 | 大 shape 归约测试可选容差对比 |
| **contiguous 约定** | `transpose` 后 tensor 的 stride 布局一致，但部分算子对非连续输入的 fast path 不同 | 全部算子补充非连续输入用例 |
| **异常语义** | 同一非法输入可能一侧抛 `c10::Error` 而另一侧返回空 tensor 或 NaN | 参照"异常行为测试"章节处理 |
| **原地操作返回值** | `abs_()` 是否返回当前 tensor 的引用 vs 新 tensor — 语义一致但指针可能不同 | 对比 `data_ptr` 是否相同 |
| **内存布局** | `as_strided` 系列 API 的 storage offset 计算差异 | 配合 `storage_offset()` 验证 |

## 新算子测试检查清单

新增测试前逐项确认，标注 `*` 的为强制项：

**Shape 维度**
- [ ] `*` 标量 (0-d tensor)
- [ ] `*` 小 shape (元素数 ≤ 64)
- [ ] `*` 大 shape (元素数 ≥ 10000)
- [ ] 含零维度 (`{0}`, `{2, 0}`)
- [ ] 全一维度 (`{1, 1, 1}`)
- [ ] 非连续 tensor (经 `transpose` / `narrow` / `as_strided`)

**Dtype**
- [ ] `*` float32
- [ ] `*` float64
- [ ] `*` int32
- [ ] `*` int64
- [ ] bool
- [ ] int8 / uint8 / int16（视算子支持情况）

**值域**
- [ ] `*` 正数
- [ ] `*` 负数
- [ ] `*` 零
- [ ] NaN / Inf / -Inf
- [ ] 极值 (`1e38f`, `1e-38f`)
- [ ] 正负零区分 (`+0.0` vs `-0.0`)

**API 变体**
- [ ] 函数式调用 (`at::abs(t)`)
- [ ] 原地操作 (`at::abs_(t)` 或 `t.abs_()`)
- [ ] out= 重载 (`at::abs_out(out, t)`)
- [ ] keepdim 参数（归约类算子）
- [ ] dim / axis 参数（含负索引）

**输出**
- [ ] `*` 第一个用例使用 `createFile()`，后续使用 `openAppend()`
- [ ] `*` 通过 `write_<op>_result_to_file()` 统一输出
- [ ] 多 dtype 场景按 `scalar_type()` 分发 `data_ptr<T>()`

## 输出文件路径

默认输出目录：`/tmp/paddle_cpp_api_test/`（由 `FileManerger::basic_path_` 控制）。

文件名自动取可执行文件名 + `.txt`：
- `torch_AbsTest` → `/tmp/paddle_cpp_api_test/torch_AbsTest.txt`
- `paddle_AbsTest` → `/tmp/paddle_cpp_api_test/paddle_AbsTest.txt`

如需自定义路径，在构造 `FileManerger` 时传入完整文件名即可覆盖（但通常不建议，以保持批量对比脚本的兼容性）。
