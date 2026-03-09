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
- Place的验证可以取HashValue()
- Device的比较可以取str()
- 如果./test/result_cmp.sh的对比结果有差异，请记录下来，在最后总结告诉我，不需要修改测试代码


### 仅运行单个测试

```bash
./torch/torch_AbsTest --gtest_filter="AbsTest.EdgeValues"
```

#### 运行对比脚本

```bash
cd .. && ./test/result_cmp.sh build
```

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
