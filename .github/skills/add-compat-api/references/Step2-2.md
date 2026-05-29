# Step 2 参考：第2步实现路径选择

本文用于补充 Step 2 的第 2 步：根据接口类别选择实现路径，避免在不合适的层面重复造轮子。

## 第2步：按接口类型分流实现

### A. `ATen/ops` 相关实现

适用范围：`/home/may/Paddle/paddle/phi/api/include/compat/ATen/ops` 下的接口。

处理原则：
- 优先在 `paddle::experimental` 命名空间查找对应 op。
- 先复用已有 `paddle::experimental::<op>` 语义，再做 compat 层参数/返回适配。
- 不直接复制 PyTorch 的底层实现细节，优先复用 Paddle 现有算子能力。

建议检查项：
- 参数顺序和默认值是否对齐
- dtype/device/layout 等选项语义是否一致
- 返回值结构（单 Tensor / 多返回）是否一致
- 异常触发条件是否一致

可用检索示例：

```bash
rg -n "paddle::experimental::<op>|experimental::<op>" "$PADDLE_ROOT"
rg -n "<op>\(" "$PADDLE_ROOT/paddle/phi/api/include/compat/ATen/ops"
```

### B. `Device`、`ScalarType` 等基础设施实现

适用范围：`Device`、`DeviceType`、`ScalarType`、类型转换、基础工具类等。

处理原则：
- 参考 PyTorch 对应实现进行语义适配。
- **宏定义要完全对齐**（条件分支、平台宏、功能开关宏的行为需一致）。
- 类和函数不要求逐字逐结构对齐，可按 Paddle 架构做适配封装。

关键约束：
- 宏语义一致性优先于类结构一致性。
- 可以调整内部实现方式，但外部可观察语义必须与 PyTorch 对齐。

建议检查项：
- `#if/#ifdef` 分支条件是否与 PyTorch 同语义
- 不同后端（CPU/CUDA/XPU/HIP）下行为是否一致
- 标量类型映射和边界值处理是否一致

### C. `Storage`、`DataPtr` 等底层张量基础实现

适用范围：`Storage`、`DataPtr`、intrusive_ptr、底层内存持有/释放相关逻辑。

处理原则：
- 当前这部分已具备相对完备实现，默认不做直接重构。
- 如确需改动，必须进入**人工审核**流程。

人工审核前置条件：
- 明确变更动机与风险（生命周期、所有权、别名、线程安全）
- 给出与 PyTorch 行为差异的可复现实验
- 提供最小修改方案与回滚方案

人工审核检查项：
- 所有权与释放路径是否安全（含 deleter）
- 拷贝/移动/共享语义是否被破坏
- device index、place、stream 语义是否回归
- 是否影响现有兼容测试稳定性

## 决策总结

- `ATen/ops`：先找 `paddle::experimental` 复用算子能力。
- 基础设施（`Device`/`ScalarType`）：按 PyTorch 语义适配，宏定义完全对齐。
- 底层张量（`Storage`/`DataPtr`）：默认不动，必须人工审核后再改。

## 实现后检查

新增或修改的 compat 头文件与测试文件完成后，必须确认以下三项，否则下游代码无法使用新接口或 CI/ctest 不会编译运行该用例：

1. **头文件自包含**：compat 头文件应直接 `#include` 其依赖的声明（如调用了 `at::cat`，就必须显式包含 `ATen/ops/cat.h`），不能依赖其他头文件的 include 顺序来隐式引入。
2. **聚合头同步暴露**：新增 `ATen/ops/<op>.h` 后，必须同步在聚合头 `ATen/Functions.h` 中加入对应的 `#include <ATen/ops/<op>.h>`。只包含 `<ATen/ATen.h>` 或 `<ATen/Functions.h>` 的下游 C++ 扩展依赖聚合头暴露全部接口，漏掉 include 会导致下游编译时找不到新接口。
3. **测试接入 CMake**：`test/cpp/compat/CMakeLists.txt` 是手动列举 `cc_test(...)` / `nv_test(...)` 目标的，每新增一个测试文件必须同步注册。详见 [Step2-3.md](Step2-3.md)。
