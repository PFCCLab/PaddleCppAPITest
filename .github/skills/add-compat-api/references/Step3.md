# Step 3 参考：从声明追踪到 PyTorch 实现

本文用于展开 add-compat-api 中的 Step 3。

## 第1步：先在 libtorch 查声明，再定位真实实现

目标：先从 libtorch 头文件确认 API 入口，再追踪到 PyTorch 源码中的最终 C++ 实现。

### 1) 先在 libtorch 中定位声明

将目标接口名记为 `<api_name>`，先在 libtorch include 目录搜索：

```bash
rg -n "<api_name>\(" "$TORCH_DIR/include/ATen" "$TORCH_DIR/include/torch"
```

常见声明位置：
- `$TORCH_DIR/include/ATen/ops/*.h`
- `$TORCH_DIR/include/ATen/core/*.h`
- `$TORCH_DIR/include/torch/csrc/api/include/torch/*.h`

### 2) 判断“有实现”还是“仅声明/转发”

找到声明后，分两类：

- 情况 A：头文件里有明确函数体
  - 可直接把该函数体作为语义参考起点。

- 情况 B：只有声明，或转发到 dispatcher
  - 常见形态：`at::_ops::<op>::call(...)`、`redispatch(...)`。
  - 这通常表示该接口由 torchgen 生成包装层，不在该头文件里放最终业务实现。

对情况 B，继续执行第 3-6 步。

### 3) 在 PyTorch YAML 中找 schema 与 dispatch

在 PyTorch 源码中搜索：

```bash
rg -n "<api_name>" "$PYTORCH_ROOT/aten/src/ATen/native/native_functions.yaml"
```

在命中的 YAML 条目中提取：
- 函数 schema（func）
- dispatch 表（CPU/CUDA/Composite/Meta 等）
- 各后端对应的 kernel 名称

说明：
- PyTorch 会使用 `$PYTORCH_ROOT/torchgen`（例如 `/home/may/pytorch/torchgen`）解析 YAML，生成 libtorch 头文件与包装层。
- 因此 libtorch 中很多接口只有声明或轻量转发，真实实现需要继续沿 dispatcher 追踪。

### 4) 沿 dispatcher 链路追踪到 kernel 符号

把调用路径映射为：

1. 头文件声明/包装
2. `at::_ops::<op>::call`（生成代码）
3. dispatcher 注册路由
4. YAML dispatch 指向的后端 kernel 符号

再用 kernel 符号搜索真实实现：

```bash
rg -n "<kernel_symbol>\(" "$PYTORCH_ROOT/aten/src/ATen/native"
```

### 5) 仍未命中时，检查生成注册文件

有些场景下，先看生成的注册文件更快：

```bash
rg -n "<api_name>|<kernel_symbol>" "$PYTORCH_ROOT/build/aten/src/ATen"
```

重点关注 `Register*.cpp` 和生成的 ops 相关文件。

### 6) 在实现 compat 前，先落一份追踪记录

正式改 Paddle compat 之前，至少记录以下映射：

- libtorch 声明文件位置
- 声明是直接实现还是 dispatcher 转发
- YAML 条目与 dispatch 键
- PyTorch 最终 kernel 文件与函数

只有映射清晰后，才进入 compat 接口实现阶段。

## 快速检查清单

- 已先从 libtorch 声明开始查找
- 已确认是否属于 torchgen 生成转发
- 已定位 native_functions.yaml 的 schema 与 dispatch
- 已定位最终 C++ kernel 实现
- 未臆造上游不存在的实现
