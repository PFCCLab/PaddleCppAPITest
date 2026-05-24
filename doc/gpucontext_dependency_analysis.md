# GPUContext 依赖全景分析

## 统计数据

| 维度 | 数量 |
|---|---|
| GPUContext 总引用 | 1699 处 |
| kernels/ 目录引用 | 1498 处 (88%) |
| core/ 目录引用 | 93 处 (5%) |
| backends/ 目录引用 | 87 处 (5%) |
| api/ 目录引用 | 15 处 (1%) |
| 直接 `dev_ctx.stream()` 调用 | 825 处 |

---

## 架构图：GPUContext 依赖全景

```mermaid
flowchart TB
    subgraph UserLayer["用户层 User Layer"]
        PyUser["Python User"]
        CppUser["C++ User (c10 compat)"]
    end

    subgraph PythonBind["Python Binding"]
        PyBindOp["pybind: Op Execution"]
        PyBindStream["pybind: CUDA Stream API"]
    end

    subgraph API_LAYER["phi/api/ 层 (15处引用)"]
        KernelDispatch["kernel_dispatch.cc"]
        TensorCopy["tensor_copy.cc"]
        TensorMethod["tensor_method.cc"]
        DataTransform["data_transform.cc"]
        ContextPool["context_pool.cc<br/>GetCurrentCUDAStream()"]
    end

    subgraph CORE_LAYER["phi/core/ 层 (93处引用)"]
        DeviceContextBase["DeviceContext (base)"]
        KernelRegistry["Kernel Registry"]
        DenseTensor["DenseTensor"]
    end

    subgraph BACKENDS_LAYER["phi/backends/ 层 (87处引用)"]
        GPUContextImpl["GPUContext::Impl<br/>stream_/eigen/cudnn/cublas"]
        DeviceContextPool["phi::DeviceContextPool<br/>(global singleton)"]
        CUDAStream["phi::CUDAStream<br/>(wraps raw cudaStream_t)"]
    end

    subgraph KERNEL_LAYER["phi/kernels/ 层 (1498处引用)"]
        direction TB
        subgraph KernelDispatchPattern["Kernel Dispatch Pattern"]
            K1["Kernel func(dev_ctx, inputs, outputs)"]
            K2["auto stream = dev_ctx.stream();"]
            K3["kernel<<<..., stream>>>(...);"]
        end

        subgraph KernelExamples["代表 Kernel (825处直接stream调用)"]
            E1["roi_pool_kernel.cu"]
            E2["interpolate_kernel.cu"]
            E3["weight_only_linear_kernel.cu"]
            E4["margin_cross_entropy_kernel.cu"]
            E5["... 800+ more"]
        end
    end

    subgraph C10_COMPAT["c10 compat 层"]
        C10Set["setCurrentCUDAStream(pool)"]
        C10Get["getCurrentCUDAStream()"]
    end

    PyUser --> PyBindOp
    CppUser --> C10Set

    PyBindOp --> KernelDispatch
    PyBindStream --> C10Set

    C10Set --> GPUContextImpl
    C10Get --> ContextPool

    KernelDispatch --> DeviceContextPool
    TensorCopy --> DeviceContextPool
    DataTransform --> DeviceContextPool

    DeviceContextPool --> GPUContextImpl
    ContextPool --> GPUContextImpl

    GPUContextImpl --> CUDAStream

    K1 --> K2 --> K3
    DeviceContextPool --> K1

    style KERNEL_LAYER fill:#f9f,stroke:#333,stroke-width:2px
    style GPUContextImpl fill:#ff9,stroke:#333,stroke-width:2px
```

---

## 关键依赖路径分析

### 路径 1：Python → phi kernel → GPUContext stream（主流路径）

```mermaid
sequenceDiagram
    autonumber
    participant Py as Python User
    participant Eager as Eager Dygraph
    participant APILib as phi/api/lib
    participant Kernel as phi/kernel
    participant GPUCtx as GPUContext
    participant CUDA as CUDA Driver

    Py->>Eager: tensor.op(inputs)
    Eager->>APILib: PrepareData(inputs)
    APILib->>APILib: DeviceContextPool::Get(place)
    APILib->>GPUCtx: return GPUContext*
    APILib->>Kernel: kernel(dev_ctx, inputs, outputs)
    Kernel->>GPUCtx: dev_ctx.stream()
    GPUCtx->>GPUCtx: impl_->stream()
    GPUCtx->>CUDA: cuda kernel launch on stream
```

### 路径 2：c10 compat → GPUContext stream（C++ API 路径）

```mermaid
sequenceDiagram
    autonumber
    participant User as C++ User
    participant C10 as c10::cuda
    participant Compat as compat CUDAStream.cpp
    participant GPUCtx as GPUContext
    participant Kernel as phi/kernel

    User->>C10: setCurrentCUDAStream(pool)
    C10->>Compat: setCurrentCUDAStream(pool)
    Compat->>GPUCtx: getMutableGPUContext(idx)
    Compat->>GPUCtx: SetStream(pool.stream())
    GPUCtx->>GPUCtx: impl_->SetStream(pool)
    Note over GPUCtx: stream_ = pool (全局修改!)

    User->>C10: at::Tensor::fill_()
    C10->>Kernel: ATen kernel dispatch
    Kernel->>GPUCtx: dev_ctx.stream()
    GPUCtx->>GPUCtx: impl_->stream()
    GPUCtx->>User: return pool (被阻塞的)
```

### 路径 3：DeviceContextPool 获取（全局单例问题）

```mermaid
sequenceDiagram
    autonumber
    participant T1 as Thread 1 (Main)
    participant T2 as Thread 2 (Worker)
    participant Pool as DeviceContextPool
    participant GPUCtx as GPUContext (global)

    T1->>Pool: Get(GPUPlace(0))
    Pool->>GPUCtx: return GPUContext*
    T1->>GPUCtx: SetStream(pool)
    Note over GPUCtx: stream_ = pool

    T2->>Pool: Get(GPUPlace(0))
    Pool->>GPUCtx: return 同一个 GPUContext*
    T2->>GPUCtx: stream()
    GPUCtx->>T2: return pool (Thread 1 设置的!)
    Note over T2: Worker 拿到了 Main 的阻塞 stream
```

---

## GPUContext 内部资源依赖

```mermaid
flowchart LR
    subgraph GPUContext["GPUContext::Impl"]
        S["stream_<br/>CUDAStream*"]
        E["eigen_device_<br/>Eigen::GpuDevice*"]
        B1["blas_handle_<br/>cublasHandle_t"]
        B2["blaslt_handle_<br/>cublasLtHandle_t"]
        D["dnn_handle_<br/>cudnnHandle_t"]
        SOL["solver_handle_<br/>cusolverDnHandle_t"]
        SP["sparse_handle_<br/>cusparseHandle_t"]
        A["allocator_<br/>Allocator*"]
        W["workspace_<br/>DnnWorkspaceHandle*"]
    end

    subgraph StreamBinding["Stream 绑定资源"]
        Eig["Eigen Stream Device"]
        CuB1["cuBLAS"]
        CuB2["cuBLASLt"]
        CuD["cuDNN"]
        CuS["cuSOLVER"]
        CuSp["cuSPARSE"]
    end

    S --> Eig
    S --> CuB1
    S --> CuB2
    S --> CuD
    S --> CuS
    S --> CuSp

    E --> Eig
    B1 --> CuB1
    B2 --> CuB2
    D --> CuD
    SOL --> CuS
    SP --> CuSp

    style S fill:#f99,stroke:#333,stroke-width:3px
```

**关键发现**：`stream_` 是核心资源，所有其他 handle（Eigen、cuBLAS、cuDNN 等）都绑定到这个 stream。如果 `stream_` 变成 thread-local，这些 handle 也需要考虑 thread-local 化。

---

## Kernel 使用 GPUContext stream 的典型模式

```mermaid
flowchart LR
    subgraph Pattern1["模式1: 直接获取stream"]
        P1A["dev_ctx.stream()"]
        P1B["cuda kernel"]
        P1C["<<<..., stream>>>"]
    end

    subgraph Pattern2["模式2: 存储到局部变量"]
        P2A["auto stream = dev_ctx.stream();"]
        P2B["phi::Stream stream;"]
        P2C["多kernel复用"]
    end

    subgraph Pattern3["模式3: 通过cudaStream_t传递"]
        P3A["cudaStream_t s = dev_ctx.stream();"]
        P3B["第三方库API"]
        P3C["cublas/cudnn调用"]
    end

    subgraph Pattern4["模式4: 封装Stream对象"]
        P4A["Stream(dev_ctx.stream())"]
        P4B["phi::Stream wrapper"]
        P4C["通用kernel launch"]
    end

    P1A --> P1B --> P1C
    P2A --> P2B --> P2C
    P3A --> P3B --> P3C
    P4A --> P4B --> P4C
```

**统计**：
- 模式1（直接获取）：~400 处
- 模式2（局部变量）：~300 处
- 模式3（cudaStream_t）：~100 处
- 模式4（Stream封装）：~25 处

---

## 修改影响面评估

### 如果 GPUContext stream 改为 thread-local

```mermaid
flowchart TD
    A["GPUContext::stream() 返回 thread-local"] --> B["直接影响"]
    A --> C["间接影响"]

    B --> B1["825处 kernel launch 自动使用 TLS stream"]
    B --> B2["c10 compat getCurrentCUDAStream() 可直接读取 GPUContext"]
    B --> B3["setCurrentCUDAStream() 只需设置 GPUContext"]

    C --> C1["Eigen device 绑定 stream → 需 TLS 化?"]
    C --> C2["cuBLAS handle 绑定 stream → 需重新创建?"]
    C --> C3["cuDNN handle 绑定 stream → 需重新创建?"]
    C --> C4["Allocator 绑定 stream → 需 per-thread allocator?"]
    C --> C5["DnnWorkspace 绑定 stream → 需 TLS 化?"]

    style B1 fill:#9f9
    style B2 fill:#9f9
    style B3 fill:#9f9
    style C1 fill:#ff9
    style C2 fill:#ff9
    style C3 fill:#ff9
    style C4 fill:#f99
    style C5 fill:#ff9
```

**风险等级**：
- 🟢 **低风险**：kernel launch 自动受益（825处）
- 🟡 **中风险**：handle 绑定问题（需验证）
- 🔴 **高风险**：allocator 绑定（可能需要 per-thread allocator）

---

## 与 PyTorch 对比

```mermaid
flowchart LR
    subgraph PaddleCurrent["Paddle 当前架构"]
        PC1["DeviceContextPool<br/>(per-device global)"]
        PC2["GPUContext<br/>(shared stream)"]
        PC3["phi kernel<br/>(global stream)"]
        PC4["c10 TLS<br/>(separate path)"]
    end

    subgraph PyTorch["PyTorch 架构"]
        PT1["DeviceGuard<br/>(per-thread)"]
        PT2["CUDAStreamPool<br/>(per-thread current)"]
        PT3["ATen kernel<br/>(TLS stream)"]
    end

    subgraph PaddleTarget["Paddle 目标架构 (方案B)"]
        PL1["DeviceContextPool<br/>(per-thread)"]
        PL2["GPUContext<br/>(TLS stream)"]
        PL3["phi kernel<br/>(TLS stream)"]
        PL4["c10<br/>(统一路径)"]
    end

    PC1 --> PC2 --> PC3
    PC4 -.->|分裂| PC2

    PT1 --> PT2 --> PT3

    PL1 --> PL2 --> PL3
    PL4 --> PL2

    style PC4 fill:#f99
    style PL4 fill:#9f9
```
