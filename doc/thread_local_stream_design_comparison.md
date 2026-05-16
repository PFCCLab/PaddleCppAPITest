# Thread-Local Stream 两种设计方案对比

## 背景

Paddle phi kernel 使用 `GPUContext::stream()` 提交 CUDA kernel，而 `GPUContext` 是 per-device 全局单例。这意味着所有线程共享同一个 stream。当主线程设置了一个被阻塞的 stream，worker 线程的 phi kernel 也会提交到同一个被阻塞的 stream 上，导致死锁。

PyTorch 的 ATen kernel 使用 `c10::cuda::getCurrentCUDAStream()`（thread-local）提交 kernel，每个线程有自己的 stream，不会相互影响。

---

## 方案 A：c10 compat 层维护独立 TLS（当前 PR #78902）

### 架构图

```mermaid
flowchart TB
    subgraph MainThread["主线程 Main Thread"]
        A1["setCurrentCUDAStream(pool)"]
        A2["c10 TLS[main] = pool"]
        A3["GPUContext.stream = pool"]
        A4["phi kernel on pool"]
        A5["BLOCKED"]
    end

    A1 --> A2
    A1 --> A3
    A3 --> A4
    A4 --> A5

    subgraph WorkerThread["Worker 线程 Worker Thread"]
        direction TB
        subgraph TorchPath["Torch Build"]
            B1["getCurrentCUDAStream()"]
            B2["c10 TLS[worker] = default"]
            B3["ATen kernel on default"]
            B4["NOT BLOCKED"]
        end

        subgraph PaddlePath["Paddle Build"]
            C1["phi kernel dispatch"]
            C2["GPUContext::stream()"]
            C3["pool (global)"]
            C4["BLOCKED"]
        end
    end

    A1 -.->|event chain blocks pool| B3
    A1 -.->|event chain blocks pool| C3
```

### 数据流

```mermaid
sequenceDiagram
    participant Main as 主线程
    participant C10TLS as c10 TLS
    participant GPUCtx as GPUContext
    participant Phi as phi kernel

    Main->>C10TLS: setCurrentCUDAStream(pool)
    Main->>GPUCtx: stream = pool (全局)
    Note over GPUCtx: 全局 stream 被阻塞

    par Torch Worker
        WorkerTorch->>C10TLS: getCurrentCUDAStream()
        C10TLS-->>WorkerTorch: default stream (TLS)
        WorkerTorch->>Phi: ATen kernel on default
        Note right of Phi: 不阻塞
    and Paddle Worker
        WorkerPaddle->>GPUCtx: stream() (全局)
        GPUCtx-->>WorkerPaddle: pool (被阻塞)
        WorkerPaddle->>Phi: phi kernel on pool
        Note right of Phi: 死锁!
    end
```

### 优点

1. **修改集中**：只改 c10 compat 层两个文件（`CUDAStream.cpp/h`）
2. **向后兼容**：非 compat 模式完全不受影响
3. **零耦合**：不需要改 phi kernel dispatch 逻辑

### 缺点

1. **治标不治本**：phi kernel 仍然使用 GPUContext 全局 stream，死锁问题在 phi kernel 路径仍然存在
2. **两套语义**：c10 TLS 和 GPUContext stream 不一致，c10 caller 和 phi kernel 看到不同的 stream
3. **维护负担**：所有通过 c10 调用的代码走 TLS，所有 phi kernel 走全局，容易混淆

---

## 方案 B：GPUContext 本身使用 Thread-Local Stream

### 架构图

```mermaid
flowchart TB
    subgraph MainThreadB["主线程 Main Thread"]
        D1["setCurrentCUDAStream(pool)"]
        D2["GPUContext.stream[main] = pool"]
        D3["phi kernel on pool"]
        D4["BLOCKED"]
    end

    D1 --> D2
    D2 --> D3
    D3 --> D4

    subgraph WorkerThreadB["Worker 线程 Worker Thread"]
        direction TB
        subgraph TorchPathB["Torch Build"]
            E1["getCurrentCUDAStream()"]
            E2["GPUContext.stream[worker] = default"]
            E3["ATen kernel on default"]
            E4["NOT BLOCKED"]
        end

        subgraph PaddlePathB["Paddle Build"]
            F1["phi kernel dispatch"]
            F2["GPUContext::stream()"]
            F3["default (thread-local)"]
            F4["NOT BLOCKED"]
        end
    end

    D1 -.->|event chain blocks pool| E3
    D1 -.->|event chain blocks pool| F3
```

### 数据流

```mermaid
sequenceDiagram
    participant Main as 主线程
    participant GPUCtxB as GPUContext (TLS)
    participant PhiB as phi kernel

    Main->>GPUCtxB: setCurrentCUDAStream(pool)
    Note over GPUCtxB: stream[main] = pool<br/>stream[worker] = default

    par Torch Worker
        WorkerTorchB->>GPUCtxB: getCurrentCUDAStream()
        GPUCtxB-->>WorkerTorchB: default (TLS)
        WorkerTorchB->>PhiB: ATen kernel on default
        Note right of PhiB: 不阻塞
    and Paddle Worker
        WorkerPaddleB->>GPUCtxB: stream() (TLS)
        GPUCtxB-->>WorkerPaddleB: default (TLS)
        WorkerPaddleB->>PhiB: phi kernel on default
        Note right of PhiB: 不阻塞
    end
```

### 核心修改

```mermaid
flowchart LR
    subgraph Before["修改前"]
        G1["Impl.stream_"]
        G2["全局指针"]
        G3["所有线程共享"]
    end

    subgraph After["修改后"]
        H1["Impl.stream_"]
        H2["全局 fallback"]
        H3["Impl.tl_streams_"]
        H4["thread_local map"]
        H5["key = this"]
        H6["每个线程独立"]
    end

    G1 --> G2 --> G3
    H1 --> H2
    H3 --> H4 --> H5 --> H6
```

### 优点

1. **治本**：phi kernel 和 c10 compat 层看到同一个 thread-local stream，彻底解决死锁
2. **语义统一**：不再有"c10 TLS vs GPUContext 全局"的分裂
3. **PyTorch 对齐**：行为与 PyTorch 完全一致（所有 kernel 都使用 thread-local stream）

### 缺点

1. **修改面较广**：需要改 `GPUContext::Impl` 的 stream 访问逻辑
2. **Allocator 问题**：`SetStream` 同时更新 allocator，如果 allocator 也是全局的，可能需要同步处理
3. **Handle 重置**：Eigen/BLAS/DNN handle 绑定到 stream，stream 变化后可能需要重置 handle（原 TODO）
4. **向后兼容**：现有代码假设 GPUContext stream 是全局的，需要验证是否有代码依赖这个假设

---

## 关键差异对比

| 维度 | 方案 A (c10 TLS) | 方案 B (GPUContext TLS) |
|---|---|---|
| **修改文件数** | 2 个文件 (CUDAStream.cpp/h) | 1 个文件 (gpu_context.cc) |
| **phi kernel 死锁** | 仍然存在 | 解决 |
| **c10/phi 语义一致性** | 两套 stream | 统一 |
| **向后兼容性** | 完全兼容 | 需验证 |
| **PR #78652 影响** | 保持现有修复 | 可回退 c10 特殊处理，简化代码 |
| **PyTorch 对齐度** | 部分对齐 (c10 层) | 完全对齐 |

---

## 结论建议

**短期**：方案 A 已足够修复 c10 compat 层的 FastDeploy 问题（PR #78652）。

**长期**：方案 B 是正确架构，因为：
1. phi kernel 是 Paddle 的核心执行路径，不能长期容忍全局 stream 死锁
2. 与 PyTorch 的语义差异是 Paddle C++ API 兼容性的根本障碍
3. `DeviceContextPool` 设计本应是 per-thread 的（PyTorch 的 DeviceGuard 就是 thread-local）

**实施路径**：
1. 先在一个实验分支实现方案 B
2. 跑通 Paddle 全量单测，确认无回归
3. 跑通 PaddleCppAPITest 死锁复现测试，确认 paddle 输出从 `0` 变 `1`
4. 逐步合并到 develop
