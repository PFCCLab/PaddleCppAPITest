##### CUDAContextLight.h 头文件 API 兼容性

对比文件：
- `/home/may/Paddle/paddle/phi/api/include/compat/ATen/cuda/CUDAContextLight.h`
- `/home/may/pytorch/aten/src/ATen/cuda/CUDAContextLight.h`

状态说明：
- `✅` 已实现（接口存在且签名兼容）
- `🔧` 部分兼容（接口存在，但类型/条件编译/导出语义有差异）
- `❌` 未实现（PyTorch 有，Paddle compat 头文件无）

---

### 基础查询接口

| torch API | paddle API 兼容性 | 备注 |
|---|---|---|
| `getNumGPUs()` | ✅ | inline 实现一致，均调用 `c10::cuda::device_count()` |
| `is_available()` | ✅ | inline 实现一致 |
| `getCurrentDeviceProperties()` | 🔧 | 已声明；返回类型在 Paddle 中为 `CUDAContextDeviceProp*`（CUDA/HIP 别名） |
| `warp_size()` | ✅ | 已声明 |
| `getDeviceProperties(DeviceIndex)` | 🔧 | 已声明；返回类型为 `CUDAContextDeviceProp*` |
| `canDeviceAccessPeer(DeviceIndex, DeviceIndex)` | ✅ | 已声明 |
| `getCUDADeviceAllocator()` | ✅ | 已声明 |

---

### Handle 相关接口

| torch API | paddle API 兼容性 | 备注 |
|---|---|---|
| `getCurrentCUDASparseHandle()` | 🔧 | 已声明；返回类型为 `CUDAContextSparseHandle`（CUDA/HIP 别名） |
| `getCurrentCUDABlasHandle()` | 🔧 | 已声明；返回类型为 `CUDAContextBlasHandle`（CUDA/HIP 别名） |
| `getCurrentCUDABlasLtHandle()` | 🔧 | 已声明；返回类型为 `CUDAContextBlasLtHandle`（CUDA/HIP 别名） |
| `getCurrentCUDASolverDnHandle()` | 🔧 | 已声明；Paddle 侧有 `#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)` 条件编译 |
| `getCurrentCudssHandle()` | 🔧 | 已声明；两侧均受 `USE_CUDSS` 条件编译控制 |

---

### Workspace 与缓存接口

| torch API | paddle API 兼容性 | 备注 |
|---|---|---|
| `clearCublasWorkspaces()` | ✅ | 已声明 |
| `clearCublasWorkspacesForStream(cudaStream_t)` | ❌ | Paddle compat 头文件中缺失 |
| `WorkspaceMapWithMutex` | ✅ | 结构体定义一致（`map` + `shared_mutex`） |
| `cublas_handle_stream_to_workspace()` | ✅ | 已声明 |
| `cublaslt_handle_stream_to_workspace()` | ✅ | 已声明 |
| `getChosenWorkspaceSize()` | ✅ | 已声明 |
| `getCUDABlasLtWorkspaceSize()` | ✅ | 已声明 |
| `getCUDABlasLtWorkspace()` | ✅ | 已声明 |

---

### 类型与头文件适配（Paddle compat 特点）

| 项目 | 兼容性 | 备注 |
|---|---|---|
| `CUDAContextDeviceProp`/`CUDAContext*Handle` 类型别名 | 🔧 | Paddle 新增统一别名：CUDA 走原生类型，HIP 走 `phi::*` 类型 |
| 直接包含 CUDA 头（`cuda_runtime_api.h` 等） | 🔧 | Paddle 通过 `paddle/phi/backends/gpu/forwards.h` 做前向声明与跨后端抽象 |
| 导出宏（`TORCH_CUDA_CPP_API`） | 🔧 | PyTorch 声明包含导出宏；Paddle compat 未使用该宏 |

---

### 兼容性统计（基于以上条目）

| 状态 | 数量 |
|---|---|
| ✅ 已实现 | 12 |
| 🔧 部分兼容 | 9 |
| ❌ 未实现 | 1 |

---

### 结论

- `CUDAContextLight.h` 的核心查询、workspace 与大多数 handle 接口在 Paddle compat 侧已具备。
- 明确缺口是 `clearCublasWorkspacesForStream(cudaStream_t)`。
- 主要差异来自跨 CUDA/HIP 适配策略：Paddle 使用类型别名与条件编译屏蔽后端差异，这使接口在“可用性”层面兼容，但在类型名与导出语义层面与上游存在差别。
