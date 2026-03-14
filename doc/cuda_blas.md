##### CUDABlas.h 头文件 API 兼容性

对比文件：
- `/home/may/Paddle/paddle/phi/api/include/compat/ATen/cuda/CUDABlas.h`
- `/home/may/pytorch/aten/src/ATen/cuda/CUDABlas.h`

状态说明：
- `✅` 已实现（接口存在且签名兼容）
- `🔧` 部分兼容（接口存在，但模板覆盖范围或能力受限）
- `❌` 未实现（PyTorch 有，Paddle compat 头文件无）

---

### 文件级差异

| 项目 | paddle API 兼容性 | 备注 |
|---|---|---|
| 头文件定位 | 🔧 | Paddle 文档注释明确是 `subset`，仅覆盖部分 CUDA BLAS 接口 |
| 额外依赖 `ATen/BlasBackend.h` | ❌ | PyTorch 包含，Paddle 未包含（对应 `scaled_gemm` 能力缺失） |
| `at::cuda::blas` 命名空间 | ✅ | 两侧一致 |

---

### GEMM 主接口

| torch API | paddle API 兼容性 | 备注 |
|---|---|---|
| `CUDABLAS_GEMM_ARGTYPES` | ✅ | 宏存在 |
| `CUDABLAS_GEMM_ARGTYPES_AND_C_DTYPE` | ✅ | 宏存在 |
| `CUDABLAS_GEMM_ARGS` | ✅ | 宏存在 |
| `gemm<Dtype, C_Dtype = Dtype>` 通用模板 | ✅ | 存在，默认 `static_assert` 未实现占位 |
| `gemm<double>` | ✅ | 已声明 |
| `gemm<float>` | ✅ | 已声明 |
| `gemm<c10::complex<double>>` | ✅ | 已声明 |
| `gemm<c10::complex<float>>` | ✅ | 已声明 |
| `gemm<at::Half>` | ✅ | 已声明 |
| `gemm<at::BFloat16>` | ✅ | 已声明 |
| `gemm<at::Half, float>` | ❌ | PyTorch 有，Paddle 缺失 |
| `gemm<at::BFloat16, float>` | ❌ | PyTorch 有，Paddle 缺失 |
| `CUDABLAS_GEMM_DTYPE_IS_FLOAT_TYPE_AND_C_DTYPE_IS_FLOAT` | ❌ | PyTorch 的条件模板宏，Paddle 未提供 |

---

### GEMM 扩展能力

| torch API | paddle API 兼容性 | 备注 |
|---|---|---|
| `gemm_internal<...>`（通用与特化） | ❌ | Paddle 缺失 |
| `GEMMAndBiasActivationEpilogue` | ❌ | Paddle 缺失 |
| `gemm_and_bias(...)` | ❌ | Paddle 缺失 |
| `int8_gemm(...)` | ❌ | Paddle 缺失 |
| `scaled_gemm(...)` | ❌ | Paddle 缺失 |

---

### 批量 GEMM (`bgemm`) 能力

| torch API | paddle API 兼容性 | 备注 |
|---|---|---|
| `CUDABLAS_BGEMM_ARGTYPES` | ❌ | Paddle 缺失 |
| `CUDABLAS_BGEMM_ARGTYPES_AND_C_DTYPE` | ❌ | Paddle 缺失 |
| `CUDABLAS_BGEMM_ARGS` | ❌ | Paddle 缺失 |
| `bgemm<...>`（通用与特化） | ❌ | Paddle 缺失 |
| `bgemm_internal<...>`（通用与特化） | ❌ | Paddle 缺失 |

---

### 其他 BLAS/LAPACK 风格接口

| torch API | paddle API 兼容性 | 备注 |
|---|---|---|
| `PointerModeGuard` | ❌ | Paddle 缺失 |
| `trsm<...>` | ❌ | Paddle 缺失 |
| `trsmBatched<...>` | ❌ | Paddle 缺失 |
| `gemv<...>` | ❌ | Paddle 缺失 |
| `dot<...>` | ❌ | Paddle 缺失 |
| `vdot<...>` | ❌ | Paddle 缺失 |
| `getrsBatched<...>` | ❌ | Paddle 缺失 |
| `geqrfBatched<...>` | ❌ | Paddle 缺失 |
| `getrfBatched<...>` | ❌ | Paddle 缺失 |
| `gelsBatched<...>` | ❌ | Paddle 缺失 |

---

### 兼容性统计（基于以上条目）

| 状态 | 数量 |
|---|---|
| ✅ 已实现 | 11 |
| 🔧 部分兼容 | 2 |
| ❌ 未实现 | 24 |

---

### 结论

- Paddle compat 版 `CUDABlas.h` 当前定位为精简子集，核心聚焦在 `gemm` 主路径。
- 与 PyTorch 相比，主要缺口在：
  - `gemm` 的扩展路径（`gemm_internal`、`gemm_and_bias`、量化 `int8/scaled`）
  - `bgemm`、`gemv`、`dot/vdot`、`trsm` 与 batched 线代接口
  - `PointerModeGuard` 等配套工具
- 若目标是提升 API 对齐度，建议优先补齐 `gemm<Half/BFloat16, float>` 与 `bgemm`，这两部分对混合精度和批量矩阵计算影响最大。
