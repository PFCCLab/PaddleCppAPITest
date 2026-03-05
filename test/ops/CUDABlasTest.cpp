#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContextLight.h>

// 【差异点1】at::cuda::blas::gemm<T> 符号可见性差异
// PyTorch（libtorch）将 at::cuda::blas::gemm<T> 编译为 hidden visibility
// （动态符号表中 nm -D 结果为小写 't'），外部代码无法链接该符号。
// Paddle compat 库中该符号为公开导出（大写 'T'），可正常从外部调用。
// 因此仅在 Paddle 构建（USE_PADDLE_API=1）时包含头文件并实例化 gemm 测试；
// Torch 构建输出 "not_exported" 占位，保持两端输出行对齐。
#if USE_PADDLE_API
#include <ATen/cuda/CUDABlas.h>
#endif

#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "../../src/file_manager.h"

extern paddle_api_test::ThreadSafeParam g_custom_param;

namespace at {
namespace test {

using paddle_api_test::FileManerger;
using paddle_api_test::ThreadSafeParam;

class CUDABlasTest : public ::testing::Test {
 protected:
  void SetUp() override {}
};

#if USE_PADDLE_API
// Write a column-major m×n result matrix (on GPU) to file as float values.
// Converts any dtype to float before serialisation so the format is uniform
// across all dtype tests.
static void write_gemm_result_to_file(FileManerger* file,
                                      const at::Tensor& result_gpu,
                                      int64_t m,
                                      int64_t n) {
  at::Tensor cpu = result_gpu.cpu().to(at::kFloat);
  float* data = cpu.data_ptr<float>();
  *file << std::to_string(m) << " " << std::to_string(n) << " ";
  for (int64_t i = 0; i < m * n; ++i) {
    *file << std::to_string(data[i]) << " ";
  }
}
#endif  // USE_PADDLE_API

// ============================================================
// at::cuda::getCurrentCUDABlasHandle tests
// ============================================================

// 验证 getCurrentCUDABlasHandle 返回非空 handle。
// 使用 `auto` 接收返回值以屏蔽类型差异：
// 【差异点2】返回类型差异
//   PyTorch：直接返回 cublasHandle_t
//   Paddle compat：返回 at::cuda::CUDAContextBlasHandle，
//     在 CUDA 构建中该类型 typedef 为 cublasHandle_t，
//     在 HIP/ROCm 构建中则为 phi::blasHandle_t（不同类型）。
TEST_F(CUDABlasTest, HandleNonNull) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();
  file << "HandleNonNull ";
  if (!at::cuda::is_available()) {
    file << "no_cuda ";
    file << "\n";
    file.saveFile();
    return;
  }
#if USE_PADDLE_API
  // 【差异点3】Paddle 的 getCurrentCUDABlasHandle() 实现依赖框架全局状态
  // Paddle 内部调用 phi::DeviceContextPool::Instance().Get(GPUPlace())，
  // 该调用要求事先通过 paddle::framework::InitDevices() 初始化
  // DeviceContextPool。 在独立 C++ 测试二进制中框架未初始化，Paddle 抛出
  // PreconditionNotMet 异常。 PyTorch 无此约束，只需 CUDA
  // 分配器初始化即可正常返回 handle。 输出 "exception_needs_pool_init"
  // 记录该行为差异。
  try {
    auto handle = at::cuda::getCurrentCUDABlasHandle();
    file << std::to_string(handle != nullptr ? 1 : 0) << " ";
  } catch (const std::exception&) {
    file << "exception_needs_pool_init ";
  }
#else
  // PyTorch 在首次创建 cuBLAS handle 前要求 CUDA 缓存分配器已初始化。
  // 分配一个 dummy GPU tensor 是触发该初始化的标准方式。
  {
    auto _init = at::zeros({1}, at::kFloat).cuda();
    (void)_init;
  }
  try {
    auto handle = at::cuda::getCurrentCUDABlasHandle();
    file << std::to_string(handle != nullptr ? 1 : 0) << " ";
  } catch (const std::exception& e) {
    file << "exception:" << e.what() << " ";
  }
#endif
  file << "\n";
  file.saveFile();
}

// Verify that two successive calls on the same thread return the same handle
// (the implementation caches one handle per CUDA stream).
TEST_F(CUDABlasTest, HandleConsistency) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "HandleConsistency ";
  if (!at::cuda::is_available()) {
    file << "no_cuda ";
    file << "\n";
    file.saveFile();
    return;
  }
#if USE_PADDLE_API
  // 与 HandleNonNull 相同的 pool-init 限制（见差异点3），
  // Paddle 在 DeviceContextPool 未初始化时抛出异常。
  try {
    auto handle1 = at::cuda::getCurrentCUDABlasHandle();
    auto handle2 = at::cuda::getCurrentCUDABlasHandle();
    file << std::to_string(handle1 == handle2 ? 1 : 0) << " ";
  } catch (const std::exception&) {
    file << "exception_needs_pool_init ";
  }
#else
  {
    auto _init = at::zeros({1}, at::kFloat).cuda();
    (void)_init;
  }
  try {
    auto handle1 = at::cuda::getCurrentCUDABlasHandle();
    auto handle2 = at::cuda::getCurrentCUDABlasHandle();
    file << std::to_string(handle1 == handle2 ? 1 : 0) << " ";
  } catch (const std::exception& e) {
    file << "exception:" << e.what() << " ";
  }
#endif
  file << "\n";
  file.saveFile();
}

#if USE_PADDLE_API

// 【差异点4】at::tensor(initializer_list<T>) 无 TensorOptions 重载缺失
// PyTorch 的 ATen/Utils.h 提供 at::tensor(std::initializer_list<float>)
// 直接推断类型的重载； Paddle compat 的 ATen/Utils.h 仅提供
// at::tensor(ArrayRef<T>, TensorOptions) 形式， 不支持不带 TensorOptions 的
// initializer_list 重载。 因此此处使用 cpu_fill_f32 辅助函数代替
// at::tensor({...}) 构造张量。

// 辅助函数：在 CPU 上从 initializer_list 填充 float32 一维张量后移至 GPU。
static at::Tensor cpu_fill_f32(std::initializer_list<float> vals) {
  auto t = at::zeros({(int64_t)vals.size()}, at::kFloat);
  float* p = t.data_ptr<float>();
  int64_t i = 0;
  for (float v : vals) p[i++] = v;
  return t.cuda();
}

// ============================================================
// at::cuda::blas::gemm<T> tests
//
// All matrix data use column-major (BLAS native) layout.
// For a 2×2 example the storage order is:
//   [A[0,0], A[1,0], A[0,1], A[1,1]]
//
// Basic reference setup (used in most tests unless otherwise noted):
//   A (col-major) = [1, 3, 2, 4]  →  matrix [[1,2],[3,4]]
//   B (col-major) = [1, 0, 0, 1]  →  identity I₂
//   alpha=1, beta=0  →  C = A·I = A  →  expected output: [1, 3, 2, 4]
// ============================================================

// --- float32 ---

TEST_F(CUDABlasTest, GemmFloat) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "GemmFloat ";
  if (!at::cuda::is_available()) {
    file << "no_cuda ";
    file << "\n";
    file.saveFile();
    return;
  }
  at::Tensor a = cpu_fill_f32({1.0f, 3.0f, 2.0f, 4.0f});
  at::Tensor b = cpu_fill_f32({1.0f, 0.0f, 0.0f, 1.0f});
  at::Tensor c = at::zeros({4}, at::kFloat).cuda();
  at::cuda::blas::gemm<float>('N',
                              'N',
                              2,
                              2,
                              2,
                              1.0f,
                              a.data_ptr<float>(),
                              2,
                              b.data_ptr<float>(),
                              2,
                              0.0f,
                              c.data_ptr<float>(),
                              2);
  write_gemm_result_to_file(&file, c, 2, 2);
  file << "\n";
  file.saveFile();
}

// --- float64 ---

TEST_F(CUDABlasTest, GemmDouble) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "GemmDouble ";
  if (!at::cuda::is_available()) {
    file << "no_cuda ";
    file << "\n";
    file.saveFile();
    return;
  }
  at::Tensor a = cpu_fill_f32({1.0f, 3.0f, 2.0f, 4.0f}).to(at::kDouble);
  at::Tensor b = cpu_fill_f32({1.0f, 0.0f, 0.0f, 1.0f}).to(at::kDouble);
  at::Tensor c = at::zeros({4}, at::kDouble).cuda();
  at::cuda::blas::gemm<double>('N',
                               'N',
                               2,
                               2,
                               2,
                               1.0,
                               a.data_ptr<double>(),
                               2,
                               b.data_ptr<double>(),
                               2,
                               0.0,
                               c.data_ptr<double>(),
                               2);
  write_gemm_result_to_file(&file, c, 2, 2);
  file << "\n";
  file.saveFile();
}

// --- float16 (at::Half) ---
// alpha / beta are at::opmath_type<at::Half> = float

TEST_F(CUDABlasTest, GemmHalf) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "GemmHalf ";
  if (!at::cuda::is_available()) {
    file << "no_cuda ";
    file << "\n";
    file.saveFile();
    return;
  }
  at::Tensor a = cpu_fill_f32({1.0f, 3.0f, 2.0f, 4.0f}).to(at::kHalf);
  at::Tensor b = cpu_fill_f32({1.0f, 0.0f, 0.0f, 1.0f}).to(at::kHalf);
  at::Tensor c = at::zeros({4}, at::kHalf).cuda();
  at::cuda::blas::gemm<at::Half>('N',
                                 'N',
                                 2,
                                 2,
                                 2,
                                 1.0f,
                                 a.data_ptr<at::Half>(),
                                 2,
                                 b.data_ptr<at::Half>(),
                                 2,
                                 0.0f,
                                 c.data_ptr<at::Half>(),
                                 2);
  write_gemm_result_to_file(&file, c, 2, 2);
  file << "\n";
  file.saveFile();
}

// --- bfloat16 (at::BFloat16) ---

TEST_F(CUDABlasTest, GemmBFloat16) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "GemmBFloat16 ";
  if (!at::cuda::is_available()) {
    file << "no_cuda ";
    file << "\n";
    file.saveFile();
    return;
  }
  at::Tensor a = cpu_fill_f32({1.0f, 3.0f, 2.0f, 4.0f}).to(at::kBFloat16);
  at::Tensor b = cpu_fill_f32({1.0f, 0.0f, 0.0f, 1.0f}).to(at::kBFloat16);
  at::Tensor c = at::zeros({4}, at::kBFloat16).cuda();
  at::cuda::blas::gemm<at::BFloat16>('N',
                                     'N',
                                     2,
                                     2,
                                     2,
                                     1.0f,
                                     a.data_ptr<at::BFloat16>(),
                                     2,
                                     b.data_ptr<at::BFloat16>(),
                                     2,
                                     0.0f,
                                     c.data_ptr<at::BFloat16>(),
                                     2);
  write_gemm_result_to_file(&file, c, 2, 2);
  file << "\n";
  file.saveFile();
}

// --- non-zero beta: C = alpha*A*B + beta*C_init ---
// C_init (col-major) = [1,1,1,1], beta=0.5
// Result = [1,3,2,4] + 0.5*[1,1,1,1] = [1.5, 3.5, 2.5, 4.5]

TEST_F(CUDABlasTest, GemmWithBeta) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "GemmWithBeta ";
  if (!at::cuda::is_available()) {
    file << "no_cuda ";
    file << "\n";
    file.saveFile();
    return;
  }
  at::Tensor a = cpu_fill_f32({1.0f, 3.0f, 2.0f, 4.0f});
  at::Tensor b = cpu_fill_f32({1.0f, 0.0f, 0.0f, 1.0f});
  at::Tensor c = cpu_fill_f32({1.0f, 1.0f, 1.0f, 1.0f});
  at::cuda::blas::gemm<float>('N',
                              'N',
                              2,
                              2,
                              2,
                              1.0f,
                              a.data_ptr<float>(),
                              2,
                              b.data_ptr<float>(),
                              2,
                              0.5f,
                              c.data_ptr<float>(),
                              2);
  write_gemm_result_to_file(&file, c, 2, 2);
  file << "\n";
  file.saveFile();
}

// --- negative values in A ---
// A (col-major) = [-1,-2,3,4]  →  matrix [[-1,3],[-2,4]]
// B = identity, C = A  →  expected: [-1,-2,3,4]

TEST_F(CUDABlasTest, GemmNegativeValues) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "GemmNegativeValues ";
  if (!at::cuda::is_available()) {
    file << "no_cuda ";
    file << "\n";
    file.saveFile();
    return;
  }
  at::Tensor a = cpu_fill_f32({-1.0f, -2.0f, 3.0f, 4.0f});
  at::Tensor b = cpu_fill_f32({1.0f, 0.0f, 0.0f, 1.0f});
  at::Tensor c = at::zeros({4}, at::kFloat).cuda();
  at::cuda::blas::gemm<float>('N',
                              'N',
                              2,
                              2,
                              2,
                              1.0f,
                              a.data_ptr<float>(),
                              2,
                              b.data_ptr<float>(),
                              2,
                              0.0f,
                              c.data_ptr<float>(),
                              2);
  write_gemm_result_to_file(&file, c, 2, 2);
  file << "\n";
  file.saveFile();
}

// --- transa='T': C = A^T · B ---
// A stored (col-major k×m = 2×2) = [1,2,3,4]
//   A[0,0]=1, A[1,0]=2, A[0,1]=3, A[1,1]=4  →  A^T = [[1,2],[3,4]]
// B (col-major) = [2,1,1,2]
//   B[0,0]=2, B[1,0]=1, B[0,1]=1, B[1,1]=2
// C = A^T · B = [[1,2],[3,4]] · [[2,1],[1,2]]
//   C[0,0]=4, C[1,0]=10, C[0,1]=5, C[1,1]=11 → col-major: [4,10,5,11]

TEST_F(CUDABlasTest, GemmTransA) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "GemmTransA ";
  if (!at::cuda::is_available()) {
    file << "no_cuda ";
    file << "\n";
    file.saveFile();
    return;
  }
  at::Tensor a = cpu_fill_f32({1.0f, 2.0f, 3.0f, 4.0f});
  at::Tensor b = cpu_fill_f32({2.0f, 1.0f, 1.0f, 2.0f});
  at::Tensor c = at::zeros({4}, at::kFloat).cuda();
  // transa='T': A is stored as k×m (2×2) with lda=k=2
  at::cuda::blas::gemm<float>('T',
                              'N',
                              2,
                              2,
                              2,
                              1.0f,
                              a.data_ptr<float>(),
                              2,
                              b.data_ptr<float>(),
                              2,
                              0.0f,
                              c.data_ptr<float>(),
                              2);
  write_gemm_result_to_file(&file, c, 2, 2);
  file << "\n";
  file.saveFile();
}

// --- transb='T': C = A · B^T ---
// A (col-major m×k = 2×2) = [1,3,2,4]  →  A = [[1,2],[3,4]]
// B stored (col-major n×k = 2×2) = [1,2,3,4], ldb=n=2
//   B[0,0]=1, B[1,0]=2, B[0,1]=3, B[1,1]=4
//   B^T: op(B)[i,j]=B[j,i]  →  row-major view: [[1,3],[2,4]]
// C = A · B^T = [[1,2],[3,4]] · [[1,2],[3,4]]
//   C[0,0]=7, C[1,0]=15, C[0,1]=10, C[1,1]=22 → col-major: [7,15,10,22]

TEST_F(CUDABlasTest, GemmTransB) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "GemmTransB ";
  if (!at::cuda::is_available()) {
    file << "no_cuda ";
    file << "\n";
    file.saveFile();
    return;
  }
  at::Tensor a = cpu_fill_f32({1.0f, 3.0f, 2.0f, 4.0f});
  at::Tensor b = cpu_fill_f32({1.0f, 2.0f, 3.0f, 4.0f});
  at::Tensor c = at::zeros({4}, at::kFloat).cuda();
  // transb='T': B is stored as n×k (2×2) with ldb=n=2
  at::cuda::blas::gemm<float>('N',
                              'T',
                              2,
                              2,
                              2,
                              1.0f,
                              a.data_ptr<float>(),
                              2,
                              b.data_ptr<float>(),
                              2,
                              0.0f,
                              c.data_ptr<float>(),
                              2);
  write_gemm_result_to_file(&file, c, 2, 2);
  file << "\n";
  file.saveFile();
}

// --- scalar (1×1) gemm ---
// A=[5], B=[3], alpha=1, beta=0  →  C=[15]

TEST_F(CUDABlasTest, GemmScalar) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "GemmScalar ";
  if (!at::cuda::is_available()) {
    file << "no_cuda ";
    file << "\n";
    file.saveFile();
    return;
  }
  at::Tensor a = cpu_fill_f32({5.0f});
  at::Tensor b = cpu_fill_f32({3.0f});
  at::Tensor c = at::zeros({1}, at::kFloat).cuda();
  at::cuda::blas::gemm<float>('N',
                              'N',
                              1,
                              1,
                              1,
                              1.0f,
                              a.data_ptr<float>(),
                              1,
                              b.data_ptr<float>(),
                              1,
                              0.0f,
                              c.data_ptr<float>(),
                              1);
  write_gemm_result_to_file(&file, c, 1, 1);
  file << "\n";
  file.saveFile();
}

// --- large matrix (100×100, ≥10000 elements) ---
// A = all-ones 100×100, B = all-ones 100×100, alpha=1, beta=0
// Each C[i,j] = sum_k A[i,k]*B[k,j] = 100.
// Sum of all C elements = 100 * 100 * 100 = 1,000,000.
// Only the sum is written to keep the output file compact.

TEST_F(CUDABlasTest, GemmLargeMatrix) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "GemmLargeMatrix ";
  if (!at::cuda::is_available()) {
    file << "no_cuda ";
    file << "\n";
    file.saveFile();
    return;
  }
  constexpr int64_t M = 100, N = 100, K = 100;
  at::Tensor a = at::ones({M * K}, at::kFloat).cuda();
  at::Tensor b = at::ones({K * N}, at::kFloat).cuda();
  at::Tensor c = at::zeros({M * N}, at::kFloat).cuda();
  at::cuda::blas::gemm<float>('N',
                              'N',
                              M,
                              N,
                              K,
                              1.0f,
                              a.data_ptr<float>(),
                              M,
                              b.data_ptr<float>(),
                              K,
                              0.0f,
                              c.data_ptr<float>(),
                              M);
  at::Tensor c_cpu = c.cpu();
  float* data = c_cpu.data_ptr<float>();
  float total = 0.0f;
  for (int64_t i = 0; i < M * N; ++i) {
    total += data[i];
  }
  file << std::to_string(M) << " " << std::to_string(N) << " ";
  file << std::to_string(total) << " ";
  file << "\n";
  file.saveFile();
}

#else  // !USE_PADDLE_API

// 【差异点1 对应桩代码】
// at::cuda::blas::gemm<T> 在 libtorch 中为 hidden visibility，外部无法链接。
// 输出 "not_exported" 占位，保持 torch/paddle 两端输出文件行数对齐，
// 以便比对脚本（result_cmp.sh）能逐行对比其余可比较的测试结果。
#define CUDABLAS_GEMM_STUB(name)           \
  TEST_F(CUDABlasTest, name) {             \
    auto file_name = g_custom_param.get(); \
    FileManerger file(file_name);          \
    file.openAppend();                     \
    file << #name " not_exported ";        \
    file << "\n";                          \
    file.saveFile();                       \
  }

CUDABLAS_GEMM_STUB(GemmFloat)
CUDABLAS_GEMM_STUB(GemmDouble)
CUDABLAS_GEMM_STUB(GemmHalf)
CUDABLAS_GEMM_STUB(GemmBFloat16)
CUDABLAS_GEMM_STUB(GemmWithBeta)
CUDABLAS_GEMM_STUB(GemmNegativeValues)
CUDABLAS_GEMM_STUB(GemmTransA)
CUDABLAS_GEMM_STUB(GemmTransB)
CUDABLAS_GEMM_STUB(GemmScalar)
CUDABLAS_GEMM_STUB(GemmLargeMatrix)

#undef CUDABLAS_GEMM_STUB

#endif  // USE_PADDLE_API

}  // namespace test
}  // namespace at
