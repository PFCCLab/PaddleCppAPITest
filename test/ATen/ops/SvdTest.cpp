#include <ATen/ATen.h>
#include <ATen/core/Tensor.h>
#include <ATen/ops/svd.h>
#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <string>
#include <vector>

#include "src/file_manager.h"

extern paddle_api_test::ThreadSafeParam g_custom_param;

namespace at {
namespace test {

using paddle_api_test::FileManerger;
using paddle_api_test::ThreadSafeParam;

// 输出 tensor 的 shape 和 dtype 信息（SVD 数值对算法敏感，不直接对比数值）
static void write_tensor_meta_to_file(FileManerger* file,
                                      const at::Tensor& result) {
  *file << std::to_string(result.dim()) << " ";
  *file << std::to_string(result.numel()) << " ";
  for (int64_t i = 0; i < result.dim(); ++i) {
    *file << std::to_string(result.sizes()[i]) << " ";
  }
  *file << std::to_string(static_cast<int>(result.scalar_type())) << " ";
}

// 输出单个标量值
static void write_scalar_to_file(FileManerger* file, double val) {
  *file << std::to_string(val) << " ";
}

static bool real_svd_reconstructs(const at::Tensor& input,
                                  const at::Tensor& U,
                                  const at::Tensor& S,
                                  const at::Tensor& V,
                                  double tolerance) {
  auto input_cont = input.contiguous();
  auto u_cont = U.contiguous();
  auto s_cont = S.contiguous();
  auto v_cont = V.contiguous();
  const float* input_data = input_cont.data_ptr<float>();
  const float* u_data = u_cont.data_ptr<float>();
  const float* s_data = s_cont.data_ptr<float>();
  const float* v_data = v_cont.data_ptr<float>();
  int64_t m = input.size(0);
  int64_t n = input.size(1);
  int64_t k = S.size(0);
  for (int64_t i = 0; i < m; ++i) {
    for (int64_t j = 0; j < n; ++j) {
      double reconstructed = 0.0;
      for (int64_t l = 0; l < k; ++l) {
        reconstructed += static_cast<double>(u_data[i * k + l]) *
                         static_cast<double>(s_data[l]) *
                         static_cast<double>(v_data[j * k + l]);
      }
      double expected = static_cast<double>(input_data[i * n + j]);
      if (std::abs(reconstructed - expected) > tolerance) {
        return false;
      }
    }
  }
  return true;
}

static bool complex_svd_reconstructs(const at::Tensor& input,
                                     const at::Tensor& U,
                                     const at::Tensor& S,
                                     const at::Tensor& V,
                                     bool conjugate_v,
                                     double tolerance) {
  auto input_cont = input.contiguous();
  auto u_cont = U.contiguous();
  auto s_cont = S.contiguous();
  auto v_cont = V.contiguous();
  const auto* input_data = input_cont.data_ptr<c10::complex<float>>();
  const auto* u_data = u_cont.data_ptr<c10::complex<float>>();
  const float* s_data = s_cont.data_ptr<float>();
  const auto* v_data = v_cont.data_ptr<c10::complex<float>>();
  int64_t m = input.size(0);
  int64_t n = input.size(1);
  int64_t k = S.size(0);
  for (int64_t i = 0; i < m; ++i) {
    for (int64_t j = 0; j < n; ++j) {
      double reconstructed_real = 0.0;
      double reconstructed_imag = 0.0;
      for (int64_t l = 0; l < k; ++l) {
        const auto* u_pair = reinterpret_cast<const float*>(&u_data[i * k + l]);
        const auto* v_pair = reinterpret_cast<const float*>(&v_data[j * k + l]);
        double ur = static_cast<double>(u_pair[0]);
        double ui = static_cast<double>(u_pair[1]);
        double vr = static_cast<double>(v_pair[0]);
        double vi = static_cast<double>(v_pair[1]);
        if (conjugate_v) {
          vi = -vi;
        }
        double scale = static_cast<double>(s_data[l]);
        reconstructed_real += scale * (ur * vr - ui * vi);
        reconstructed_imag += scale * (ur * vi + ui * vr);
      }
      const auto* expected_pair =
          reinterpret_cast<const float*>(&input_data[i * n + j]);
      double diff_real =
          reconstructed_real - static_cast<double>(expected_pair[0]);
      double diff_imag =
          reconstructed_imag - static_cast<double>(expected_pair[1]);
      double error = std::sqrt(diff_real * diff_real + diff_imag * diff_imag);
      if (error > tolerance) {
        return false;
      }
    }
  }
  return true;
}

class SvdTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // 基准 2D 矩阵 {3, 4}
    test_matrix = at::zeros({3, 4}, at::kFloat);
    float* data = test_matrix.data_ptr<float>();
    for (int i = 0; i < 12; ++i) {
      data[i] = static_cast<float>(i + 1);
    }
  }
  at::Tensor test_matrix;
};

// ========== 基础功能 ==========

TEST_F(SvdTest, BasicSvd) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();
  file << "BasicSvd ";
  auto [U, S, V] = at::svd(test_matrix);
  write_tensor_meta_to_file(&file, U);
  write_tensor_meta_to_file(&file, S);
  write_tensor_meta_to_file(&file, V);
  file << "\n";
  file.saveFile();
}

// ========== Shape 覆盖 ==========

// 小矩阵 {2, 3}
TEST_F(SvdTest, SmallMatrix) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "SmallMatrix ";
  at::Tensor t = at::zeros({2, 3}, at::kFloat);
  float* data = t.data_ptr<float>();
  data[0] = 1.0f;
  data[1] = 2.0f;
  data[2] = 3.0f;
  data[3] = 4.0f;
  data[4] = 5.0f;
  data[5] = 6.0f;
  auto [U, S, V] = at::svd(t);
  write_tensor_meta_to_file(&file, U);
  write_tensor_meta_to_file(&file, S);
  write_tensor_meta_to_file(&file, V);
  file << "\n";
  file.saveFile();
}

// 方阵 {4, 4}
TEST_F(SvdTest, SquareMatrix) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "SquareMatrix ";
  at::Tensor t = at::zeros({4, 4}, at::kFloat);
  float* data = t.data_ptr<float>();
  for (int i = 0; i < 16; ++i) {
    data[i] = static_cast<float>(i + 1);
  }
  auto [U, S, V] = at::svd(t);
  write_tensor_meta_to_file(&file, U);
  write_tensor_meta_to_file(&file, S);
  write_tensor_meta_to_file(&file, V);
  file << "\n";
  file.saveFile();
}

// 宽矩阵 {3, 5} (m < n)
TEST_F(SvdTest, WideMatrix) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "WideMatrix ";
  at::Tensor t = at::zeros({3, 5}, at::kFloat);
  float* data = t.data_ptr<float>();
  for (int i = 0; i < 15; ++i) {
    data[i] = static_cast<float>(i + 1);
  }
  auto [U, S, V] = at::svd(t);
  write_tensor_meta_to_file(&file, U);
  write_tensor_meta_to_file(&file, S);
  write_tensor_meta_to_file(&file, V);
  file << "\n";
  file.saveFile();
}

// 大矩阵 {10, 10}
TEST_F(SvdTest, LargeMatrix) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "LargeMatrix ";
  at::Tensor t = at::zeros({10, 10}, at::kFloat);
  float* data = t.data_ptr<float>();
  for (int i = 0; i < 100; ++i) {
    data[i] = static_cast<float>(i + 1);
  }
  auto [U, S, V] = at::svd(t);
  write_tensor_meta_to_file(&file, U);
  write_tensor_meta_to_file(&file, S);
  write_tensor_meta_to_file(&file, V);
  file << "\n";
  file.saveFile();
}

// batch 矩阵 {2, 3, 4}
TEST_F(SvdTest, BatchMatrix) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "BatchMatrix ";
  at::Tensor t = at::zeros({2, 3, 4}, at::kFloat);
  float* data = t.data_ptr<float>();
  for (int i = 0; i < 24; ++i) {
    data[i] = static_cast<float>(i + 1);
  }
  auto [U, S, V] = at::svd(t);
  write_tensor_meta_to_file(&file, U);
  write_tensor_meta_to_file(&file, S);
  write_tensor_meta_to_file(&file, V);
  file << "\n";
  file.saveFile();
}

// ========== Dtype 覆盖 ==========

// float64
TEST_F(SvdTest, Float64Dtype) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "Float64Dtype ";
  at::Tensor t = at::zeros({3, 4}, at::kDouble);
  double* data = t.data_ptr<double>();
  for (int i = 0; i < 12; ++i) {
    data[i] = static_cast<double>(i + 1);
  }
  auto [U, S, V] = at::svd(t);
  write_tensor_meta_to_file(&file, U);
  write_tensor_meta_to_file(&file, S);
  write_tensor_meta_to_file(&file, V);
  file << "\n";
  file.saveFile();
}

TEST_F(SvdTest, RealReconstruction) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "RealReconstruction ";
  at::Tensor t = at::zeros({2, 3}, at::kFloat);
  float* data = t.data_ptr<float>();
  data[0] = 1.0f;
  data[1] = 2.0f;
  data[2] = 3.0f;
  data[3] = 4.0f;
  data[4] = 5.0f;
  data[5] = 7.0f;
  auto [U, S, V] = at::svd(t);
  file << std::to_string(real_svd_reconstructs(t, U, S, V, 1e-3)) << " ";
  file << "\n";
  file.saveFile();
}

TEST_F(SvdTest, ComplexReconstruction) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "ComplexReconstruction ";
  at::Tensor t = at::zeros({2, 3}, at::kComplexFloat);
  auto* data = t.data_ptr<c10::complex<float>>();
  data[0] = c10::complex<float>(1.0f, 2.0f);
  data[1] = c10::complex<float>(2.0f, -1.0f);
  data[2] = c10::complex<float>(3.0f, 0.0f);
  data[3] = c10::complex<float>(4.0f, 1.0f);
  data[4] = c10::complex<float>(5.0f, -2.0f);
  data[5] = c10::complex<float>(6.0f, 3.0f);
  auto [U, S, V] = at::svd(t);
  bool dtype_ok = U.scalar_type() == at::kComplexFloat &&
                  S.scalar_type() == at::kFloat &&
                  V.scalar_type() == at::kComplexFloat;
  bool reconstructs_with_conj =
      complex_svd_reconstructs(t, U, S, V, /*conjugate_v=*/true, 1e-3);
  bool reconstructs_without_conj =
      complex_svd_reconstructs(t, U, S, V, /*conjugate_v=*/false, 1e-3);
  // Libtorch may expose V with a lazy conjugate bit; direct data_ptr reads can
  // therefore observe a different storage convention from Paddle's materialized
  // complex tensor. Compare the public reconstruction property, not storage.
  bool exactly_one_reconstruction =
      reconstructs_with_conj != reconstructs_without_conj;
  file << std::to_string(dtype_ok) << " ";
  file << std::to_string(exactly_one_reconstruction) << " ";
  file << "\n";
  file.saveFile();
}

// ========== API 变体 ==========

// some=false (full matrices)
TEST_F(SvdTest, FullMatrices) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "FullMatrices ";
  auto [U, S, V] = at::svd(test_matrix, /*some=*/false);
  write_tensor_meta_to_file(&file, U);
  write_tensor_meta_to_file(&file, S);
  write_tensor_meta_to_file(&file, V);
  file << "\n";
  file.saveFile();
}

// compute_uv=false
TEST_F(SvdTest, NoComputeUv) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "NoComputeUv ";
  auto [U, S, V] = at::svd(test_matrix, /*some=*/true, /*compute_uv=*/false);
  write_tensor_meta_to_file(&file, U);
  write_tensor_meta_to_file(&file, S);
  write_tensor_meta_to_file(&file, V);
  // 验证 U 和 V 都是零
  float u_sum = U.abs().sum().item().template to<float>();
  float v_sum = V.abs().sum().item().template to<float>();
  bool u_all_zero = u_sum == 0.0f;
  bool v_all_zero = v_sum == 0.0f;
  file << std::to_string(u_all_zero) << " ";
  file << std::to_string(v_all_zero) << " ";
  file << "\n";
  file.saveFile();
}

// 方法调用 t.svd()
TEST_F(SvdTest, MethodSvd) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "MethodSvd ";
  auto [U, S, V] = test_matrix.svd();
  write_tensor_meta_to_file(&file, U);
  write_tensor_meta_to_file(&file, S);
  write_tensor_meta_to_file(&file, V);
  file << "\n";
  file.saveFile();
}

// compute_uv=false + some=false
TEST_F(SvdTest, FullNoComputeUv) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "FullNoComputeUv ";
  auto [U, S, V] = at::svd(test_matrix, /*some=*/false, /*compute_uv=*/false);
  write_tensor_meta_to_file(&file, U);
  write_tensor_meta_to_file(&file, S);
  write_tensor_meta_to_file(&file, V);
  float u_sum = U.abs().sum().item().template to<float>();
  float v_sum = V.abs().sum().item().template to<float>();
  bool u_all_zero = u_sum == 0.0f;
  bool v_all_zero = v_sum == 0.0f;
  file << std::to_string(u_all_zero) << " ";
  file << std::to_string(v_all_zero) << " ";
  file << "\n";
  file.saveFile();
}

}  // namespace test
}  // namespace at
