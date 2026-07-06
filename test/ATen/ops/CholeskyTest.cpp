#include <ATen/ATen.h>
#include <ATen/core/Tensor.h>
#include <ATen/ops/cholesky.h>
#include <gtest/gtest.h>

#include <algorithm>
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

// 使用 stride 按逻辑行优先顺序写出 tensor 元素，兼容不同内存布局
static void write_cholesky_result_to_file(FileManerger* file,
                                          const at::Tensor& result) {
  *file << std::to_string(result.dim()) << " ";
  *file << std::to_string(result.numel()) << " ";
  for (int64_t i = 0; i < result.dim(); ++i) {
    *file << std::to_string(result.sizes()[i]) << " ";
  }

  // 使用 stride 按逻辑顺序访问元素
  int64_t ndim = result.dim();
  std::vector<int64_t> strides(ndim);
  for (int64_t i = 0; i < ndim; ++i) {
    strides[i] = result.stride(i);
  }

  switch (result.scalar_type()) {
    case at::kFloat: {
      float* data = result.data_ptr<float>();
      if (ndim == 2) {
        for (int64_t i = 0; i < result.size(0); ++i) {
          for (int64_t j = 0; j < result.size(1); ++j) {
            *file << std::to_string(data[i * strides[0] + j * strides[1]])
                  << " ";
          }
        }
      } else if (ndim == 3) {
        for (int64_t b = 0; b < result.size(0); ++b) {
          for (int64_t i = 0; i < result.size(1); ++i) {
            for (int64_t j = 0; j < result.size(2); ++j) {
              *file
                  << std::to_string(
                         data[b * strides[0] + i * strides[1] + j * strides[2]])
                  << " ";
            }
          }
        }
      }
      break;
    }
    case at::kDouble: {
      double* data = result.data_ptr<double>();
      if (ndim == 2) {
        for (int64_t i = 0; i < result.size(0); ++i) {
          for (int64_t j = 0; j < result.size(1); ++j) {
            *file << std::to_string(data[i * strides[0] + j * strides[1]])
                  << " ";
          }
        }
      } else if (ndim == 3) {
        for (int64_t b = 0; b < result.size(0); ++b) {
          for (int64_t i = 0; i < result.size(1); ++i) {
            for (int64_t j = 0; j < result.size(2); ++j) {
              *file
                  << std::to_string(
                         data[b * strides[0] + i * strides[1] + j * strides[2]])
                  << " ";
            }
          }
        }
      }
      break;
    }
    default: {
      *file << "unsupported_dtype ";
      break;
    }
  }
}

// 构建对角占优的对称正定矩阵
static at::Tensor make_spd_matrix(const std::vector<int64_t>& shape,
                                  at::ScalarType dtype) {
  at::Tensor A = at::zeros(shape, dtype);
  int64_t n = shape[shape.size() - 1];
  int64_t m = shape[shape.size() - 2];
  int64_t batch = 1;
  for (size_t i = 0; i + 2 < shape.size(); ++i) {
    batch *= shape[i];
  }

  if (dtype == at::kFloat) {
    float* data = A.data_ptr<float>();
    for (int64_t b = 0; b < batch; ++b) {
      for (int64_t i = 0; i < m; ++i) {
        for (int64_t j = 0; j < n; ++j) {
          int64_t idx = b * m * n + i * n + j;
          if (i == j) {
            data[idx] = static_cast<float>(n);
          } else {
            data[idx] = 0.5f;
          }
        }
      }
    }
  } else if (dtype == at::kDouble) {
    double* data = A.data_ptr<double>();
    for (int64_t b = 0; b < batch; ++b) {
      for (int64_t i = 0; i < m; ++i) {
        for (int64_t j = 0; j < n; ++j) {
          int64_t idx = b * m * n + i * n + j;
          if (i == j) {
            data[idx] = static_cast<double>(n);
          } else {
            data[idx] = 0.5;
          }
        }
      }
    }
  }
  return A;
}

template <typename T>
static double max_reconstruction_error_data(const at::Tensor& factor,
                                            const at::Tensor& original,
                                            bool upper) {
  const T* factor_data = factor.data_ptr<T>();
  const T* original_data = original.data_ptr<T>();
  const int64_t ndim = factor.dim();
  const int64_t n = factor.size(ndim - 1);
  const int64_t batch = factor.numel() / (n * n);
  std::vector<int64_t> factor_strides(ndim);
  std::vector<int64_t> original_strides(ndim);
  for (int64_t i = 0; i < ndim; ++i) {
    factor_strides[i] = factor.stride(i);
    original_strides[i] = original.stride(i);
  }

  double max_error = 0.0;
  for (int64_t b = 0; b < batch; ++b) {
    const int64_t factor_batch_offset = ndim == 3 ? b * factor_strides[0] : 0;
    const int64_t original_batch_offset =
        ndim == 3 ? b * original_strides[0] : 0;
    for (int64_t i = 0; i < n; ++i) {
      for (int64_t j = 0; j < n; ++j) {
        double reconstructed = 0.0;
        for (int64_t k = 0; k < n; ++k) {
          const int64_t left_idx =
              factor_batch_offset + (upper ? k * factor_strides[ndim - 2] +
                                                 i * factor_strides[ndim - 1]
                                           : i * factor_strides[ndim - 2] +
                                                 k * factor_strides[ndim - 1]);
          const int64_t right_idx =
              factor_batch_offset + (upper ? k * factor_strides[ndim - 2] +
                                                 j * factor_strides[ndim - 1]
                                           : j * factor_strides[ndim - 2] +
                                                 k * factor_strides[ndim - 1]);
          reconstructed += static_cast<double>(factor_data[left_idx]) *
                           static_cast<double>(factor_data[right_idx]);
        }
        const int64_t original_idx = original_batch_offset +
                                     i * original_strides[ndim - 2] +
                                     j * original_strides[ndim - 1];
        max_error = std::max(
            max_error,
            std::fabs(reconstructed -
                      static_cast<double>(original_data[original_idx])));
      }
    }
  }
  return max_error;
}

static double max_reconstruction_error(const at::Tensor& factor,
                                       const at::Tensor& original,
                                       bool upper) {
  if (factor.scalar_type() == at::kFloat) {
    return max_reconstruction_error_data<float>(factor, original, upper);
  }
  return max_reconstruction_error_data<double>(factor, original, upper);
}

static void write_reconstruction_error_to_file(FileManerger* file,
                                               const at::Tensor& factor,
                                               const at::Tensor& original,
                                               bool upper) {
  const double max_error = max_reconstruction_error(factor, original, upper);
  *file << std::to_string(factor.dim()) << " ";
  *file << std::to_string(factor.numel()) << " ";
  for (int64_t i = 0; i < factor.dim(); ++i) {
    *file << std::to_string(factor.sizes()[i]) << " ";
  }
  *file << std::to_string(max_error) << " ";
}

class CholeskyTest : public ::testing::Test {
 protected:
  void SetUp() override {}
};

// ========== 基础功能 ==========

TEST_F(CholeskyTest, BasicCholesky) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();
  file << "BasicCholesky ";
  at::Tensor A = make_spd_matrix({3, 3}, at::kFloat);
  at::Tensor result = at::cholesky(A);
  write_cholesky_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

TEST_F(CholeskyTest, UpperTrue) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "UpperTrue ";
  at::Tensor A = make_spd_matrix({3, 3}, at::kFloat);
  at::Tensor result = at::cholesky(A, /*upper=*/true);
  write_cholesky_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// ========== Shape 覆盖 ==========

// 小矩阵 {2, 2}
TEST_F(CholeskyTest, SmallMatrix) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "SmallMatrix ";
  at::Tensor A = make_spd_matrix({2, 2}, at::kFloat);
  at::Tensor result = at::cholesky(A);
  write_cholesky_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// 大矩阵 {8, 8}
TEST_F(CholeskyTest, LargeMatrix) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "LargeMatrix ";
  at::Tensor A = make_spd_matrix({8, 8}, at::kFloat);
  at::Tensor result = at::cholesky(A);
  write_cholesky_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// batch 矩阵 {2, 3, 3}
TEST_F(CholeskyTest, BatchMatrix) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "BatchMatrix ";
  at::Tensor A = make_spd_matrix({2, 3, 3}, at::kFloat);
  at::Tensor result = at::cholesky(A);
  write_cholesky_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

TEST_F(CholeskyTest, LowerReconstructsInput) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "LowerReconstructsInput ";
  at::Tensor A = make_spd_matrix({3, 3}, at::kFloat);
  at::Tensor result = at::cholesky(A);
  const double max_error = max_reconstruction_error(result, A, false);
  EXPECT_LT(max_error, 1e-4);
  write_reconstruction_error_to_file(&file, result, A, false);
  file << "\n";
  file.saveFile();
}

TEST_F(CholeskyTest, UpperReconstructsInput) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "UpperReconstructsInput ";
  at::Tensor A = make_spd_matrix({3, 3}, at::kFloat);
  at::Tensor result = at::cholesky(A, /*upper=*/true);
  const double max_error = max_reconstruction_error(result, A, true);
  EXPECT_LT(max_error, 1e-4);
  write_reconstruction_error_to_file(&file, result, A, true);
  file << "\n";
  file.saveFile();
}

TEST_F(CholeskyTest, BatchReconstructsInput) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "BatchReconstructsInput ";
  at::Tensor A = make_spd_matrix({2, 3, 3}, at::kFloat);
  at::Tensor result = at::cholesky(A);
  const double max_error = max_reconstruction_error(result, A, false);
  EXPECT_LT(max_error, 1e-4);
  write_reconstruction_error_to_file(&file, result, A, false);
  file << "\n";
  file.saveFile();
}

// ========== Dtype 覆盖 ==========

// float64
TEST_F(CholeskyTest, Float64Dtype) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "Float64Dtype ";
  at::Tensor A = make_spd_matrix({3, 3}, at::kDouble);
  at::Tensor result = at::cholesky(A);
  write_cholesky_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// ========== API 变体 ==========

// 方法调用 t.cholesky()
TEST_F(CholeskyTest, MethodCholesky) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "MethodCholesky ";
  at::Tensor A = make_spd_matrix({3, 3}, at::kFloat);
  at::Tensor result = A.cholesky();
  write_cholesky_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// 方法调用 t.cholesky(true)
TEST_F(CholeskyTest, MethodCholeskyUpper) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "MethodCholeskyUpper ";
  at::Tensor A = make_spd_matrix({3, 3}, at::kFloat);
  at::Tensor result = A.cholesky(/*upper=*/true);
  write_cholesky_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// ========== 异常测试 ==========

// 非正定矩阵应抛出异常
TEST_F(CholeskyTest, NonPositiveDefinite) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "NonPositiveDefinite ";
  at::Tensor A = at::zeros({3, 3}, at::kFloat);
  try {
    at::Tensor result = at::cholesky(A);
    write_cholesky_result_to_file(&file, result);
  } catch (const std::exception&) {
    file << "exception ";
  }
  file << "\n";
  file.saveFile();
}

}  // namespace test
}  // namespace at
