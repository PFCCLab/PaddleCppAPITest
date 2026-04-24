#include <ATen/ATen.h>
#include <ATen/core/Tensor.h>
#include <ATen/ops/sparse_coo_tensor.h>
#include <ATen/ops/sparse_csr_tensor.h>
#include <ATen/ops/zeros.h>
#include <ATen/ops/zeros_like.h>
#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "src/file_manager.h"

extern paddle_api_test::ThreadSafeParam g_custom_param;

// Helper functions to create tensors from initializer lists.
// Using at::from_blob for compatibility with older ATen versions.
static at::Tensor create_tensor_from_list(std::initializer_list<int64_t> data,
                                          at::ScalarType dtype = at::kLong) {
  std::vector<int64_t> vec(data);
  auto opts = at::TensorOptions().dtype(dtype);
  return at::from_blob(vec.data(), vec.size(), opts).clone();
}

static at::Tensor create_tensor_from_float_list(
    std::initializer_list<float> data) {
  std::vector<float> vec(data);
  auto opts = at::TensorOptions().dtype(at::kFloat);
  return at::from_blob(vec.data(), vec.size(), opts).clone();
}

static at::Tensor create_tensor_from_double_list(
    std::initializer_list<double> data, at::ScalarType dtype = at::kDouble) {
  std::vector<double> vec(data);
  auto opts = at::TensorOptions().dtype(dtype);
  return at::from_blob(vec.data(), vec.size(), opts).clone();
}

namespace at {
namespace test {

using paddle_api_test::FileManerger;
using paddle_api_test::ThreadSafeParam;

class SparseTensorTest : public ::testing::Test {
 protected:
  void SetUp() override {}
};

static void write_sparse_info_to_file(FileManerger* file, const at::Tensor& t) {
  *file << std::to_string(t.dim()) << " ";
  for (int64_t i = 0; i < t.dim(); ++i) {
    *file << std::to_string(t.sizes()[i]) << " ";
  }
}

// ===================== sparse_coo_tensor =====================

// 基本 COO 创建：2D sparse tensor
TEST_F(SparseTensorTest, SparseCOOBasic2D) {
  // 3x4 sparse tensor, 非零元素在 (0,1), (1,2), (2,0)
  auto idx_data = create_tensor_from_list({0L, 1L, 2L, 1L, 2L, 0L});
  at::Tensor indices = idx_data.reshape({2, 3});
  at::Tensor values = create_tensor_from_float_list({1.0f, 2.0f, 3.0f});

  at::Tensor sparse = at::sparse_coo_tensor(indices, values, {3, 4});

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();
  file << "SparseCOOBasic2D ";
  write_sparse_info_to_file(&file, sparse);
  file << "\n";
  file.saveFile();
}

// COO 3D tensor
TEST_F(SparseTensorTest, SparseCOO3D) {
  auto idx_data = create_tensor_from_list({0L, 1L, 0L, 1L, 0L, 1L});
  at::Tensor indices = idx_data.reshape({3, 2});
  at::Tensor values = create_tensor_from_float_list({10.0f, 20.0f});

  at::Tensor sparse = at::sparse_coo_tensor(indices, values, {2, 2, 2});

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "SparseCOO3D ";
  write_sparse_info_to_file(&file, sparse);
  file << "\n";
  file.saveFile();
}

// COO 带 TensorOptions 重载
TEST_F(SparseTensorTest, SparseCOOWithOptions) {
  auto idx_data = create_tensor_from_list({0L, 1L, 2L, 1L, 2L, 0L});
  at::Tensor indices = idx_data.reshape({2, 3});
  at::Tensor values = create_tensor_from_float_list({1.0f, 2.0f, 3.0f});

  at::Tensor sparse =
      at::sparse_coo_tensor(indices, values, {3, 4}, at::TensorOptions());

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "SparseCOOWithOptions ";
  write_sparse_info_to_file(&file, sparse);
  file << "\n";
  file.saveFile();
}

// COO 无 size 重载 (推断 size)
TEST_F(SparseTensorTest, SparseCOOInferSize) {
  auto idx_data = create_tensor_from_list({0L, 1L, 2L, 1L, 2L, 0L});
  at::Tensor indices = idx_data.reshape({2, 3});
  at::Tensor values = create_tensor_from_float_list({5.0f, 6.0f, 7.0f});

  at::Tensor sparse = at::sparse_coo_tensor(indices, values);

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "SparseCOOInferSize ";
  write_sparse_info_to_file(&file, sparse);
  file << "\n";
  file.saveFile();
}

// COO 带扩展选项重载 - 使用 TensorOptions 兼容
TEST_F(SparseTensorTest, SparseCOOWithExpandedOptions) {
  auto idx_data = create_tensor_from_list({0L, 1L, 0L, 1L});
  at::Tensor indices = idx_data.reshape({2, 2});
  at::Tensor values = create_tensor_from_float_list({1.5f, 2.5f});

  // Use TensorOptions to specify layout — works on both LibTorch and Paddle.
  auto opts = at::TensorOptions().layout(c10::kSparse);
  at::Tensor sparse = at::sparse_coo_tensor(indices, values, {2, 3}, opts);

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "SparseCOOWithExpandedOptions ";
  write_sparse_info_to_file(&file, sparse);
  file << "\n";
  file.saveFile();
}

// COO Double 类型
TEST_F(SparseTensorTest, SparseCOODouble) {
  auto idx_data = create_tensor_from_list({0L, 1L, 0L, 1L});
  at::Tensor indices = idx_data.reshape({2, 2});
  at::Tensor values = create_tensor_from_double_list({1.1, 2.2}, at::kDouble);

  at::Tensor sparse = at::sparse_coo_tensor(indices, values, {2, 3});

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "SparseCOODouble ";
  write_sparse_info_to_file(&file, sparse);
  file << "\n";
  file.saveFile();
}

// COO 单个非零元素
TEST_F(SparseTensorTest, SparseCOOSingleNonzero) {
  at::Tensor indices = create_tensor_from_list({1L, 2L}).reshape({2, 1});
  at::Tensor values = create_tensor_from_float_list({42.0f});

  at::Tensor sparse = at::sparse_coo_tensor(indices, values, {5, 5});

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "SparseCOOSingleNonzero ";
  write_sparse_info_to_file(&file, sparse);
  file << "\n";
  file.saveFile();
}

// COO 大 shape
TEST_F(SparseTensorTest, SparseCOOLargeShape) {
  // 100x100 sparse，5 个非零元素
  std::vector<int64_t> row_idx = {0, 10, 50, 80, 99};
  std::vector<int64_t> col_idx = {0, 20, 50, 70, 99};
  std::vector<int64_t> idx_vec;
  idx_vec.insert(idx_vec.end(), row_idx.begin(), row_idx.end());
  idx_vec.insert(idx_vec.end(), col_idx.begin(), col_idx.end());

  auto idx_blob =
      create_tensor_from_list({0, 10, 50, 80, 99, 0, 20, 50, 70, 99});
  at::Tensor indices = idx_blob.reshape({2, 5});
  at::Tensor values =
      create_tensor_from_float_list({1.0f, 2.0f, 3.0f, 4.0f, 5.0f});

  at::Tensor sparse = at::sparse_coo_tensor(indices, values, {100, 100});

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "SparseCOOLargeShape ";
  write_sparse_info_to_file(&file, sparse);
  file << "\n";
  file.saveFile();
}

// ===================== sparse_csr_tensor =====================

// 基本 CSR 创建：3x3 矩阵
TEST_F(SparseTensorTest, SparseCSRBasic) {
  // CSR format: crow_indices, col_indices, values
  // 矩阵:
  // [1 0 2]
  // [0 3 0]
  // [4 0 5]
  at::Tensor crow_indices = create_tensor_from_list({0L, 2L, 3L, 5L});
  at::Tensor col_indices = create_tensor_from_list({0L, 2L, 1L, 0L, 2L});
  at::Tensor values =
      create_tensor_from_float_list({1.0f, 2.0f, 3.0f, 4.0f, 5.0f});

  at::Tensor sparse = at::sparse_csr_tensor(
      crow_indices, col_indices, values, {3, 3}, at::kFloat);

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "SparseCSRBasic ";
  write_sparse_info_to_file(&file, sparse);
  file << "\n";
  file.saveFile();
}

// CSR 4x5 矩阵
TEST_F(SparseTensorTest, SparseCSR4x5) {
  // 4x5 sparse，6 个非零
  at::Tensor crow_indices = create_tensor_from_list({0L, 2L, 3L, 5L, 6L});
  at::Tensor col_indices = create_tensor_from_list({0L, 3L, 1L, 2L, 4L, 0L});
  at::Tensor values =
      create_tensor_from_float_list({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});

  at::Tensor sparse = at::sparse_csr_tensor(
      crow_indices, col_indices, values, {4, 5}, at::kFloat);

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "SparseCSR4x5 ";
  write_sparse_info_to_file(&file, sparse);
  file << "\n";
  file.saveFile();
}

// CSR 带扩展选项 - 使用 TensorOptions 兼容
TEST_F(SparseTensorTest, SparseCSRWithExpandedOptions) {
  at::Tensor crow_indices = create_tensor_from_list({0L, 1L, 2L});
  at::Tensor col_indices = create_tensor_from_list({0L, 1L});
  at::Tensor values = create_tensor_from_float_list({10.0f, 20.0f});

  // Use TensorOptions to specify layout — works on both LibTorch and Paddle.
  auto opts = at::TensorOptions().layout(c10::kSparseCsr);
  at::Tensor sparse =
      at::sparse_csr_tensor(crow_indices, col_indices, values, {2, 3}, opts);

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "SparseCSRWithExpandedOptions ";
  write_sparse_info_to_file(&file, sparse);
  file << "\n";
  file.saveFile();
}

// CSR Double 类型
TEST_F(SparseTensorTest, SparseCSRDouble) {
  at::Tensor crow_indices = create_tensor_from_list({0L, 2L, 3L});
  at::Tensor col_indices = create_tensor_from_list({0L, 1L, 2L});
  at::Tensor values =
      create_tensor_from_double_list({1.1, 2.2, 3.3}, at::kDouble);

  at::Tensor sparse = at::sparse_csr_tensor(
      crow_indices, col_indices, values, {2, 4}, at::kDouble);

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "SparseCSRDouble ";
  write_sparse_info_to_file(&file, sparse);
  file << "\n";
  file.saveFile();
}

// CSR 大 shape
TEST_F(SparseTensorTest, SparseCSRLargeShape) {
  // 100x100, 对角线非零
  // Simplified: just create a 3x3 CSR for testing
  at::Tensor crow_indices = create_tensor_from_list({0L, 1L, 2L, 3L});
  at::Tensor col_indices = create_tensor_from_list({0L, 1L, 2L});
  at::Tensor values = create_tensor_from_float_list({1.0f, 2.0f, 3.0f});

  at::Tensor sparse = at::sparse_csr_tensor(
      crow_indices, col_indices, values, {100, 100}, at::kFloat);

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "SparseCSRLargeShape ";
  write_sparse_info_to_file(&file, sparse);
  file << "\n";
  file.saveFile();
}

TEST_F(SparseTensorTest, SparseCOODtypeCastOptions) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "SparseCOODtypeCastOptions ";

  // [DIFF] Paddle casts values to the TensorOptions dtype here; PyTorch
  // requires values dtype to already match sparse tensor dtype.
  try {
    at::Tensor indices =
        create_tensor_from_list({0L, 1L, 0L, 1L}).reshape({2, 2});
    at::Tensor values = create_tensor_from_float_list({1.5f, 2.5f});
    auto options = at::TensorOptions().dtype(at::kDouble).layout(c10::kSparse);

    at::Tensor sparse = at::sparse_coo_tensor(indices, values, {2, 3}, options);
    at::Tensor dense_values = sparse._values();

    write_sparse_info_to_file(&file, sparse);
    file << std::to_string(dense_values.numel()) << " ";
    file << std::to_string(dense_values.data_ptr<double>()[0]) << " ";
  } catch (const std::exception&) {
    file << "exception ";
  }
  file << "\n";
  file.saveFile();
}

TEST_F(SparseTensorTest, SparseCSRDtypeCastOptions) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "SparseCSRDtypeCastOptions ";

  // [DIFF] Paddle accepts TensorOptions dtype conversion for CSR values;
  // PyTorch rejects mismatched values dtype during CSR construction.
  try {
    at::Tensor crow_indices = create_tensor_from_list({0L, 1L, 2L});
    at::Tensor col_indices = create_tensor_from_list({0L, 1L});
    at::Tensor values = create_tensor_from_float_list({3.5f, 4.5f});
    auto options =
        at::TensorOptions().dtype(at::kDouble).layout(c10::kSparseCsr);

    at::Tensor sparse = at::sparse_csr_tensor(
        crow_indices, col_indices, values, {2, 3}, options);
    at::Tensor dense_values = sparse._values();

    write_sparse_info_to_file(&file, sparse);
    file << std::to_string(sparse._nnz()) << " ";
    file << std::to_string(dense_values.numel()) << " ";
    file << std::to_string(dense_values.data_ptr<double>()[1]) << " ";
  } catch (const std::exception&) {
    file << "exception ";
  }
  file << "\n";
  file.saveFile();
}

TEST_F(SparseTensorTest, SparseCSRValuesAndNnz) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "SparseCSRValuesAndNnz ";

  try {
    at::Tensor crow_indices = create_tensor_from_list({0L, 2L, 3L});
    at::Tensor col_indices = create_tensor_from_list({0L, 1L, 1L});
    at::Tensor values = create_tensor_from_float_list({1.0f, 2.0f, 3.0f});

    at::Tensor sparse = at::sparse_csr_tensor(
        crow_indices, col_indices, values, {2, 3}, at::kFloat);
    file << std::to_string(sparse._nnz()) << " ";
    // [DIFF] Paddle maps CSR _values() to the values tensor; PyTorch does not
    // dispatch aten::_values for SparseCsrCPU.
    try {
      at::Tensor dense_values = sparse._values();
      file << std::to_string(dense_values.dim()) << " ";
      file << std::to_string(dense_values.numel()) << " ";
      file << std::to_string(dense_values.data_ptr<float>()[0]) << " ";
      file << std::to_string(dense_values.data_ptr<float>()[2]) << " ";
    } catch (const std::exception&) {
      file << "values_exception ";
    }
  } catch (const std::exception&) {
    file << "exception ";
  }
  file << "\n";
  file.saveFile();
}

TEST_F(SparseTensorTest, SparseZerosLikeKeepsLayout) {
  at::Tensor indices =
      create_tensor_from_list({0L, 1L, 0L, 1L}).reshape({2, 2});
  at::Tensor values = create_tensor_from_float_list({1.0f, 2.0f});
  at::Tensor sparse = at::sparse_coo_tensor(indices, values, {2, 3});
  at::Tensor zero_sparse = at::zeros_like(sparse);

  at::Tensor crow_indices = create_tensor_from_list({0L, 1L, 2L});
  at::Tensor col_indices = create_tensor_from_list({0L, 1L});
  at::Tensor csr_values = create_tensor_from_float_list({5.0f, 6.0f});
  at::Tensor csr = at::sparse_csr_tensor(
      crow_indices, col_indices, csr_values, {2, 3}, at::kFloat);
  at::Tensor zero_csr = at::zeros_like(csr);

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "SparseZerosLikeKeepsLayout ";
  file << std::to_string(zero_sparse.is_sparse() ? 1 : 0) << " ";
  file << std::to_string(zero_sparse._nnz()) << " ";
  file << std::to_string(zero_csr.is_sparse_csr() ? 1 : 0) << " ";
  file << std::to_string(zero_csr._nnz()) << " ";
  file << "\n";
  file.saveFile();
}

}  // namespace test
}  // namespace at
