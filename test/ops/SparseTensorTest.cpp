#include <ATen/ATen.h>
#include <ATen/core/Tensor.h>
#include <ATen/ops/sparse_coo_tensor.h>
#include <ATen/ops/sparse_csr_tensor.h>
#include <ATen/ops/zeros.h>
#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "../../src/file_manager.h"

extern paddle_api_test::ThreadSafeParam g_custom_param;

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
  at::Tensor indices = at::tensor({0L, 1L, 2L, 1L, 2L, 0L}).reshape({2, 3});
  at::Tensor values = at::tensor({1.0f, 2.0f, 3.0f});

  at::Tensor sparse = at::sparse_coo_tensor(indices, values, {3, 4});

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();
  write_sparse_info_to_file(&file, sparse);
  file.saveFile();
}

// COO 3D tensor
TEST_F(SparseTensorTest, SparseCOO3D) {
  at::Tensor indices = at::tensor({0L, 1L, 0L, 1L, 0L, 1L}).reshape({3, 2});
  at::Tensor values = at::tensor({10.0f, 20.0f});

  at::Tensor sparse = at::sparse_coo_tensor(indices, values, {2, 2, 2});

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  write_sparse_info_to_file(&file, sparse);
  file.saveFile();
}

// COO 带 TensorOptions 重载
TEST_F(SparseTensorTest, SparseCOOWithOptions) {
  at::Tensor indices = at::tensor({0L, 1L, 2L, 1L, 2L, 0L}).reshape({2, 3});
  at::Tensor values = at::tensor({1.0f, 2.0f, 3.0f});

  at::Tensor sparse =
      at::sparse_coo_tensor(indices, values, {3, 4}, at::TensorOptions());

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  write_sparse_info_to_file(&file, sparse);
  file.saveFile();
}

// COO 无 size 重载 (推断 size)
TEST_F(SparseTensorTest, SparseCOOInferSize) {
  at::Tensor indices = at::tensor({0L, 1L, 2L, 1L, 2L, 0L}).reshape({2, 3});
  at::Tensor values = at::tensor({5.0f, 6.0f, 7.0f});

  at::Tensor sparse = at::sparse_coo_tensor(indices, values);

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  write_sparse_info_to_file(&file, sparse);
  file.saveFile();
}

// COO 带扩展选项重载
TEST_F(SparseTensorTest, SparseCOOWithExpandedOptions) {
  at::Tensor indices = at::tensor({0L, 1L, 0L, 1L}).reshape({2, 2});
  at::Tensor values = at::tensor({1.5f, 2.5f});

  at::Tensor sparse = at::sparse_coo_tensor(indices,
                                            values,
                                            {2, 3},
                                            at::kFloat,     // dtype
                                            c10::kSparse,   // layout
                                            c10::nullopt,   // device
                                            c10::nullopt);  // pin_memory

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  write_sparse_info_to_file(&file, sparse);
  file.saveFile();
}

// COO Double 类型
TEST_F(SparseTensorTest, SparseCOODouble) {
  at::Tensor indices = at::tensor({0L, 1L, 0L, 1L}).reshape({2, 2});
  at::Tensor values = at::tensor({1.1, 2.2}, at::kDouble);

  at::Tensor sparse = at::sparse_coo_tensor(indices, values, {2, 3});

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  write_sparse_info_to_file(&file, sparse);
  file.saveFile();
}

// COO 单个非零元素
TEST_F(SparseTensorTest, SparseCOOSingleNonzero) {
  at::Tensor indices = at::tensor({1L, 2L}).reshape({2, 1});
  at::Tensor values = at::tensor({42.0f});

  at::Tensor sparse = at::sparse_coo_tensor(indices, values, {5, 5});

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  write_sparse_info_to_file(&file, sparse);
  file.saveFile();
}

// COO 大 shape
TEST_F(SparseTensorTest, SparseCOOLargeShape) {
  // 100x100 sparse，5 个非零元素
  std::vector<int64_t> row_idx = {0, 10, 50, 80, 99};
  std::vector<int64_t> col_idx = {0, 20, 50, 70, 99};
  std::vector<int64_t> idx_data;
  idx_data.insert(idx_data.end(), row_idx.begin(), row_idx.end());
  idx_data.insert(idx_data.end(), col_idx.begin(), col_idx.end());

  at::Tensor indices =
      at::tensor(at::ArrayRef<int64_t>(idx_data)).reshape({2, 5});
  at::Tensor values = at::tensor({1.0f, 2.0f, 3.0f, 4.0f, 5.0f});

  at::Tensor sparse = at::sparse_coo_tensor(indices, values, {100, 100});

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  write_sparse_info_to_file(&file, sparse);
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
  at::Tensor crow_indices = at::tensor({0L, 2L, 3L, 5L});
  at::Tensor col_indices = at::tensor({0L, 2L, 1L, 0L, 2L});
  at::Tensor values = at::tensor({1.0f, 2.0f, 3.0f, 4.0f, 5.0f});

  at::Tensor sparse =
      at::sparse_csr_tensor(crow_indices, col_indices, values, {3, 3});

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  write_sparse_info_to_file(&file, sparse);
  file.saveFile();
}

// CSR 4x5 矩阵
TEST_F(SparseTensorTest, SparseCSR4x5) {
  // 4x5 sparse，6 个非零
  at::Tensor crow_indices = at::tensor({0L, 2L, 3L, 5L, 6L});
  at::Tensor col_indices = at::tensor({0L, 3L, 1L, 2L, 4L, 0L});
  at::Tensor values = at::tensor({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});

  at::Tensor sparse =
      at::sparse_csr_tensor(crow_indices, col_indices, values, {4, 5});

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  write_sparse_info_to_file(&file, sparse);
  file.saveFile();
}

// CSR 带扩展选项
TEST_F(SparseTensorTest, SparseCSRWithExpandedOptions) {
  at::Tensor crow_indices = at::tensor({0L, 1L, 2L});
  at::Tensor col_indices = at::tensor({0L, 1L});
  at::Tensor values = at::tensor({10.0f, 20.0f});

  at::Tensor sparse = at::sparse_csr_tensor(crow_indices,
                                            col_indices,
                                            values,
                                            {2, 3},
                                            at::kFloat,       // dtype
                                            c10::kSparseCsr,  // layout
                                            c10::nullopt,     // device
                                            c10::nullopt);    // pin_memory

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  write_sparse_info_to_file(&file, sparse);
  file.saveFile();
}

// CSR Double 类型
TEST_F(SparseTensorTest, SparseCSRDouble) {
  at::Tensor crow_indices = at::tensor({0L, 2L, 3L});
  at::Tensor col_indices = at::tensor({0L, 1L, 2L});
  at::Tensor values = at::tensor({1.1, 2.2, 3.3}, at::kDouble);

  at::Tensor sparse =
      at::sparse_csr_tensor(crow_indices, col_indices, values, {2, 4});

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  write_sparse_info_to_file(&file, sparse);
  file.saveFile();
}

// CSR 大 shape
TEST_F(SparseTensorTest, SparseCSRLargeShape) {
  // 100x100, 对角线非零
  std::vector<int64_t> crow(101);
  std::vector<int64_t> col(100);
  std::vector<float> vals(100);
  for (int64_t i = 0; i <= 100; ++i) crow[i] = i;
  for (int64_t i = 0; i < 100; ++i) {
    col[i] = i;
    vals[i] = static_cast<float>(i + 1);
  }

  at::Tensor crow_indices = at::tensor(at::ArrayRef<int64_t>(crow));
  at::Tensor col_indices = at::tensor(at::ArrayRef<int64_t>(col));
  at::Tensor values = at::tensor(at::ArrayRef<float>(vals));

  at::Tensor sparse =
      at::sparse_csr_tensor(crow_indices, col_indices, values, {100, 100});

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  write_sparse_info_to_file(&file, sparse);
  file.saveFile();
}

}  // namespace test
}  // namespace at
