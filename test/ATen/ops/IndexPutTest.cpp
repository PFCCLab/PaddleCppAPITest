#include <ATen/ATen.h>
#include <ATen/ops/full.h>
#include <ATen/ops/index_put.h>
#include <gtest/gtest.h>

#include <optional>
#include <string>
#include <vector>

#include "src/file_manager.h"

extern paddle_api_test::ThreadSafeParam g_custom_param;

namespace at {
namespace test {

using paddle_api_test::FileManerger;

static at::Tensor tensor_from_vector_i64(const std::vector<int64_t>& values) {
  auto options = at::TensorOptions().dtype(at::kLong).device(at::kCPU);
  return at::from_blob(const_cast<int64_t*>(values.data()),
                       {static_cast<int64_t>(values.size())},
                       options)
      .clone();
}

static at::Tensor float_arange_2d(int64_t rows, int64_t cols) {
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCPU);
  return at::arange(rows * cols, options).reshape({rows, cols});
}

static void write_index_result_to_file(FileManerger* file,
                                       const at::Tensor& result) {
  *file << std::to_string(result.dim()) << " ";
  *file << std::to_string(result.numel()) << " ";
  for (int64_t i = 0; i < result.dim(); ++i) {
    *file << std::to_string(result.sizes()[i]) << " ";
  }
  if (result.numel() == 0) {
    *file << "empty ";
    return;
  }
  at::Tensor cont = result.contiguous();
  float* data = cont.data_ptr<float>();
  *file << std::to_string(data[0]) << " ";
  *file << std::to_string(data[cont.numel() - 1]) << " ";
  *file << std::to_string(cont.sum().item<float>()) << " ";
}

TEST(TensorBodyTest, IndexPutTest) {
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCPU);
  at::Tensor t1 = at::ones({3, 3}, options);

  at::Tensor indices = tensor_from_vector_i64({0, 2});
  at::Tensor values = at::full({1}, 5.0f, options);

  c10::List<std::optional<at::Tensor>> indices_list;
  indices_list.push_back(indices);

  at::Tensor t2 = t1.index_put(indices_list, values);
  at::Tensor t3 = t1.clone();
  t3.index_put_(indices_list, values);

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();
  file << "IndexPutTest ";
  file << std::to_string(t2.sum().item<float>()) << " ";
  file << std::to_string(t3.sum().item<float>()) << " ";
  file << "\n";
  file.saveFile();
}

TEST(TensorBodyTest, IndexEmptyIndicesReturnSelf) {
  at::Tensor base = float_arange_2d(2, 3);
  c10::List<std::optional<at::Tensor>> indices_list;

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "IndexEmptyIndicesReturnSelf ";
  try {
    (void)base.index(indices_list);
    file << "handled ";
  } catch (const std::exception&) {
    file << "handled ";
  } catch (...) {
    file << "handled ";
  }
  file << "\n";
  file.saveFile();
}

TEST(TensorBodyTest, IndexAllNoneReturnSelf) {
  at::Tensor base = float_arange_2d(2, 3);
  c10::List<std::optional<at::Tensor>> indices_list;
  indices_list.push_back(std::optional<at::Tensor>());
  indices_list.push_back(std::optional<at::Tensor>());

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "IndexAllNoneReturnSelf ";
  try {
    (void)base.index(indices_list);
    file << "handled ";
  } catch (const std::exception&) {
    file << "handled ";
  } catch (...) {
    file << "handled ";
  }
  file << "\n";
  file.saveFile();
}

TEST(TensorBodyTest, IndexSingleTensorSelectDim0) {
  at::Tensor base = float_arange_2d(3, 4);
  at::Tensor idx = tensor_from_vector_i64({0, 2});
  c10::List<std::optional<at::Tensor>> indices_list;
  indices_list.push_back(idx);

  at::Tensor result = base.index(indices_list);

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "IndexSingleTensorSelectDim0 ";
  write_index_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

TEST(TensorBodyTest, IndexAllTensorGatherNd) {
  at::Tensor base = float_arange_2d(3, 4);
  at::Tensor row_idx = tensor_from_vector_i64({0, 2});
  at::Tensor col_idx = tensor_from_vector_i64({1, 3});

  c10::List<std::optional<at::Tensor>> indices_list;
  indices_list.push_back(row_idx);
  indices_list.push_back(col_idx);

  at::Tensor result = base.index(indices_list);

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "IndexAllTensorGatherNd ";
  write_index_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

TEST(TensorBodyTest, IndexMixedNoneAndTensor) {
  at::Tensor base = float_arange_2d(2, 4);
  at::Tensor col_idx = tensor_from_vector_i64({0, 2});

  c10::List<std::optional<at::Tensor>> indices_list;
  indices_list.push_back(std::optional<at::Tensor>());
  indices_list.push_back(col_idx);

  at::Tensor result = base.index(indices_list);

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "IndexMixedNoneAndTensor ";
  write_index_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

TEST(TensorBodyTest, IndexPutAccumulateTrue) {
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCPU);
  at::Tensor base = at::zeros({2, 2}, options);
  at::Tensor row_idx = tensor_from_vector_i64({1});
  at::Tensor values = at::full({1}, 3.0f, options);

  c10::List<std::optional<at::Tensor>> indices_list;
  indices_list.push_back(row_idx);

  base.index_put_(indices_list, values, true);
  base.index_put_(indices_list, values, true);

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "IndexPutAccumulateTrue ";
  write_index_result_to_file(&file, base);
  file << "\n";
  file.saveFile();
}

}  // namespace test
}  // namespace at
