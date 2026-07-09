#include <ATen/ATen.h>
#include <ATen/ops/index_reduce.h>
#include <gtest/gtest.h>

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

static at::Tensor tensor_from_vector_i32(const std::vector<int32_t>& values) {
  auto options = at::TensorOptions().dtype(at::kInt).device(at::kCPU);
  return at::from_blob(const_cast<int32_t*>(values.data()),
                       {static_cast<int64_t>(values.size())},
                       options)
      .clone();
}

static at::Tensor tensor_from_vector_f32(const std::vector<float>& values) {
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCPU);
  return at::from_blob(const_cast<float*>(values.data()),
                       {static_cast<int64_t>(values.size())},
                       options)
      .clone();
}

static void write_index_reduce_result_to_file(FileManerger* file,
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
  if (result.dtype() == at::kFloat) {
    float* data = cont.data_ptr<float>();
    *file << std::to_string(data[0]) << " ";
    *file << std::to_string(data[cont.numel() - 1]) << " ";
    *file << std::to_string(cont.sum().item<float>()) << " ";
  } else if (result.dtype() == at::kDouble) {
    double* data = cont.data_ptr<double>();
    *file << std::to_string(data[0]) << " ";
    *file << std::to_string(data[cont.numel() - 1]) << " ";
    *file << std::to_string(cont.sum().item<double>()) << " ";
  } else if (result.dtype() == at::kInt) {
    int* data = cont.data_ptr<int>();
    *file << std::to_string(data[0]) << " ";
    *file << std::to_string(data[cont.numel() - 1]) << " ";
    *file << std::to_string(cont.sum().item<int>()) << " ";
  } else if (result.dtype() == at::kLong) {
    int64_t* data = cont.data_ptr<int64_t>();
    *file << std::to_string(data[0]) << " ";
    *file << std::to_string(data[cont.numel() - 1]) << " ";
    *file << std::to_string(cont.sum().item<int64_t>()) << " ";
  }
}

class IndexReduceTest : public ::testing::Test {
 protected:
  void SetUp() override {}
};

// index_reduce with "prod" reduce, float dtype, small shape
TEST_F(IndexReduceTest, IndexReduceProdFloatSmall) {
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCPU);
  at::Tensor self = at::ones({3, 4}, options);
  at::Tensor index = tensor_from_vector_i64({0, 2});
  at::Tensor source = at::full({2, 4}, 2.0f, options);

  at::Tensor result = self.index_reduce(0, index, source, "prod");

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();
  file << "IndexReduceProdFloatSmall ";
  write_index_reduce_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// index_reduce with "mean" reduce, float dtype, small shape
TEST_F(IndexReduceTest, IndexReduceMeanFloatSmall) {
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCPU);
  at::Tensor self = at::zeros({3, 4}, options);
  at::Tensor index = tensor_from_vector_i64({0, 1, 0});
  at::Tensor source = at::full({3, 4}, 3.0f, options);

  at::Tensor result = self.index_reduce(0, index, source, "mean");

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "IndexReduceMeanFloatSmall ";
  write_index_reduce_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// index_reduce with "amax" reduce, float dtype
TEST_F(IndexReduceTest, IndexReduceAmaxFloat) {
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCPU);
  at::Tensor self = at::arange(0, 12, options).reshape({3, 4});
  at::Tensor index = tensor_from_vector_i64({0, 2, 0});
  at::Tensor source = at::full({3, 4}, 10.0f, options);

  at::Tensor result = self.index_reduce(0, index, source, "amax");

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "IndexReduceAmaxFloat ";
  write_index_reduce_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// index_reduce with "amin" reduce, float dtype
TEST_F(IndexReduceTest, IndexReduceAminFloat) {
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCPU);
  at::Tensor self = at::arange(0, 12, options).reshape({3, 4});
  at::Tensor index = tensor_from_vector_i64({1, 1, 0});
  at::Tensor source = at::full({3, 4}, -5.0f, options);

  at::Tensor result = self.index_reduce(0, index, source, "amin");

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "IndexReduceAminFloat ";
  write_index_reduce_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// index_reduce with include_self=false
TEST_F(IndexReduceTest, IndexReduceProdExcludeSelf) {
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCPU);
  at::Tensor self = at::full({3, 4}, 5.0f, options);
  at::Tensor index = tensor_from_vector_i64({0, 2});
  at::Tensor source = at::full({2, 4}, 2.0f, options);

  at::Tensor result = self.index_reduce(0, index, source, "prod", false);

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "IndexReduceProdExcludeSelf ";
  write_index_reduce_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// index_reduce_ in-place version
TEST_F(IndexReduceTest, IndexReduceInplaceProd) {
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCPU);
  at::Tensor self = at::ones({3, 4}, options);
  at::Tensor index = tensor_from_vector_i64({0, 2});
  at::Tensor source = at::full({2, 4}, 3.0f, options);

  self.index_reduce_(0, index, source, "prod");

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "IndexReduceInplaceProd ";
  write_index_reduce_result_to_file(&file, self);
  file << "\n";
  file.saveFile();
}

// index_reduce with double dtype
TEST_F(IndexReduceTest, IndexReduceMeanDouble) {
  auto options = at::TensorOptions().dtype(at::kDouble).device(at::kCPU);
  at::Tensor self = at::zeros({3, 4}, options);
  at::Tensor index = tensor_from_vector_i64({0, 1, 0});
  at::Tensor source = at::full({3, 4}, 2.0, options);

  at::Tensor result = self.index_reduce(0, index, source, "mean");

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "IndexReduceMeanDouble ";
  write_index_reduce_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// index_reduce with int dtype
TEST_F(IndexReduceTest, IndexReduceAmaxInt) {
  auto options = at::TensorOptions().dtype(at::kInt).device(at::kCPU);
  at::Tensor self = at::arange(0, 12, options).reshape({3, 4});
  at::Tensor index = tensor_from_vector_i64({0, 2, 0});
  at::Tensor source = at::full({3, 4}, 20, options);

  at::Tensor result = self.index_reduce(0, index, source, "amax");

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "IndexReduceAmaxInt ";
  write_index_reduce_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// index_reduce with long dtype
TEST_F(IndexReduceTest, IndexReduceAminLong) {
  auto options = at::TensorOptions().dtype(at::kLong).device(at::kCPU);
  at::Tensor self = at::arange(0, 12, options).reshape({3, 4});
  at::Tensor index = tensor_from_vector_i64({1, 1, 0});
  at::Tensor source = at::full({3, 4}, -3L, options);

  at::Tensor result = self.index_reduce(0, index, source, "amin");

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "IndexReduceAminLong ";
  write_index_reduce_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// index_reduce with large shape
TEST_F(IndexReduceTest, IndexReduceProdLargeShape) {
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCPU);
  at::Tensor self = at::ones({50, 20}, options);
  std::vector<int64_t> idx_vec;
  for (int64_t i = 0; i < 10; ++i) {
    idx_vec.push_back(i % 5);
  }
  at::Tensor index = tensor_from_vector_i64(idx_vec);
  at::Tensor source = at::full({10, 20}, 2.0f, options);

  at::Tensor result = self.index_reduce(0, index, source, "prod");

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "IndexReduceProdLargeShape ";
  write_index_reduce_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// index_reduce on dim=1
TEST_F(IndexReduceTest, IndexReduceDim1) {
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCPU);
  at::Tensor self = at::zeros({4, 6}, options);
  at::Tensor index = tensor_from_vector_i64({1, 3, 5});
  at::Tensor source = at::full({4, 3}, 4.0f, options);

  at::Tensor result = self.index_reduce(1, index, source, "mean");

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "IndexReduceDim1 ";
  write_index_reduce_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

TEST_F(IndexReduceTest, IndexReduceNegativeDim) {
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCPU);
  at::Tensor self = at::zeros({4, 6}, options);
  at::Tensor index = tensor_from_vector_i64({1, 3, 5});
  at::Tensor source = at::full({4, 3}, 4.0f, options);

  at::Tensor result = self.index_reduce(-1, index, source, "mean");

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "IndexReduceNegativeDim ";
  write_index_reduce_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

TEST_F(IndexReduceTest, IndexReduceInt32Index) {
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCPU);
  at::Tensor self = at::ones({3, 4}, options);
  at::Tensor index = tensor_from_vector_i32({0, 2});
  at::Tensor source = at::full({2, 4}, 2.0f, options);

  at::Tensor result = self.index_reduce(0, index, source, "prod");

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "IndexReduceInt32Index ";
  write_index_reduce_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// index_reduce with boundary shape (1D tensor)
TEST_F(IndexReduceTest, IndexReduce1D) {
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCPU);
  at::Tensor self = at::ones({5}, options);
  at::Tensor index = tensor_from_vector_i64({0, 2, 4});
  at::Tensor source = at::full({3}, 3.0f, options);

  at::Tensor result = self.index_reduce(0, index, source, "prod");

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "IndexReduce1D ";
  write_index_reduce_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// index_reduce non-member function version
TEST_F(IndexReduceTest, IndexReduceNonMember) {
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCPU);
  at::Tensor self = at::ones({3, 4}, options);
  at::Tensor index = tensor_from_vector_i64({0, 2});
  at::Tensor source = at::full({2, 4}, 2.0f, options);

  at::Tensor result = at::index_reduce(self, 0, index, source, "prod");

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "IndexReduceNonMember ";
  write_index_reduce_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// index_reduce invalid reduce type - should throw
TEST_F(IndexReduceTest, IndexReduceInvalidReduce) {
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCPU);
  at::Tensor self = at::ones({3, 4}, options);
  at::Tensor index = tensor_from_vector_i64({0, 2});
  at::Tensor source = at::full({2, 4}, 2.0f, options);

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "IndexReduceInvalidReduce ";
  try {
    (void)self.index_reduce(0, index, source, "invalid");
    file << "no_exception ";
  } catch (const std::exception&) {
    file << "exception ";
  }
  file << "\n";
  file.saveFile();
}

TEST_F(IndexReduceTest, IndexReduceSumReduceThrows) {
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCPU);
  at::Tensor self = at::ones({3, 4}, options);
  at::Tensor index = tensor_from_vector_i64({0, 2});
  at::Tensor source = at::full({2, 4}, 2.0f, options);

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "IndexReduceSumReduceThrows ";
  try {
    (void)self.index_reduce(0, index, source, "sum");
    file << "no_exception ";
  } catch (const std::exception&) {
    file << "exception ";
  }
  file << "\n";
  file.saveFile();
}

TEST_F(IndexReduceTest, IndexReducePaddleReduceNameThrows) {
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCPU);
  at::Tensor self = at::ones({3, 4}, options);
  at::Tensor index = tensor_from_vector_i64({0, 2});
  at::Tensor source = at::full({2, 4}, 2.0f, options);

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "IndexReducePaddleReduceNameThrows ";
  const char* reduce_names[] = {"add", "assign", "multiply"};
  for (const char* reduce : reduce_names) {
    try {
      (void)self.index_reduce(0, index, source, reduce);
      file << "no_exception ";
    } catch (const std::exception&) {
      file << "exception ";
    }
  }
  file << "\n";
  file.saveFile();
}

TEST_F(IndexReduceTest, IndexReduceFloatIndexThrows) {
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCPU);
  at::Tensor self = at::ones({3, 4}, options);
  at::Tensor index = tensor_from_vector_f32({0.0f, 2.0f});
  at::Tensor source = at::full({2, 4}, 2.0f, options);

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "IndexReduceFloatIndexThrows ";
  try {
    (void)self.index_reduce(0, index, source, "prod");
    file << "no_exception ";
  } catch (const std::exception&) {
    file << "exception ";
  }
  file << "\n";
  file.saveFile();
}

TEST_F(IndexReduceTest, IndexReduceNegativeIndexThrows) {
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCPU);
  at::Tensor self = at::ones({3, 4}, options);
  at::Tensor index = tensor_from_vector_i64({0, -1});
  at::Tensor source = at::full({2, 4}, 2.0f, options);

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "IndexReduceNegativeIndexThrows ";
  try {
    (void)self.index_reduce(0, index, source, "prod");
    file << "no_exception ";
  } catch (const std::exception&) {
    file << "exception ";
  }
  file << "\n";
  file.saveFile();
}

TEST_F(IndexReduceTest, IndexReduceUpperBoundIndexThrows) {
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCPU);
  at::Tensor self = at::ones({3, 4}, options);
  at::Tensor index = tensor_from_vector_i32({0, 3});
  at::Tensor source = at::full({2, 4}, 2.0f, options);

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "IndexReduceUpperBoundIndexThrows ";
  try {
    (void)self.index_reduce(0, index, source, "prod");
    file << "no_exception ";
  } catch (const std::exception&) {
    file << "exception ";
  }
  file << "\n";
  file.saveFile();
}

TEST_F(IndexReduceTest, IndexReduceSourceDtypeMismatchThrows) {
  auto self_options = at::TensorOptions().dtype(at::kFloat).device(at::kCPU);
  auto source_options = at::TensorOptions().dtype(at::kDouble).device(at::kCPU);
  at::Tensor self = at::ones({3, 4}, self_options);
  at::Tensor index = tensor_from_vector_i64({0, 2});
  at::Tensor source = at::full({2, 4}, 2.0, source_options);

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "IndexReduceSourceDtypeMismatchThrows ";
  try {
    (void)self.index_reduce(0, index, source, "prod");
    file << "no_exception ";
  } catch (const std::exception&) {
    file << "exception ";
  }
  file << "\n";
  file.saveFile();
}

TEST_F(IndexReduceTest, IndexReduceOutWritesOutput) {
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCPU);
  at::Tensor self = at::ones({3, 4}, options);
  at::Tensor index = tensor_from_vector_i64({0, 2});
  at::Tensor source = at::full({2, 4}, 2.0f, options);
  at::Tensor out = at::zeros({3, 4}, options);

  at::index_reduce_out(out, self, 0, index, source, "prod", false);

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "IndexReduceOutWritesOutput ";
  write_index_reduce_result_to_file(&file, out);
  file << "\n";
  file.saveFile();
}

}  // namespace test
}  // namespace at
