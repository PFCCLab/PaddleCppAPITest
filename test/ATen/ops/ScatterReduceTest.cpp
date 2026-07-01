#include <ATen/ATen.h>
#include <ATen/core/Tensor.h>
#include <ATen/ops/scatter_reduce.h>
#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "src/file_manager.h"

extern paddle_api_test::ThreadSafeParam g_custom_param;

namespace at {
namespace test {

using paddle_api_test::FileManerger;

class ScatterReduceTest : public ::testing::Test {
 protected:
  void SetUp() override {}
};

static void write_scatter_reduce_result_to_file(FileManerger* file,
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
  if (cont.scalar_type() == c10::ScalarType::Float) {
    float* data = cont.data_ptr<float>();
    *file << std::to_string(data[0]) << " ";
    *file << std::to_string(data[cont.numel() - 1]) << " ";
    *file << std::to_string(cont.sum().item<float>()) << " ";
  } else if (cont.scalar_type() == c10::ScalarType::Double) {
    double* data = cont.data_ptr<double>();
    *file << std::to_string(data[0]) << " ";
    *file << std::to_string(data[cont.numel() - 1]) << " ";
    *file << std::to_string(cont.sum().item<double>()) << " ";
  } else if (cont.scalar_type() == c10::ScalarType::Int) {
    int* data = cont.data_ptr<int>();
    *file << std::to_string(data[0]) << " ";
    *file << std::to_string(data[cont.numel() - 1]) << " ";
    *file << std::to_string(cont.sum().item<int>()) << " ";
  } else if (cont.scalar_type() == c10::ScalarType::Long) {
    int64_t* data = cont.data_ptr<int64_t>();
    *file << std::to_string(data[0]) << " ";
    *file << std::to_string(data[cont.numel() - 1]) << " ";
    *file << std::to_string(cont.sum().item<int64_t>()) << " ";
  }
}

static at::Tensor make_index_1x5() {
  at::Tensor index = at::zeros({1, 5}, at::kLong);
  index.data_ptr<int64_t>()[0] = 0;
  index.data_ptr<int64_t>()[1] = 1;
  index.data_ptr<int64_t>()[2] = 2;
  index.data_ptr<int64_t>()[3] = 0;
  index.data_ptr<int64_t>()[4] = 0;
  return index;
}

// Shape: small 2D, Dtype: kFloat, Reduce: sum
TEST_F(ScatterReduceTest, ScatterReduceSumFloatSmall) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();
  file << "ScatterReduceSumFloatSmall ";

  at::Tensor self = at::zeros({3, 5}, at::kFloat);
  at::Tensor index = make_index_1x5();
  at::Tensor src = at::full({1, 5}, 1.0f, at::kFloat);
  src.data_ptr<float>()[0] = 1.0f;
  src.data_ptr<float>()[1] = 2.0f;
  src.data_ptr<float>()[2] = 3.0f;
  src.data_ptr<float>()[3] = 4.0f;
  src.data_ptr<float>()[4] = 5.0f;
  at::Tensor result = self.scatter_reduce(0, index, src, "sum");
  write_scatter_reduce_result_to_file(&file, result);

  file << "\n";
  file.saveFile();
}

// API variant: free function at::scatter_reduce.
TEST_F(ScatterReduceTest, ScatterReduceFreeFunctionSumFloatSmall) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "ScatterReduceFreeFunctionSumFloatSmall ";

  at::Tensor self = at::zeros({3, 5}, at::kFloat);
  at::Tensor index = make_index_1x5();
  at::Tensor src = at::full({1, 5}, 1.0f, at::kFloat);
  src.data_ptr<float>()[0] = 1.0f;
  src.data_ptr<float>()[1] = 2.0f;
  src.data_ptr<float>()[2] = 3.0f;
  src.data_ptr<float>()[3] = 4.0f;
  src.data_ptr<float>()[4] = 5.0f;
  at::Tensor result = at::scatter_reduce(self, 0, index, src, "sum");
  write_scatter_reduce_result_to_file(&file, result);

  file << "\n";
  file.saveFile();
}

// Shape: small 2D, Dtype: kDouble, Reduce: sum
TEST_F(ScatterReduceTest, ScatterReduceSumDoubleSmall) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "ScatterReduceSumDoubleSmall ";

  at::Tensor self = at::zeros({3, 5}, at::kDouble);
  at::Tensor index = make_index_1x5();
  at::Tensor src = at::full({1, 5}, 1.0, at::kDouble);
  src.data_ptr<double>()[0] = 1.0;
  src.data_ptr<double>()[1] = 2.0;
  src.data_ptr<double>()[2] = 3.0;
  src.data_ptr<double>()[3] = 4.0;
  src.data_ptr<double>()[4] = 5.0;
  at::Tensor result = self.scatter_reduce(0, index, src, "sum");
  write_scatter_reduce_result_to_file(&file, result);

  file << "\n";
  file.saveFile();
}

// Shape: small 2D, Dtype: kInt, Reduce: sum
TEST_F(ScatterReduceTest, ScatterReduceSumIntSmall) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "ScatterReduceSumIntSmall ";

  at::Tensor self = at::zeros({3, 5}, at::kInt);
  at::Tensor index = make_index_1x5();
  at::Tensor src = at::full({1, 5}, 1, at::kInt);
  src.data_ptr<int>()[0] = 1;
  src.data_ptr<int>()[1] = 2;
  src.data_ptr<int>()[2] = 3;
  src.data_ptr<int>()[3] = 4;
  src.data_ptr<int>()[4] = 5;
  at::Tensor result = self.scatter_reduce(0, index, src, "sum");
  write_scatter_reduce_result_to_file(&file, result);

  file << "\n";
  file.saveFile();
}

// Shape: small 2D, Dtype: kLong, Reduce: sum
TEST_F(ScatterReduceTest, ScatterReduceSumLongSmall) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "ScatterReduceSumLongSmall ";

  at::Tensor self = at::zeros({3, 5}, at::kLong);
  at::Tensor index = make_index_1x5();
  at::Tensor src = at::full({1, 5}, 1, at::kLong);
  src.data_ptr<int64_t>()[0] = 1;
  src.data_ptr<int64_t>()[1] = 2;
  src.data_ptr<int64_t>()[2] = 3;
  src.data_ptr<int64_t>()[3] = 4;
  src.data_ptr<int64_t>()[4] = 5;
  at::Tensor result = self.scatter_reduce(0, index, src, "sum");
  write_scatter_reduce_result_to_file(&file, result);

  file << "\n";
  file.saveFile();
}

// Shape: medium 2D, Dtype: kFloat, Reduce: sum
TEST_F(ScatterReduceTest, ScatterReduceSumFloatMedium) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "ScatterReduceSumFloatMedium ";

  at::Tensor self = at::zeros({6, 10}, at::kFloat);
  at::Tensor index = at::zeros({2, 10}, at::kLong);
  for (int64_t j = 0; j < 10; ++j) {
    index.data_ptr<int64_t>()[j] = j % 6;
    index.data_ptr<int64_t>()[10 + j] = (j + 3) % 6;
  }
  at::Tensor src = at::ones({2, 10}, at::kFloat);
  at::Tensor result = self.scatter_reduce(0, index, src, "sum");
  write_scatter_reduce_result_to_file(&file, result);

  file << "\n";
  file.saveFile();
}

// Shape: large 2D, Dtype: kFloat, Reduce: sum
TEST_F(ScatterReduceTest, ScatterReduceSumFloatLarge) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "ScatterReduceSumFloatLarge ";

  at::Tensor self = at::zeros({10, 20}, at::kFloat);
  at::Tensor index = at::zeros({4, 20}, at::kLong);
  for (int64_t j = 0; j < 20; ++j) {
    index.data_ptr<int64_t>()[j] = j % 10;
    index.data_ptr<int64_t>()[20 + j] = (j + 2) % 10;
    index.data_ptr<int64_t>()[40 + j] = (j + 5) % 10;
    index.data_ptr<int64_t>()[60 + j] = (j + 7) % 10;
  }
  at::Tensor src = at::ones({4, 20}, at::kFloat);
  at::Tensor result = self.scatter_reduce(0, index, src, "sum");
  write_scatter_reduce_result_to_file(&file, result);

  file << "\n";
  file.saveFile();
}

// Exception: PyTorch scatter_reduce does not support replace mode.
TEST_F(ScatterReduceTest, ScatterReduceReplaceFloatSmall) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "ScatterReduceReplaceFloatSmall ";

  try {
    at::Tensor self = at::zeros({3, 5}, at::kFloat);
    at::Tensor index = make_index_1x5();
    at::Tensor src = at::full({1, 5}, 1.0f, at::kFloat);
    src.data_ptr<float>()[0] = 1.0f;
    src.data_ptr<float>()[1] = 2.0f;
    src.data_ptr<float>()[2] = 3.0f;
    src.data_ptr<float>()[3] = 4.0f;
    src.data_ptr<float>()[4] = 5.0f;
    at::Tensor result = self.scatter_reduce(0, index, src, "replace");
    write_scatter_reduce_result_to_file(&file, result);
  } catch (const std::exception&) {
    file << "exception ";
  }

  file << "\n";
  file.saveFile();
}

// Shape: small 2D, Dtype: kFloat, Reduce: amax
TEST_F(ScatterReduceTest, ScatterReduceAmaxFloatSmall) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "ScatterReduceAmaxFloatSmall ";

  at::Tensor self = at::zeros({3, 5}, at::kFloat);
  at::Tensor index = make_index_1x5();
  at::Tensor src = at::full({1, 5}, 1.0f, at::kFloat);
  src.data_ptr<float>()[0] = 10.0f;
  src.data_ptr<float>()[1] = 20.0f;
  src.data_ptr<float>()[2] = 30.0f;
  src.data_ptr<float>()[3] = 40.0f;
  src.data_ptr<float>()[4] = 50.0f;
  at::Tensor result = self.scatter_reduce(0, index, src, "amax");
  write_scatter_reduce_result_to_file(&file, result);

  file << "\n";
  file.saveFile();
}

// Shape: small 2D, Dtype: kFloat, Reduce: amin
TEST_F(ScatterReduceTest, ScatterReduceAminFloatSmall) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "ScatterReduceAminFloatSmall ";

  at::Tensor self = at::full({3, 5}, 100.0f, at::kFloat);
  at::Tensor index = make_index_1x5();
  at::Tensor src = at::full({1, 5}, 1.0f, at::kFloat);
  src.data_ptr<float>()[0] = 10.0f;
  src.data_ptr<float>()[1] = 20.0f;
  src.data_ptr<float>()[2] = 30.0f;
  src.data_ptr<float>()[3] = 40.0f;
  src.data_ptr<float>()[4] = 50.0f;
  at::Tensor result = self.scatter_reduce(0, index, src, "amin");
  write_scatter_reduce_result_to_file(&file, result);

  file << "\n";
  file.saveFile();
}

// Shape: small 2D, Dtype: kFloat, Reduce: mean
TEST_F(ScatterReduceTest, ScatterReduceMeanFloatSmall) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "ScatterReduceMeanFloatSmall ";

  at::Tensor self = at::zeros({3, 5}, at::kFloat);
  at::Tensor index = make_index_1x5();
  at::Tensor src = at::full({1, 5}, 1.0f, at::kFloat);
  src.data_ptr<float>()[0] = 10.0f;
  src.data_ptr<float>()[1] = 20.0f;
  src.data_ptr<float>()[2] = 30.0f;
  src.data_ptr<float>()[3] = 40.0f;
  src.data_ptr<float>()[4] = 50.0f;
  at::Tensor result = self.scatter_reduce(0, index, src, "mean");
  write_scatter_reduce_result_to_file(&file, result);

  file << "\n";
  file.saveFile();
}

// Shape: small 2D, Dtype: kFloat, Reduce: prod
TEST_F(ScatterReduceTest, ScatterReduceProdFloatSmall) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "ScatterReduceProdFloatSmall ";

  at::Tensor self = at::ones({3, 5}, at::kFloat);
  at::Tensor index = make_index_1x5();
  at::Tensor src = at::full({1, 5}, 1.0f, at::kFloat);
  src.data_ptr<float>()[0] = 2.0f;
  src.data_ptr<float>()[1] = 3.0f;
  src.data_ptr<float>()[2] = 4.0f;
  src.data_ptr<float>()[3] = 5.0f;
  src.data_ptr<float>()[4] = 6.0f;
  at::Tensor result = self.scatter_reduce(0, index, src, "prod");
  write_scatter_reduce_result_to_file(&file, result);

  file << "\n";
  file.saveFile();
}

// Shape: small 2D, Dtype: kFloat, Reduce: sum, dim=1
TEST_F(ScatterReduceTest, ScatterReduceSumFloatDim1) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "ScatterReduceSumFloatDim1 ";

  at::Tensor self = at::zeros({2, 4}, at::kFloat);
  at::Tensor index = at::zeros({2, 4}, at::kLong);
  index.data_ptr<int64_t>()[0] = 0;
  index.data_ptr<int64_t>()[1] = 1;
  index.data_ptr<int64_t>()[2] = 2;
  index.data_ptr<int64_t>()[3] = 1;
  index.data_ptr<int64_t>()[4] = 3;
  index.data_ptr<int64_t>()[5] = 0;
  index.data_ptr<int64_t>()[6] = 1;
  index.data_ptr<int64_t>()[7] = 2;
  at::Tensor src = at::full({2, 4}, 1.0f, at::kFloat);
  src.data_ptr<float>()[0] = 1.0f;
  src.data_ptr<float>()[1] = 2.0f;
  src.data_ptr<float>()[2] = 3.0f;
  src.data_ptr<float>()[3] = 4.0f;
  src.data_ptr<float>()[4] = 5.0f;
  src.data_ptr<float>()[5] = 6.0f;
  src.data_ptr<float>()[6] = 7.0f;
  src.data_ptr<float>()[7] = 8.0f;
  at::Tensor result = self.scatter_reduce(1, index, src, "sum");
  write_scatter_reduce_result_to_file(&file, result);

  file << "\n";
  file.saveFile();
}

// Shape: small 2D, Dtype: kFloat, Reduce: sum, dim=-1
TEST_F(ScatterReduceTest, ScatterReduceSumFloatNegativeDim) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "ScatterReduceSumFloatNegativeDim ";

  at::Tensor self = at::zeros({2, 4}, at::kFloat);
  at::Tensor index = at::zeros({2, 4}, at::kLong);
  index.data_ptr<int64_t>()[0] = 0;
  index.data_ptr<int64_t>()[1] = 1;
  index.data_ptr<int64_t>()[2] = 2;
  index.data_ptr<int64_t>()[3] = 1;
  index.data_ptr<int64_t>()[4] = 3;
  index.data_ptr<int64_t>()[5] = 0;
  index.data_ptr<int64_t>()[6] = 1;
  index.data_ptr<int64_t>()[7] = 2;
  at::Tensor src = at::full({2, 4}, 1.0f, at::kFloat);
  src.data_ptr<float>()[0] = 1.0f;
  src.data_ptr<float>()[1] = 2.0f;
  src.data_ptr<float>()[2] = 3.0f;
  src.data_ptr<float>()[3] = 4.0f;
  src.data_ptr<float>()[4] = 5.0f;
  src.data_ptr<float>()[5] = 6.0f;
  src.data_ptr<float>()[6] = 7.0f;
  src.data_ptr<float>()[7] = 8.0f;
  at::Tensor result = self.scatter_reduce(-1, index, src, "sum");
  write_scatter_reduce_result_to_file(&file, result);

  file << "\n";
  file.saveFile();
}

// Shape: small 2D, Dtype: kFloat, In-place scatter_reduce_, dim=-1
TEST_F(ScatterReduceTest, ScatterReduceInplaceFloatNegativeDim) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "ScatterReduceInplaceFloatNegativeDim ";

  at::Tensor self = at::zeros({2, 4}, at::kFloat);
  at::Tensor index = at::zeros({2, 4}, at::kLong);
  index.data_ptr<int64_t>()[0] = 0;
  index.data_ptr<int64_t>()[1] = 1;
  index.data_ptr<int64_t>()[2] = 2;
  index.data_ptr<int64_t>()[3] = 1;
  index.data_ptr<int64_t>()[4] = 3;
  index.data_ptr<int64_t>()[5] = 0;
  index.data_ptr<int64_t>()[6] = 1;
  index.data_ptr<int64_t>()[7] = 2;
  at::Tensor src = at::full({2, 4}, 1.0f, at::kFloat);
  src.data_ptr<float>()[0] = 1.0f;
  src.data_ptr<float>()[1] = 2.0f;
  src.data_ptr<float>()[2] = 3.0f;
  src.data_ptr<float>()[3] = 4.0f;
  src.data_ptr<float>()[4] = 5.0f;
  src.data_ptr<float>()[5] = 6.0f;
  src.data_ptr<float>()[6] = 7.0f;
  src.data_ptr<float>()[7] = 8.0f;
  self.scatter_reduce_(-1, index, src, "sum");
  write_scatter_reduce_result_to_file(&file, self);

  file << "\n";
  file.saveFile();
}

// Shape: small 2D, Dtype: kFloat, Reduce: sum, include_self=false
TEST_F(ScatterReduceTest, ScatterReduceSumFloatNoIncludeSelf) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "ScatterReduceSumFloatNoIncludeSelf ";

  at::Tensor self = at::full({3, 5}, 5.0f, at::kFloat);
  at::Tensor index = make_index_1x5();
  at::Tensor src = at::full({1, 5}, 1.0f, at::kFloat);
  src.data_ptr<float>()[0] = 1.0f;
  src.data_ptr<float>()[1] = 2.0f;
  src.data_ptr<float>()[2] = 3.0f;
  src.data_ptr<float>()[3] = 4.0f;
  src.data_ptr<float>()[4] = 5.0f;
  at::Tensor result = self.scatter_reduce(0, index, src, "sum", false);
  write_scatter_reduce_result_to_file(&file, result);

  file << "\n";
  file.saveFile();
}

// Exception: PyTorch scatter_reduce rejects negative indices.
TEST_F(ScatterReduceTest, ScatterReduceNegativeIndex) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "ScatterReduceNegativeIndex ";

  try {
    at::Tensor self = at::zeros({2, 4}, at::kFloat);
    at::Tensor index = at::zeros({2, 4}, at::kLong);
    index.data_ptr<int64_t>()[0] = 0;
    index.data_ptr<int64_t>()[1] = -1;
    index.data_ptr<int64_t>()[2] = 2;
    index.data_ptr<int64_t>()[3] = 1;
    index.data_ptr<int64_t>()[4] = 3;
    index.data_ptr<int64_t>()[5] = 0;
    index.data_ptr<int64_t>()[6] = 1;
    index.data_ptr<int64_t>()[7] = 2;
    at::Tensor src = at::ones({2, 4}, at::kFloat);
    at::Tensor result = self.scatter_reduce(1, index, src, "sum");
    write_scatter_reduce_result_to_file(&file, result);
  } catch (const std::exception&) {
    file << "exception ";
  }

  file << "\n";
  file.saveFile();
}

// Exception: PyTorch scatter_reduce_ rejects negative indices.
TEST_F(ScatterReduceTest, ScatterReduceInplaceNegativeIndex) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "ScatterReduceInplaceNegativeIndex ";

  try {
    at::Tensor self = at::zeros({2, 4}, at::kFloat);
    at::Tensor index = at::zeros({2, 4}, at::kLong);
    index.data_ptr<int64_t>()[0] = 0;
    index.data_ptr<int64_t>()[1] = -1;
    index.data_ptr<int64_t>()[2] = 2;
    index.data_ptr<int64_t>()[3] = 1;
    index.data_ptr<int64_t>()[4] = 3;
    index.data_ptr<int64_t>()[5] = 0;
    index.data_ptr<int64_t>()[6] = 1;
    index.data_ptr<int64_t>()[7] = 2;
    at::Tensor src = at::ones({2, 4}, at::kFloat);
    self.scatter_reduce_(1, index, src, "sum");
    write_scatter_reduce_result_to_file(&file, self);
  } catch (const std::exception&) {
    file << "exception ";
  }

  file << "\n";
  file.saveFile();
}

// Exception: PyTorch scatter_reduce rejects index >= self.size(dim).
TEST_F(ScatterReduceTest, ScatterReduceIndexUpperBound) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "ScatterReduceIndexUpperBound ";

  try {
    at::Tensor self = at::zeros({2, 4}, at::kFloat);
    at::Tensor index = at::zeros({2, 4}, at::kLong);
    index.data_ptr<int64_t>()[0] = 0;
    index.data_ptr<int64_t>()[1] = 4;
    index.data_ptr<int64_t>()[2] = 2;
    index.data_ptr<int64_t>()[3] = 1;
    index.data_ptr<int64_t>()[4] = 3;
    index.data_ptr<int64_t>()[5] = 0;
    index.data_ptr<int64_t>()[6] = 1;
    index.data_ptr<int64_t>()[7] = 2;
    at::Tensor src = at::ones({2, 4}, at::kFloat);
    at::Tensor result = self.scatter_reduce(1, index, src, "sum");
    write_scatter_reduce_result_to_file(&file, result);
  } catch (const std::exception&) {
    file << "exception ";
  }

  file << "\n";
  file.saveFile();
}

// Exception: PyTorch scatter_reduce_ rejects index >= self.size(dim).
TEST_F(ScatterReduceTest, ScatterReduceInplaceIndexUpperBound) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "ScatterReduceInplaceIndexUpperBound ";

  try {
    at::Tensor self = at::zeros({2, 4}, at::kFloat);
    at::Tensor index = at::zeros({2, 4}, at::kLong);
    index.data_ptr<int64_t>()[0] = 0;
    index.data_ptr<int64_t>()[1] = 4;
    index.data_ptr<int64_t>()[2] = 2;
    index.data_ptr<int64_t>()[3] = 1;
    index.data_ptr<int64_t>()[4] = 3;
    index.data_ptr<int64_t>()[5] = 0;
    index.data_ptr<int64_t>()[6] = 1;
    index.data_ptr<int64_t>()[7] = 2;
    at::Tensor src = at::ones({2, 4}, at::kFloat);
    self.scatter_reduce_(1, index, src, "sum");
    write_scatter_reduce_result_to_file(&file, self);
  } catch (const std::exception&) {
    file << "exception ";
  }

  file << "\n";
  file.saveFile();
}

// PyTorch C++ scatter_reduce accepts int32 index tensors.
TEST_F(ScatterReduceTest, ScatterReduceIntIndex) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "ScatterReduceIntIndex ";

  at::Tensor self = at::zeros({3, 5}, at::kFloat);
  at::Tensor index = at::zeros({1, 5}, at::kInt);
  index.data_ptr<int>()[0] = 0;
  index.data_ptr<int>()[1] = 1;
  index.data_ptr<int>()[2] = 2;
  index.data_ptr<int>()[3] = 0;
  index.data_ptr<int>()[4] = 0;
  at::Tensor src = at::full({1, 5}, 1.0f, at::kFloat);
  src.data_ptr<float>()[0] = 1.0f;
  src.data_ptr<float>()[1] = 2.0f;
  src.data_ptr<float>()[2] = 3.0f;
  src.data_ptr<float>()[3] = 4.0f;
  src.data_ptr<float>()[4] = 5.0f;
  at::Tensor result = self.scatter_reduce(0, index, src, "sum");
  write_scatter_reduce_result_to_file(&file, result);

  file << "\n";
  file.saveFile();
}

// PyTorch C++ scatter_reduce_ accepts int32 index tensors.
TEST_F(ScatterReduceTest, ScatterReduceInplaceIntIndex) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "ScatterReduceInplaceIntIndex ";

  at::Tensor self = at::zeros({3, 5}, at::kFloat);
  at::Tensor index = at::zeros({1, 5}, at::kInt);
  index.data_ptr<int>()[0] = 0;
  index.data_ptr<int>()[1] = 1;
  index.data_ptr<int>()[2] = 2;
  index.data_ptr<int>()[3] = 0;
  index.data_ptr<int>()[4] = 0;
  at::Tensor src = at::full({1, 5}, 1.0f, at::kFloat);
  src.data_ptr<float>()[0] = 1.0f;
  src.data_ptr<float>()[1] = 2.0f;
  src.data_ptr<float>()[2] = 3.0f;
  src.data_ptr<float>()[3] = 4.0f;
  src.data_ptr<float>()[4] = 5.0f;
  self.scatter_reduce_(0, index, src, "sum");
  write_scatter_reduce_result_to_file(&file, self);

  file << "\n";
  file.saveFile();
}

// PyTorch C++ scatter_reduce rejects non-empty floating index tensors.
TEST_F(ScatterReduceTest, ScatterReduceFloatIndex) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "ScatterReduceFloatIndex ";

  try {
    at::Tensor self = at::zeros({2, 2}, at::kFloat);
    at::Tensor index = at::zeros({1, 2}, at::kFloat);
    at::Tensor src = at::ones({1, 2}, at::kFloat);
    at::Tensor result = self.scatter_reduce(0, index, src, "sum");
    write_scatter_reduce_result_to_file(&file, result);
  } catch (const std::exception&) {
    file << "exception ";
  }

  file << "\n";
  file.saveFile();
}

// PyTorch C++ scatter_reduce_ rejects non-empty floating index tensors.
TEST_F(ScatterReduceTest, ScatterReduceInplaceFloatIndex) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "ScatterReduceInplaceFloatIndex ";

  try {
    at::Tensor self = at::zeros({2, 2}, at::kFloat);
    at::Tensor index = at::zeros({1, 2}, at::kFloat);
    at::Tensor src = at::ones({1, 2}, at::kFloat);
    self.scatter_reduce_(0, index, src, "sum");
    write_scatter_reduce_result_to_file(&file, self);
  } catch (const std::exception&) {
    file << "exception ";
  }

  file << "\n";
  file.saveFile();
}

// Shape: small 2D, Dtype: kFloat, In-place scatter_reduce_
TEST_F(ScatterReduceTest, ScatterReduceInplaceFloatSmall) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "ScatterReduceInplaceFloatSmall ";

  at::Tensor self = at::zeros({3, 5}, at::kFloat);
  at::Tensor index = make_index_1x5();
  at::Tensor src = at::full({1, 5}, 1.0f, at::kFloat);
  src.data_ptr<float>()[0] = 1.0f;
  src.data_ptr<float>()[1] = 2.0f;
  src.data_ptr<float>()[2] = 3.0f;
  src.data_ptr<float>()[3] = 4.0f;
  src.data_ptr<float>()[4] = 5.0f;
  self.scatter_reduce_(0, index, src, "sum");
  write_scatter_reduce_result_to_file(&file, self);

  file << "\n";
  file.saveFile();
}

// Boundary: empty tensor
TEST_F(ScatterReduceTest, ScatterReduceEmpty) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "ScatterReduceEmpty ";

  at::Tensor self = at::zeros({0, 5}, at::kFloat);
  at::Tensor index = at::empty({0, 5}, at::kLong);
  at::Tensor src = at::empty({0, 5}, at::kFloat);
  at::Tensor result = self.scatter_reduce(0, index, src, "sum");
  write_scatter_reduce_result_to_file(&file, result);

  file << "\n";
  file.saveFile();
}

// Boundary: empty floating index tensor is accepted by PyTorch C++.
TEST_F(ScatterReduceTest, ScatterReduceEmptyFloatIndex) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "ScatterReduceEmptyFloatIndex ";

  at::Tensor self = at::zeros({0, 2}, at::kFloat);
  at::Tensor index = at::empty({0, 2}, at::kFloat);
  at::Tensor src = at::empty({0, 2}, at::kFloat);
  at::Tensor result = self.scatter_reduce(0, index, src, "sum");
  write_scatter_reduce_result_to_file(&file, result);

  file << "\n";
  file.saveFile();
}

TEST_F(ScatterReduceTest, ScatterReduceEmptyIndexRankMismatch) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "ScatterReduceEmptyIndexRankMismatch ";

  at::Tensor self = at::zeros({2, 2}, at::kFloat);
  float* self_data = self.data_ptr<float>();
  self_data[0] = 1.0f;
  self_data[1] = 2.0f;
  self_data[2] = 3.0f;
  self_data[3] = 4.0f;
  at::Tensor index = at::empty({0}, at::kLong);
  at::Tensor src = at::empty({0}, at::kFloat);
  at::Tensor result = self.scatter_reduce(0, index, src, "sum");
  write_scatter_reduce_result_to_file(&file, result);

  file << "\n";
  file.saveFile();
}

TEST_F(ScatterReduceTest, ScatterReduceInplaceEmptyIndexRankMismatch) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "ScatterReduceInplaceEmptyIndexRankMismatch ";

  at::Tensor self = at::zeros({2, 2}, at::kFloat);
  float* self_data = self.data_ptr<float>();
  self_data[0] = 1.0f;
  self_data[1] = 2.0f;
  self_data[2] = 3.0f;
  self_data[3] = 4.0f;
  at::Tensor index = at::empty({0}, at::kLong);
  at::Tensor src = at::empty({0}, at::kFloat);
  self.scatter_reduce_(0, index, src, "sum");
  write_scatter_reduce_result_to_file(&file, self);

  file << "\n";
  file.saveFile();
}

TEST_F(ScatterReduceTest, ScatterReduceEmptyIndexSrcDtypeMismatch) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "ScatterReduceEmptyIndexSrcDtypeMismatch ";

  try {
    at::Tensor self = at::zeros({2, 2}, at::kFloat);
    at::Tensor index = at::empty({0}, at::kLong);
    at::Tensor src = at::empty({0}, at::kInt);
    at::Tensor result = self.scatter_reduce(0, index, src, "sum");
    write_scatter_reduce_result_to_file(&file, result);
  } catch (const std::exception&) {
    file << "exception ";
  }

  file << "\n";
  file.saveFile();
}

TEST_F(ScatterReduceTest, ScatterReduceInplaceEmptyIndexSrcDtypeMismatch) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "ScatterReduceInplaceEmptyIndexSrcDtypeMismatch ";

  try {
    at::Tensor self = at::zeros({2, 2}, at::kFloat);
    at::Tensor index = at::empty({0}, at::kLong);
    at::Tensor src = at::empty({0}, at::kInt);
    self.scatter_reduce_(0, index, src, "sum");
    write_scatter_reduce_result_to_file(&file, self);
  } catch (const std::exception&) {
    file << "exception ";
  }

  file << "\n";
  file.saveFile();
}

// Shape: small 2D, Dtype: kFloat, Reduce: prod, include_self=false
TEST_F(ScatterReduceTest, ScatterReduceProdFloatNoIncludeSelf) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "ScatterReduceProdFloatNoIncludeSelf ";

  at::Tensor self = at::full({3, 5}, 5.0f, at::kFloat);
  at::Tensor index = make_index_1x5();
  at::Tensor src = at::full({1, 5}, 1.0f, at::kFloat);
  src.data_ptr<float>()[0] = 2.0f;
  src.data_ptr<float>()[1] = 3.0f;
  src.data_ptr<float>()[2] = 4.0f;
  src.data_ptr<float>()[3] = 5.0f;
  src.data_ptr<float>()[4] = 6.0f;
  at::Tensor result = self.scatter_reduce(0, index, src, "prod", false);
  write_scatter_reduce_result_to_file(&file, result);

  file << "\n";
  file.saveFile();
}

// Shape: small 2D, Dtype: kFloat, In-place scatter_reduce_, Reduce: prod
TEST_F(ScatterReduceTest, ScatterReduceInplaceProdFloatSmall) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "ScatterReduceInplaceProdFloatSmall ";

  at::Tensor self = at::ones({3, 5}, at::kFloat);
  at::Tensor index = make_index_1x5();
  at::Tensor src = at::full({1, 5}, 1.0f, at::kFloat);
  src.data_ptr<float>()[0] = 2.0f;
  src.data_ptr<float>()[1] = 3.0f;
  src.data_ptr<float>()[2] = 4.0f;
  src.data_ptr<float>()[3] = 5.0f;
  src.data_ptr<float>()[4] = 6.0f;
  self.scatter_reduce_(0, index, src, "prod");
  write_scatter_reduce_result_to_file(&file, self);

  file << "\n";
  file.saveFile();
}

// Shape: small 2D, Dtype: kFloat, Reduce: amax, include_self=false
TEST_F(ScatterReduceTest, ScatterReduceAmaxFloatNoIncludeSelf) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "ScatterReduceAmaxFloatNoIncludeSelf ";

  // Use self=25.0f so some src values are below self (10, 20) and some above
  // (30, 40, 50). This ensures amax with include_self=false differs from true.
  at::Tensor self = at::full({3, 5}, 25.0f, at::kFloat);
  at::Tensor index = make_index_1x5();
  at::Tensor src = at::full({1, 5}, 1.0f, at::kFloat);
  src.data_ptr<float>()[0] = 10.0f;
  src.data_ptr<float>()[1] = 20.0f;
  src.data_ptr<float>()[2] = 30.0f;
  src.data_ptr<float>()[3] = 40.0f;
  src.data_ptr<float>()[4] = 50.0f;
  at::Tensor result = self.scatter_reduce(0, index, src, "amax", false);
  write_scatter_reduce_result_to_file(&file, result);

  file << "\n";
  file.saveFile();
}

// Shape: small 2D, Dtype: kFloat, Reduce: amin, include_self=false
TEST_F(ScatterReduceTest, ScatterReduceAminFloatNoIncludeSelf) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "ScatterReduceAminFloatNoIncludeSelf ";

  at::Tensor self = at::full({3, 5}, 5.0f, at::kFloat);
  at::Tensor index = make_index_1x5();
  at::Tensor src = at::full({1, 5}, 1.0f, at::kFloat);
  src.data_ptr<float>()[0] = 10.0f;
  src.data_ptr<float>()[1] = 20.0f;
  src.data_ptr<float>()[2] = 30.0f;
  src.data_ptr<float>()[3] = 40.0f;
  src.data_ptr<float>()[4] = 50.0f;
  at::Tensor result = self.scatter_reduce(0, index, src, "amin", false);
  write_scatter_reduce_result_to_file(&file, result);

  file << "\n";
  file.saveFile();
}

// Shape: small 2D, Dtype: kFloat, Reduce: mean, include_self=false
TEST_F(ScatterReduceTest, ScatterReduceMeanFloatNoIncludeSelf) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "ScatterReduceMeanFloatNoIncludeSelf ";

  at::Tensor self = at::full({3, 5}, 5.0f, at::kFloat);
  at::Tensor index = make_index_1x5();
  at::Tensor src = at::full({1, 5}, 1.0f, at::kFloat);
  src.data_ptr<float>()[0] = 10.0f;
  src.data_ptr<float>()[1] = 20.0f;
  src.data_ptr<float>()[2] = 30.0f;
  src.data_ptr<float>()[3] = 40.0f;
  src.data_ptr<float>()[4] = 50.0f;
  at::Tensor result = self.scatter_reduce(0, index, src, "mean", false);
  write_scatter_reduce_result_to_file(&file, result);

  file << "\n";
  file.saveFile();
}

TEST_F(ScatterReduceTest, ScatterReduceRankMismatch) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "ScatterReduceRankMismatch ";

  try {
    at::Tensor self = at::zeros({2, 2}, at::kFloat);
    at::Tensor index = at::zeros({2}, at::kLong);
    at::Tensor src = at::ones({2, 2}, at::kFloat);
    at::Tensor result = self.scatter_reduce(0, index, src, "sum");
    write_scatter_reduce_result_to_file(&file, result);
  } catch (const std::exception&) {
    file << "exception ";
  }

  file << "\n";
  file.saveFile();
}

TEST_F(ScatterReduceTest, ScatterReduceIndexLargerThanSrc) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "ScatterReduceIndexLargerThanSrc ";

  try {
    at::Tensor self = at::zeros({3, 2}, at::kFloat);
    at::Tensor index = at::zeros({3, 2}, at::kLong);
    at::Tensor src = at::ones({2, 2}, at::kFloat);
    at::Tensor result = self.scatter_reduce(0, index, src, "sum");
    write_scatter_reduce_result_to_file(&file, result);
  } catch (const std::exception&) {
    file << "exception ";
  }

  file << "\n";
  file.saveFile();
}

TEST_F(ScatterReduceTest, ScatterReduceIndexLargerThanSelf) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "ScatterReduceIndexLargerThanSelf ";

  try {
    at::Tensor self = at::zeros({2, 2}, at::kFloat);
    at::Tensor index = at::zeros({1, 3}, at::kLong);
    at::Tensor src = at::ones({1, 3}, at::kFloat);
    at::Tensor result = self.scatter_reduce(0, index, src, "sum");
    write_scatter_reduce_result_to_file(&file, result);
  } catch (const std::exception&) {
    file << "exception ";
  }

  file << "\n";
  file.saveFile();
}

TEST_F(ScatterReduceTest, ScatterReduceInplaceShapeMismatch) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "ScatterReduceInplaceShapeMismatch ";

  try {
    at::Tensor self = at::zeros({2, 2}, at::kFloat);
    at::Tensor index = at::zeros({1, 3}, at::kLong);
    at::Tensor src = at::ones({1, 3}, at::kFloat);
    at::Tensor result = self.scatter_reduce_(0, index, src, "sum");
    write_scatter_reduce_result_to_file(&file, result);
  } catch (const std::exception&) {
    file << "exception ";
  }

  file << "\n";
  file.saveFile();
}

// Exception: invalid reduce mode
TEST_F(ScatterReduceTest, ScatterReduceInvalidReduce) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "ScatterReduceInvalidReduce ";

  try {
    at::Tensor self = at::zeros({3, 5}, at::kFloat);
    at::Tensor index = make_index_1x5();
    at::Tensor src = at::full({1, 5}, 1.0f, at::kFloat);
    src.data_ptr<float>()[0] = 1.0f;
    src.data_ptr<float>()[1] = 2.0f;
    src.data_ptr<float>()[2] = 3.0f;
    src.data_ptr<float>()[3] = 4.0f;
    src.data_ptr<float>()[4] = 5.0f;
    at::Tensor result = self.scatter_reduce(0, index, src, "invalid");
    write_scatter_reduce_result_to_file(&file, result);
  } catch (const std::exception&) {
    file << "exception ";
  }

  file << "\n";
  file.saveFile();
}

// Exception: dim out of int range
TEST_F(ScatterReduceTest, ScatterReduceDimOutOfRange) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "ScatterReduceDimOutOfRange ";

  try {
    at::Tensor self = at::zeros({2, 2}, at::kFloat);
    at::Tensor index = make_index_1x5();
    at::Tensor src = at::ones({1, 5}, at::kFloat);
    at::Tensor result = self.scatter_reduce(
        static_cast<int64_t>(INT_MAX) + 1, index, src, "sum");
    write_scatter_reduce_result_to_file(&file, result);
  } catch (const std::exception&) {
    file << "exception ";
  }

  file << "\n";
  file.saveFile();
}

}  // namespace test
}  // namespace at
