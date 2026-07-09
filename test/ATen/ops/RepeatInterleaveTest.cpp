#include <ATen/ATen.h>
#include <ATen/core/Tensor.h>
#include <ATen/ops/arange.h>
#include <ATen/ops/ones.h>
#include <ATen/ops/zeros.h>
#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "src/file_manager.h"

extern paddle_api_test::ThreadSafeParam g_custom_param;

namespace at {
namespace test {

using paddle_api_test::FileManerger;
using paddle_api_test::ThreadSafeParam;

static void write_repeat_interleave_result_to_file(FileManerger* file,
                                                   const at::Tensor& result) {
  *file << std::to_string(result.dim()) << " ";
  *file << std::to_string(result.numel()) << " ";
  for (int64_t i = 0; i < result.dim(); ++i) {
    *file << std::to_string(result.sizes()[i]) << " ";
  }
}

// Helper to create a 1D Long tensor from initializer list
static at::Tensor make_long_tensor(const std::vector<int64_t>& values) {
  at::Tensor t = at::zeros({static_cast<int64_t>(values.size())}, at::kLong);
  int64_t* data = t.data_ptr<int64_t>();
  for (size_t i = 0; i < values.size(); ++i) {
    data[i] = values[i];
  }
  return t;
}

// Helper to create a 1D Float tensor from initializer list
static at::Tensor make_float_tensor(const std::vector<float>& values) {
  at::Tensor t = at::zeros({static_cast<int64_t>(values.size())}, at::kFloat);
  float* data = t.data_ptr<float>();
  for (size_t i = 0; i < values.size(); ++i) {
    data[i] = values[i];
  }
  return t;
}

// Helper to create a scalar Long tensor
static at::Tensor make_scalar_long_tensor(int64_t value) {
  at::Tensor t = at::zeros({}, at::kLong);
  t.data_ptr<int64_t>()[0] = value;
  return t;
}

class RepeatInterleaveTest : public ::testing::Test {
 protected:
  void SetUp() override {
    tensor = at::ones({2, 3, 4}, at::kFloat);
    float* data = tensor.data_ptr<float>();
    for (int64_t i = 0; i < tensor.numel(); ++i) {
      data[i] = static_cast<float>(i);
    }
  }

  at::Tensor tensor;
};

// ========== Scalar repeats tests ==========

// Scalar repeats with explicit dim
TEST_F(RepeatInterleaveTest, ScalarRepeatsWithDim) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();
  file << "ScalarRepeatsWithDim ";
  at::Tensor result = tensor.repeat_interleave(2, 1);
  write_repeat_interleave_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// Scalar repeats without dim (flatten first, then repeat)
TEST_F(RepeatInterleaveTest, ScalarRepeatsWithoutDim) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "ScalarRepeatsWithoutDim ";
  at::Tensor result = tensor.repeat_interleave(2);
  write_repeat_interleave_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// Scalar repeats with dim=0
TEST_F(RepeatInterleaveTest, ScalarRepeatsDim0) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "ScalarRepeatsDim0 ";
  at::Tensor result = tensor.repeat_interleave(3, 0);
  write_repeat_interleave_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// Scalar repeats with negative dim
TEST_F(RepeatInterleaveTest, ScalarRepeatsNegativeDim) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "ScalarRepeatsNegativeDim ";
  at::Tensor result = tensor.repeat_interleave(2, -1);
  write_repeat_interleave_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// ========== Tensor repeats tests ==========

// Tensor repeats with explicit dim
TEST_F(RepeatInterleaveTest, TensorRepeatsWithDim) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "TensorRepeatsWithDim ";
  at::Tensor repeats = make_long_tensor({2, 1, 3});
  at::Tensor result = tensor.repeat_interleave(repeats, 1);
  write_repeat_interleave_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// Tensor repeats without dim
TEST_F(RepeatInterleaveTest, TensorRepeatsWithoutDim) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "TensorRepeatsWithoutDim ";
  // Input {2, 3} flattens to {6}, repeats must match flattened size
  at::Tensor repeats = make_long_tensor({2, 1, 3, 1, 2, 1});
  at::Tensor small_tensor = at::ones({2, 3}, at::kFloat);
  at::Tensor result = small_tensor.repeat_interleave(repeats);
  write_repeat_interleave_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// Scalar tensor (0-dim) as repeats
TEST_F(RepeatInterleaveTest, ScalarTensorRepeats) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "ScalarTensorRepeats ";
  at::Tensor repeats = make_scalar_long_tensor(2);
  at::Tensor result = tensor.repeat_interleave(repeats, 1);
  write_repeat_interleave_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// ========== Standalone function test ==========

// at::repeat_interleave(repeats) standalone function
TEST_F(RepeatInterleaveTest, StandaloneRepeatInterleave) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "StandaloneRepeatInterleave ";
  at::Tensor repeats = make_long_tensor({2, 1, 3});
  at::Tensor result = at::repeat_interleave(repeats);
  write_repeat_interleave_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// ========== Shape coverage ==========

// 1D tensor
TEST_F(RepeatInterleaveTest, OneDTensor) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "OneDTensor ";
  at::Tensor t1d = at::arange(5, at::kFloat);
  at::Tensor result = t1d.repeat_interleave(2);
  write_repeat_interleave_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// Large shape
TEST_F(RepeatInterleaveTest, LargeShape) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "LargeShape ";
  at::Tensor large = at::ones({100, 50}, at::kFloat);
  at::Tensor result = large.repeat_interleave(3, 0);
  write_repeat_interleave_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// Empty tensor
TEST_F(RepeatInterleaveTest, EmptyTensor) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "EmptyTensor ";
  at::Tensor empty = at::ones({0, 3}, at::kFloat);
  at::Tensor result = empty.repeat_interleave(2, 0);
  write_repeat_interleave_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// All-one dimensions
TEST_F(RepeatInterleaveTest, AllOneShape) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "AllOneShape ";
  at::Tensor t = at::ones({1, 1, 1}, at::kFloat);
  at::Tensor result = t.repeat_interleave(2, 0);
  write_repeat_interleave_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// ========== Dtype coverage ==========

// float64
TEST_F(RepeatInterleaveTest, Float64Dtype) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "Float64Dtype ";
  at::Tensor t = at::ones({2, 3}, at::kDouble);
  at::Tensor result = t.repeat_interleave(2, 0);
  file << std::to_string(static_cast<int>(result.scalar_type())) << " ";
  write_repeat_interleave_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// int32
TEST_F(RepeatInterleaveTest, Int32Dtype) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "Int32Dtype ";
  at::Tensor t = at::ones({2, 3}, at::kInt);
  at::Tensor result = t.repeat_interleave(2, 0);
  file << std::to_string(static_cast<int>(result.scalar_type())) << " ";
  write_repeat_interleave_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// int64
TEST_F(RepeatInterleaveTest, Int64Dtype) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "Int64Dtype ";
  at::Tensor t = at::ones({2, 3}, at::kLong);
  at::Tensor result = t.repeat_interleave(2, 0);
  file << std::to_string(static_cast<int>(result.scalar_type())) << " ";
  write_repeat_interleave_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// ========== Boundary cases ==========

// Zero repeats
TEST_F(RepeatInterleaveTest, ZeroRepeats) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "ZeroRepeats ";
  at::Tensor t = at::ones({2, 3}, at::kFloat);
  at::Tensor result = t.repeat_interleave(0, 0);
  write_repeat_interleave_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// Zero repeats without dim
TEST_F(RepeatInterleaveTest, ZeroRepeatsNoDim) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "ZeroRepeatsNoDim ";
  at::Tensor t = at::ones({2, 3}, at::kFloat);
  at::Tensor result = t.repeat_interleave(0);
  write_repeat_interleave_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// Empty standalone repeats
TEST_F(RepeatInterleaveTest, EmptyStandaloneRepeats) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "EmptyStandaloneRepeats ";
  at::Tensor repeats = make_long_tensor({});
  at::Tensor result = at::repeat_interleave(repeats);
  write_repeat_interleave_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

TEST_F(RepeatInterleaveTest, EmptyStandaloneRepeatsInvalidOutputSize) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "EmptyStandaloneRepeatsInvalidOutputSize ";
  try {
    at::Tensor repeats = make_long_tensor({});
    at::Tensor result = at::repeat_interleave(repeats, 1);
    write_repeat_interleave_result_to_file(&file, result);
  } catch (const std::exception&) {
    file << "exception ";
  }
  file << "\n";
  file.saveFile();
}

// ========== Data integrity ==========

// Verify data after scalar repeat_interleave
TEST_F(RepeatInterleaveTest, ScalarDataIntegrity) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "ScalarDataIntegrity ";
  at::Tensor t = make_float_tensor({1.0f, 2.0f, 3.0f});
  at::Tensor result = t.repeat_interleave(2);
  float* data = result.data_ptr<float>();
  bool correct = (data[0] == 1.0f && data[1] == 1.0f && data[2] == 2.0f &&
                  data[3] == 2.0f && data[4] == 3.0f && data[5] == 3.0f);
  file << std::to_string(correct) << " ";
  file << "\n";
  file.saveFile();
}

// Verify data after tensor repeat_interleave
TEST_F(RepeatInterleaveTest, TensorDataIntegrity) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "TensorDataIntegrity ";
  at::Tensor t = make_float_tensor({1.0f, 2.0f, 3.0f});
  at::Tensor repeats = make_long_tensor({2, 1, 3});
  at::Tensor result = t.repeat_interleave(repeats);
  float* data = result.data_ptr<float>();
  bool correct = (data[0] == 1.0f && data[1] == 1.0f && data[2] == 2.0f &&
                  data[3] == 3.0f && data[4] == 3.0f && data[5] == 3.0f);
  file << std::to_string(correct) << " ";
  file << "\n";
  file.saveFile();
}

// ========== Exception cases ==========

// Negative scalar repeats
TEST_F(RepeatInterleaveTest, NegativeRepeats) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "NegativeRepeats ";
  try {
    at::Tensor result = tensor.repeat_interleave(-1, 0);
    write_repeat_interleave_result_to_file(&file, result);
  } catch (const std::exception&) {
    file << "exception ";
  }
  file << "\n";
  file.saveFile();
}

// Invalid 2D repeats tensor
TEST_F(RepeatInterleaveTest, InvalidRepeatsDim) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "InvalidRepeatsDim ";
  try {
    at::Tensor repeats = at::ones({2, 3}, at::kLong);
    at::Tensor result = tensor.repeat_interleave(repeats, 0);
    write_repeat_interleave_result_to_file(&file, result);
  } catch (const std::exception&) {
    file << "exception ";
  }
  file << "\n";
  file.saveFile();
}

// Repeats size mismatch
TEST_F(RepeatInterleaveTest, RepeatsSizeMismatch) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "RepeatsSizeMismatch ";
  try {
    at::Tensor repeats = make_long_tensor({2, 3});  // size 2, but dim 1 has 3
    at::Tensor result = tensor.repeat_interleave(repeats, 1);
    write_repeat_interleave_result_to_file(&file, result);
  } catch (const std::exception&) {
    file << "exception ";
  }
  file << "\n";
  file.saveFile();
}

TEST_F(RepeatInterleaveTest, TensorRepeatsNegativeOutputSize) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "TensorRepeatsNegativeOutputSize ";
  try {
    at::Tensor repeats = make_long_tensor({1, 2, 1});
    at::Tensor result = tensor.repeat_interleave(repeats, 1, -1);
    write_repeat_interleave_result_to_file(&file, result);
  } catch (const std::exception&) {
    file << "exception ";
  }
  file << "\n";
  file.saveFile();
}

TEST_F(RepeatInterleaveTest, ScalarRepeatsNegativeOutputSize) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "ScalarRepeatsNegativeOutputSize ";
  try {
    at::Tensor result = tensor.repeat_interleave(2, 1, -1);
    write_repeat_interleave_result_to_file(&file, result);
  } catch (const std::exception&) {
    file << "exception ";
  }
  file << "\n";
  file.saveFile();
}

TEST_F(RepeatInterleaveTest, ScalarZeroRepeatsInvalidOutputSize) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "ScalarZeroRepeatsInvalidOutputSize ";
  try {
    at::Tensor result = tensor.repeat_interleave(0, 1, 1);
    write_repeat_interleave_result_to_file(&file, result);
  } catch (const std::exception&) {
    file << "exception ";
  }
  file << "\n";
  file.saveFile();
}

TEST_F(RepeatInterleaveTest, StandaloneRepeatsZeroOutputSizeMismatch) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "StandaloneRepeatsZeroOutputSizeMismatch ";
  try {
    at::Tensor repeats = make_long_tensor({1, 2, 1});
    at::Tensor result = at::repeat_interleave(repeats, 0);
    write_repeat_interleave_result_to_file(&file, result);
  } catch (const std::exception&) {
    file << "exception ";
  }
  file << "\n";
  file.saveFile();
}

TEST_F(RepeatInterleaveTest, TensorRepeatsZeroOutputSizeMismatch) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "TensorRepeatsZeroOutputSizeMismatch ";
  try {
    at::Tensor repeats = make_long_tensor({1, 2, 1});
    at::Tensor result = tensor.repeat_interleave(repeats, 1, 0);
    write_repeat_interleave_result_to_file(&file, result);
  } catch (const std::exception&) {
    file << "exception ";
  }
  file << "\n";
  file.saveFile();
}

TEST_F(RepeatInterleaveTest, TensorRepeatsZeroOutputSizeValid) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "TensorRepeatsZeroOutputSizeValid ";
  try {
    at::Tensor t = at::ones({3}, at::kFloat);
    at::Tensor repeats = make_long_tensor({0, 0, 0});
    at::Tensor result = t.repeat_interleave(repeats, 0, 0);
    write_repeat_interleave_result_to_file(&file, result);
  } catch (const std::exception&) {
    file << "exception ";
  }
  file << "\n";
  file.saveFile();
}

}  // namespace test
}  // namespace at
