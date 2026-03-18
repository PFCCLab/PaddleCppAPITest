#include <ATen/ATen.h>
#include <ATen/core/Tensor.h>
#include <ATen/ops/zeros.h>
#include <c10/util/Exception.h>
#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "../../src/file_manager.h"

extern paddle_api_test::ThreadSafeParam g_custom_param;

namespace at {
namespace test {

using paddle_api_test::FileManerger;

class SelectTest : public ::testing::Test {
 protected:
  void SetUp() override {}
};

static void write_result_to_file(FileManerger* file, const at::Tensor& result) {
  *file << std::to_string(result.dim()) << " ";
  *file << std::to_string(result.numel()) << " ";
  if (result.scalar_type() == at::kFloat) {
    float* data = result.data_ptr<float>();
    for (int64_t i = 0; i < result.numel(); ++i) {
      *file << std::to_string(data[i]) << " ";
    }
  } else if (result.scalar_type() == at::kDouble) {
    double* data = result.data_ptr<double>();
    for (int64_t i = 0; i < result.numel(); ++i) {
      *file << std::to_string(data[i]) << " ";
    }
  } else if (result.scalar_type() == at::kInt) {
    int* data = result.data_ptr<int>();
    for (int64_t i = 0; i < result.numel(); ++i) {
      *file << std::to_string(data[i]) << " ";
    }
  } else if (result.scalar_type() == at::kLong) {
    int64_t* data = result.data_ptr<int64_t>();
    for (int64_t i = 0; i < result.numel(); ++i) {
      *file << std::to_string(data[i]) << " ";
    }
  }
}

TEST_F(SelectTest, SelectBasic) {
  at::Tensor t1 = at::zeros({3, 3}, at::kFloat);
  float* data = t1.data_ptr<float>();
  for (int i = 0; i < 9; ++i) {
    data[i] = static_cast<float>(i);
  }

  at::Tensor result = t1.select(0, 1);

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();
  write_result_to_file(&file, result);
  file.saveFile();
}

TEST_F(SelectTest, SelectSymint) {
  at::Tensor t1 = at::zeros({3, 3}, at::kFloat);
  float* data = t1.data_ptr<float>();
  for (int i = 0; i < 9; ++i) {
    data[i] = static_cast<float>(i);
  }

  int64_t dim = 1;
  c10::SymInt index = c10::SymInt(2);
  at::Tensor result = t1.select_symint(dim, index);

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  write_result_to_file(&file, result);
  file.saveFile();
}

// [DIFF] Paddle select with negative dim causes double free or corruption
// SIGABRT
TEST_F(SelectTest, SelectNegativeDim) {
  at::Tensor t1 = at::zeros({3, 3}, at::kFloat);
  float* data = t1.data_ptr<float>();
  for (int i = 0; i < 9; ++i) {
    data[i] = static_cast<float>(i);
  }

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
#if USE_PADDLE_API
  file << "known_crash_on_negative_dim ";
#else
  at::Tensor result = t1.select(-1, 0);
  write_result_to_file(&file, result);
#endif
  file.saveFile();
}

TEST_F(SelectTest, SelectException) {
  at::Tensor t1 = at::zeros({3, 3}, at::kFloat);

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  try {
    at::Tensor result = t1.select(0, 5);  // out of bounds
    write_result_to_file(&file, result);
  } catch (const std::exception& e) {
    file << "exception: " << e.what();
  }
  file.saveFile();
}

}  // namespace test
}  // namespace at
