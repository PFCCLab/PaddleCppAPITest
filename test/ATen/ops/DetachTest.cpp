#include <ATen/ATen.h>
#include <ATen/core/Tensor.h>
#include <ATen/ops/zeros.h>
#include <c10/util/Exception.h>
#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "src/file_manager.h"

extern paddle_api_test::ThreadSafeParam g_custom_param;

namespace at {
namespace test {

using paddle_api_test::FileManerger;

class DetachTest : public ::testing::Test {
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

TEST_F(DetachTest, DetachBasic) {
  at::Tensor t1 = at::zeros({3, 3}, at::kFloat);
  float* data = t1.data_ptr<float>();
  for (int i = 0; i < 9; ++i) {
    data[i] = static_cast<float>(i);
  }

  at::Tensor result = t1.detach();

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();
  file << "DetachBasic ";
  write_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

TEST_F(DetachTest, DetachInplace) {
  at::Tensor t1 = at::zeros({3, 3}, at::kFloat);
  float* data = t1.data_ptr<float>();
  for (int i = 0; i < 9; ++i) {
    data[i] = static_cast<float>(i);
  }

  at::Tensor result = t1.detach_();

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "DetachInplace ";
  write_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// 返回当前用例的结果文件名
std::string GetTestCaseResultFileName() {
  std::string base = g_custom_param.get();
  std::string test_name =
      ::testing::UnitTest::GetInstance()->current_test_info()->name();
  if (base.size() >= 4 && base.substr(base.size() - 4) == ".txt") {
    base.resize(base.size() - 4);
  }
  return base + "_" + test_name + ".txt";
}

TEST_F(DetachTest, TensorData) {
  FileManerger file(GetTestCaseResultFileName());
  file.openAppend();
  file << "TensorData ";
  at::Tensor tensor = at::ones({2, 3, 4}, at::kFloat);
  at::Tensor result = tensor.tensor_data();
  file << std::to_string(result.dim()) << " ";
  file << std::to_string(result.numel()) << " ";
  file << "\n";
  file.saveFile();
}

TEST_F(DetachTest, VariableData) {
  FileManerger file(GetTestCaseResultFileName());
  file.openAppend();
  file << "VariableData ";
  at::Tensor tensor = at::ones({2, 3, 4}, at::kFloat);
  at::Tensor result = tensor.variable_data();
  file << std::to_string(result.dim()) << " ";
  file << std::to_string(result.numel()) << " ";
  file << "\n";
  file.saveFile();
}

}  // namespace test
}  // namespace at
