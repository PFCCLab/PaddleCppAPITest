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

class ReciprocalTest : public ::testing::Test {
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

TEST_F(ReciprocalTest, MethodReciprocalBasic) {
  at::Tensor t1 = at::zeros({4}, at::kFloat);
  t1.data_ptr<float>()[0] = 1.0f;
  t1.data_ptr<float>()[1] = -2.0f;
  t1.data_ptr<float>()[2] = 4.0f;
  t1.data_ptr<float>()[3] = -0.5f;

  at::Tensor result = t1.reciprocal();

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();
  file << "MethodReciprocalBasic ";
  write_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

TEST_F(ReciprocalTest, MethodReciprocalInplace) {
  at::Tensor t1 = at::zeros({2, 2}, at::kFloat);
  t1.data_ptr<float>()[0] = 1.0f;
  t1.data_ptr<float>()[1] = -2.0f;
  t1.data_ptr<float>()[2] = 4.0f;
  t1.data_ptr<float>()[3] = -0.5f;

  at::Tensor result = t1.reciprocal_();

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "MethodReciprocalInplace ";
  write_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

TEST_F(ReciprocalTest, ScalarReciprocal) {
  at::Tensor t1 = at::zeros({}, at::kFloat);
  t1.data_ptr<float>()[0] = 5.0f;

  at::Tensor result = t1.reciprocal();

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "ScalarReciprocal ";
  write_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

TEST_F(ReciprocalTest, ZeroExtentsReciprocal) {
  at::Tensor t1 = at::zeros({0}, at::kFloat);

  at::Tensor result = t1.reciprocal();

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "ZeroExtentsReciprocal ";
  write_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

TEST_F(ReciprocalTest, ExceptionZeroReciprocal) {
  at::Tensor t1 = at::zeros({2}, at::kFloat);
  t1.data_ptr<float>()[0] = 0.0f;
  t1.data_ptr<float>()[1] = 1.0f;

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "ExceptionZeroReciprocal ";
  try {
    at::Tensor result = t1.reciprocal();
    write_result_to_file(&file, result);
  } catch (const std::exception& e) {
    file << "exception ";
  }
  file << "\n";
  file.saveFile();
}

}  // namespace test
}  // namespace at
