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

class StdTest : public ::testing::Test {
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

TEST_F(StdTest, StdDim) {
  at::Tensor t1 = at::zeros({3, 3}, at::kFloat);
  float* data = t1.data_ptr<float>();
  for (int i = 0; i < 9; ++i) {
    data[i] = static_cast<float>(i);
  }

  at::Tensor result =
      t1.std(0);  // OptionalIntArrayRef but taking int implicitly or std(int
                  // dim) is deprecated but might be there.

  // Wait, let's use the explicit single int dim if available, or just array
  // ref.
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();
  file << "StdDim ";
  write_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

TEST_F(StdTest, StdUnbiased) {
  at::Tensor t1 = at::zeros({3, 3}, at::kFloat);
  float* data = t1.data_ptr<float>();
  for (int i = 0; i < 9; ++i) {
    data[i] = static_cast<float>(i);
  }

  at::Tensor result = t1.std(true);

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "StdUnbiased ";
  write_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

TEST_F(StdTest, StdDimUnbiasedKeepdim) {
  at::Tensor t1 = at::zeros({3, 3}, at::kFloat);
  float* data = t1.data_ptr<float>();
  for (int i = 0; i < 9; ++i) {
    data[i] = static_cast<float>(i);
  }

  at::Tensor result = t1.std(at::IntArrayRef({1}), true, true);

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "StdDimUnbiasedKeepdim ";
  write_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

TEST_F(StdTest, StdDimCorrectionKeepdim) {
  at::Tensor t1 = at::zeros({3, 3}, at::kFloat);
  float* data = t1.data_ptr<float>();
  for (int i = 0; i < 9; ++i) {
    data[i] = static_cast<float>(i);
  }

  int correction = 1;
  at::Tensor result = t1.std(at::IntArrayRef({0}), correction, false);

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "StdDimCorrectionKeepdim ";
  write_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

TEST_F(StdTest, StdException) {
  at::Tensor t1 = at::zeros({3}, at::kFloat);

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "StdException ";
  try {
    at::Tensor result =
        t1.std(at::IntArrayRef({1}), true, true);  // dim out of bounds
    write_result_to_file(&file, result);
  } catch (const std::exception& e) {
    file << "exception: ";  // 报错堆栈不完全一致，先删除堆栈信息，后续再完善
  }
  file << "\n";
  file.saveFile();
}

// 从 TensorTest.cpp 迁移的 Std 测试

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

TEST_F(StdTest, StdDimTensorBody) {
  FileManerger file(GetTestCaseResultFileName());
  file.openAppend();
  file << "StdDim ";
  at::Tensor input = at::ones({2, 3}, at::kFloat).fill_(2.0f);
  input.fill_(1.0f);
  input.data_ptr<float>()[1] = 3.0f;
  at::Tensor result = input.std(1);
  file << std::to_string(result.dim()) << " ";
  file << "\n";
  file.saveFile();
}

TEST_F(StdTest, StdAllTensorBody) {
  FileManerger file(GetTestCaseResultFileName());
  file.openAppend();
  file << "StdAll ";
  at::Tensor input = at::ones({2, 3}, at::kFloat);
  input.fill_(1.0f);
  input.data_ptr<float>()[1] = 3.0f;
  at::Tensor result = input.std(true);
  file << std::to_string(result.dim()) << " ";
  file << "\n";
  file.saveFile();
}

TEST_F(StdTest, StdDimsTensorBody) {
  FileManerger file(GetTestCaseResultFileName());
  file.openAppend();
  file << "StdDims ";
  at::Tensor input = at::ones({2, 3}, at::kFloat);
  input.fill_(1.0f);
  input.data_ptr<float>()[1] = 3.0f;
  at::Tensor result = input.std({1}, true, true);
  file << std::to_string(result.dim()) << " ";
  file << std::to_string(result.sizes()[0]) << " ";
  file << "\n";
  file.saveFile();
}

TEST_F(StdTest, StdCorrectionTensorBody) {
  FileManerger file(GetTestCaseResultFileName());
  file.openAppend();
  file << "StdCorrection ";
  at::Tensor input = at::ones({2, 3}, at::kFloat);
  input.fill_(1.0f);
  input.data_ptr<float>()[1] = 3.0f;
  at::Tensor result = input.std({1}, 1.0, true);
  file << std::to_string(result.dim()) << " ";
  file << "\n";
  file.saveFile();
}

}  // namespace test
}  // namespace at
