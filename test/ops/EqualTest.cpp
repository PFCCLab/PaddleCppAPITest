#include <ATen/ATen.h>
#include <ATen/core/Tensor.h>
#include <c10/util/Exception.h>
#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "../../src/file_manager.h"

extern paddle_api_test::ThreadSafeParam g_custom_param;

namespace at {
namespace test {

using paddle_api_test::FileManerger;

class EqualTest : public ::testing::Test {
 protected:
  void SetUp() override {}
};

static void write_bool_result_to_file(FileManerger* file, bool result) {
  *file << std::to_string(result) << " ";
}

TEST_F(EqualTest, BasicEqual) {
  at::Tensor t1 = at::zeros({4}, at::kFloat);
  at::Tensor t2 = at::zeros({4}, at::kFloat);

  bool result = t1.equal(t2);

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();
  file << "BasicEqual ";
  write_bool_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

TEST_F(EqualTest, NotEqualContent) {
  at::Tensor t1 = at::zeros({4}, at::kFloat);
  at::Tensor t2 = at::zeros({4}, at::kFloat);
  t2.data_ptr<float>()[0] = 1.0f;

  bool result = t1.equal(t2);

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "NotEqualContent ";
  write_bool_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

TEST_F(EqualTest, NotEqualShape) {
  at::Tensor t1 = at::zeros({4}, at::kFloat);
  at::Tensor t2 = at::zeros({2, 2}, at::kFloat);

  bool result = t1.equal(t2);

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "NotEqualShape ";
  write_bool_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// [DIFF] Test paddle equal exception when comparing tensors of different types
// Torch returns false without checking specific data types, whereas Paddle
// throws: "The type of data we are trying to retrieve (int32) does not match
// the type of data (float32)..."
TEST_F(EqualTest, NotEqualDtype) {
  at::Tensor t1 = at::zeros({4}, at::kFloat);
  at::Tensor t2 = at::zeros({4}, at::kInt);

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "NotEqualDtype ";
  try {
    bool result = t1.equal(t2);
    write_bool_result_to_file(&file, result);
  } catch (const std::exception& e) {
    file << "exception ";
  }
  file << "\n";
  file.saveFile();
}

TEST_F(EqualTest, EqualScalar) {
  // {} empty shape
  at::Tensor t1 = at::zeros({}, at::kFloat);
  at::Tensor t2 = at::zeros({}, at::kFloat);

  bool result = t1.equal(t2);

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "EqualScalar ";
  write_bool_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

TEST_F(EqualTest, ExceptionTest) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "ExceptionTest ";
  try {
    at::Tensor t1;  // undefined tensor
    at::Tensor t2 = at::zeros({4}, at::kFloat);
    bool result = t1.equal(t2);
    write_bool_result_to_file(&file, result);
  } catch (const std::exception& e) {
    file << "exception ";
  }
  file << "\n";
  file.saveFile();
}

}  // namespace test
}  // namespace at
