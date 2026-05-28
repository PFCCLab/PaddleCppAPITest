#include <ATen/ATen.h>
#include <gtest/gtest.h>

#include <vector>

#include "src/file_manager.h"

extern paddle_api_test::ThreadSafeParam g_custom_param;

namespace at {
namespace test {

using paddle_api_test::FileManerger;

template <typename T>
static at::Tensor tensor_from_vector_1d(const std::vector<T>& values,
                                        at::ScalarType dtype) {
  auto options = at::TensorOptions().dtype(dtype).device(at::kCPU);
  return at::from_blob(const_cast<T*>(values.data()),
                       {static_cast<int64_t>(values.size())},
                       options)
      .clone();
}

TEST(TensorBodyTest, IsVariableTest) {
  auto file_name = g_custom_param.get();
  paddle_api_test::FileManerger file(file_name);
  file.createFile();
  file << "IsVariableTest ";

  at::Tensor t = at::ones({2, 3}, at::TensorOptions().dtype(at::kFloat));
  file << std::to_string(t.is_variable() ? 1 : 0) << " ";

  file << "\n";
  file.saveFile();
}

TEST(TensorBodyTest, MaskedSelectTest) {
  auto file_name = g_custom_param.get();
  paddle_api_test::FileManerger file(file_name);
  file.openAppend();
  file << "MaskedSelectTest ";

  at::Tensor t =
      tensor_from_vector_1d<float>({1.0f, 2.0f, 3.0f, 4.0f}, at::kFloat)
          .reshape({2, 2});
  at::Tensor mask = tensor_from_vector_1d<int>({1, 0, 0, 1}, at::kInt)
                        .to(at::kBool)
                        .reshape({2, 2});

  at::Tensor result = t.masked_select(mask);

  file << std::to_string(result.size(0)) << " ";
  file << std::to_string(result.sum().item<float>()) << " ";

  file << "\n";
  file.saveFile();
}

// 返回当前用例的结果文件名（用于逐个用例对比）
std::string GetTestCaseResultFileName() {
  std::string base = g_custom_param.get();
  std::string test_name =
      ::testing::UnitTest::GetInstance()->current_test_info()->name();
  if (base.size() >= 4 && base.substr(base.size() - 4) == ".txt") {
    base.resize(base.size() - 4);
  }
  return base + "_" + test_name + ".txt";
}

TEST(TensorBodyTest, Rename) {
  auto file_name = g_custom_param.get();
  paddle_api_test::FileManerger file(file_name);
  file.openAppend();
  file << "Rename ";
  at::Tensor tensor = at::ones({2, 3, 4}, at::kFloat);
  at::Tensor renamed = tensor.rename(std::nullopt);
  file << std::to_string(renamed.sizes().size()) << " ";
  file << "\n";
  file.saveFile();
}

TEST(TensorBodyTest, DtypeMethod) {
  FileManerger file(GetTestCaseResultFileName());
  file.openAppend();
  file << "DtypeMethod ";
  at::Tensor tensor = at::ones({2, 3, 4}, at::kFloat);
  c10::ScalarType dt = tensor.scalar_type();
  file << std::to_string(static_cast<int>(dt)) << " ";
  file << "\n";
  file.saveFile();
}

TEST(TensorBodyTest, CopyMethod) {
  FileManerger file(GetTestCaseResultFileName());
  file.openAppend();
  file << "CopyMethod ";
  at::Tensor tensor = at::ones({2, 3, 4}, at::kFloat);
  at::Tensor src = at::ones({2, 3, 4}, at::kFloat).fill_(2.0f);
  tensor.copy_(src);
  file << std::to_string(tensor.dim()) << " ";
  file << "\n";
  file.saveFile();
}

TEST(TensorBodyTest, FloorDivide) {
  FileManerger file(GetTestCaseResultFileName());
  file.openAppend();
  file << "FloorDivide ";
  at::Tensor input = at::ones({2, 3}, at::kFloat).fill_(7.0f);
  at::Scalar divisor = 3.0f;
  input.floor_divide_(divisor);
  float* data = input.data_ptr<float>();
  file << std::to_string(static_cast<int>(data[0])) << " ";
  file << "\n";
  file.saveFile();
}

TEST(TensorBodyTest, NbytesMethod) {
  FileManerger file(GetTestCaseResultFileName());
  file.openAppend();
  file << "NbytesMethod ";
  at::Tensor tensor = at::ones({2, 3, 4}, at::kFloat);
  size_t nbytes = tensor.nbytes();
  file << std::to_string(nbytes) << " ";
  file << "\n";
  file.saveFile();
}

TEST(TensorBodyTest, ItemsizeMethod) {
  FileManerger file(GetTestCaseResultFileName());
  file.openAppend();
  file << "ItemsizeMethod ";
  at::Tensor tensor = at::ones({2, 3, 4}, at::kFloat);
  size_t itemsize = tensor.itemsize();
  file << std::to_string(itemsize) << " ";
  file << "\n";
  file.saveFile();
}

TEST(TensorBodyTest, ElementSizeMethod) {
  FileManerger file(GetTestCaseResultFileName());
  file.openAppend();
  file << "ElementSizeMethod ";
  at::Tensor tensor = at::ones({2, 3, 4}, at::kFloat);
  int64_t elem_size = tensor.element_size();
  file << std::to_string(elem_size) << " ";
  file << "\n";
  file.saveFile();
}

TEST(TensorBodyTest, CloneMethod) {
  FileManerger file(GetTestCaseResultFileName());
  file.openAppend();
  file << "CloneMethod ";
  at::Tensor tensor = at::ones({2, 3, 4}, at::kFloat);
  at::Tensor cloned = tensor.clone();
  file << std::to_string(cloned.dim()) << " ";
  file << std::to_string(cloned.numel()) << " ";
  file << "\n";
  file.saveFile();
}

}  // namespace test
}  // namespace at
