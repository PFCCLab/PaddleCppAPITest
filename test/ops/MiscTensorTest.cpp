#include <ATen/ATen.h>
#include <gtest/gtest.h>

#include <vector>

#include "../../src/file_manager.h"

extern paddle_api_test::ThreadSafeParam g_custom_param;

namespace at {
namespace test {

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

}  // namespace test
}  // namespace at
