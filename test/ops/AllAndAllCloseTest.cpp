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

TEST(TensorBodyTest, AllTest) {
  auto file_name = g_custom_param.get();
  paddle_api_test::FileManerger file(file_name);
  file.createFile();
  file << "AllTest ";

  at::Tensor t1 = tensor_from_vector_1d<int>({1, 1, 1}, at::kInt).to(at::kBool);
  at::Tensor t2 = tensor_from_vector_1d<int>({1, 0, 1}, at::kInt).to(at::kBool);

  file << std::to_string(t1.all().item<bool>() ? 1 : 0) << " ";
  file << std::to_string(t2.all().item<bool>() ? 1 : 0) << " ";

  at::Tensor t3 = tensor_from_vector_1d<int>({1, 0, 1, 1}, at::kInt)
                      .to(at::kBool)
                      .reshape({2, 2});
  at::Tensor t3_all_dim0 = t3.all(0);
  file << std::to_string(t3_all_dim0.size(0)) << " ";
  file << std::to_string(t3_all_dim0[0].item<bool>() ? 1 : 0) << " ";

  file << "\n";
  file.saveFile();
}

TEST(TensorBodyTest, AllCloseTest) {
  auto file_name = g_custom_param.get();
  paddle_api_test::FileManerger file(file_name);
  file.openAppend();
  file << "AllCloseTest ";

  at::Tensor t1 = tensor_from_vector_1d<float>({1.0f, 2.0f, 3.0f}, at::kFloat);
  at::Tensor t2 =
      tensor_from_vector_1d<float>({1.00001f, 2.0f, 3.0f}, at::kFloat);

  file << std::to_string(t1.allclose(t1) ? 1 : 0) << " ";
  file << std::to_string(t1.allclose(t2) ? 1 : 0) << " ";
  file << std::to_string(t1.allclose(t2, 1e-4) ? 1 : 0) << " ";

  file << "\n";
  file.saveFile();
}

}  // namespace test
}  // namespace at
