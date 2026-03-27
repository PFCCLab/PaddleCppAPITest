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

TEST(TensorBodyTest, CoalesceTest) {
  auto file_name = g_custom_param.get();
  paddle_api_test::FileManerger file(file_name);
  file.createFile();
  file << "CoalesceTest ";

  // Create an uncoalesced sparse tensor (duplicate indices)
  at::Tensor indices =
      tensor_from_vector_1d<int64_t>({0, 0, 1, 0, 0, 1}, at::kLong)
          .reshape({2, 3});
  at::Tensor values =
      tensor_from_vector_1d<float>({1.0f, 2.0f, 3.0f}, at::kFloat);

  at::Tensor sparse = at::sparse_coo_tensor(indices, values, {2, 2});

  file << std::to_string(sparse.is_coalesced() ? 1 : 0) << " ";

  at::Tensor coalesced = sparse.coalesce();
  file << std::to_string(coalesced.is_coalesced() ? 1 : 0) << " ";

  auto coal_values = coalesced._values();
  file << std::to_string(coal_values.size(0)) << " ";

  file << "\n";
  file.saveFile();
}

}  // namespace test
}  // namespace at
