#include <ATen/ATen.h>
#include <ATen/ops/full.h>
#include <gtest/gtest.h>

#include <vector>

#include "../../src/file_manager.h"

extern paddle_api_test::ThreadSafeParam g_custom_param;

namespace at {
namespace test {

static at::Tensor tensor_from_vector_i64(const std::vector<int64_t>& values) {
  auto options = at::TensorOptions().dtype(at::kLong).device(at::kCPU);
  return at::from_blob(const_cast<int64_t*>(values.data()),
                       {static_cast<int64_t>(values.size())},
                       options)
      .clone();
}

TEST(TensorBodyTest, IndexPutTest) {
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCPU);
  at::Tensor t1 = at::ones({3, 3}, options);

  at::Tensor indices = tensor_from_vector_i64({0, 2});
  at::Tensor values = at::full({1}, 5.0f, options);

  c10::List<std::optional<at::Tensor>> indices_list;
  indices_list.push_back(indices);

  at::Tensor t2 = t1.index_put(indices_list, values);
  at::Tensor t3 = t1.clone();
  t3.index_put_(indices_list, values);

  auto file_name = g_custom_param.get();
  paddle_api_test::FileManerger file(file_name);
  file.createFile();
  file << "IndexPutTest ";
  file << std::to_string(t2.sum().item<float>()) << " ";
  file << std::to_string(t3.sum().item<float>()) << " ";
  file << "\n";
  file.saveFile();
}

}  // namespace test
}  // namespace at
