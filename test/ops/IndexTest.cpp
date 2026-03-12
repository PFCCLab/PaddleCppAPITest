#include <ATen/ATen.h>
#if USE_PADDLE_API
#include <ATen/indexing.h>
#else
#include <ATen/TensorIndexing.h>
#endif
#include <ATen/core/Tensor.h>
#include <ATen/ops/arange.h>
#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "../../src/file_manager.h"

extern paddle_api_test::ThreadSafeParam g_custom_param;

namespace at {
namespace test {

using paddle_api_test::FileManerger;
using paddle_api_test::ThreadSafeParam;

class IndexTest : public ::testing::Test {
 protected:
  void SetUp() override {}
};

TEST_F(IndexTest, SliceIndexKeepsStride) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();

  at::Tensor base = at::arange(0, 24, at::TensorOptions().dtype(at::kFloat))
                        .reshape({2, 3, 4});

  using at::indexing::Slice;
  at::Tensor result = base.index({Slice(), Slice(1, 3), Slice()});

  file << std::to_string(result.dim()) << " ";
  file << std::to_string(result.sizes()[0]) << " ";
  file << std::to_string(result.sizes()[1]) << " ";
  file << std::to_string(result.sizes()[2]) << " ";
  file << std::to_string(result.numel()) << " ";
  file << std::to_string(static_cast<int>(result.is_contiguous())) << " ";

  auto strides = result.strides();
  file << std::to_string(strides[0]) << " ";
  file << std::to_string(strides[1]) << " ";
  file << std::to_string(strides[2]) << " ";

  file << std::to_string(result.storage_offset()) << " ";

  float* base_data = base.data_ptr<float>();
  float* result_data = result.data_ptr<float>();
  file << std::to_string(static_cast<int64_t>(result_data - base_data)) << " ";

  file << std::to_string(result_data[0]) << " ";
  file << std::to_string(result_data[1]) << " ";
  file.saveFile();
}

}  // namespace test
}  // namespace at
