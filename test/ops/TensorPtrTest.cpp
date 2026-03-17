#include <ATen/ATen.h>
#include <gtest/gtest.h>

#include "../../src/file_manager.h"

extern paddle_api_test::ThreadSafeParam g_custom_param;

TEST(TensorBodyTest, PtrTest) {
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCPU);
  at::Tensor t = at::ones({2, 3}, options);

  // [DIFF] // const float* const_ptr = t.const_data_ptr<float>();
  //   EXPECT_NE(const_ptr, nullptr);

  const void* void_const_ptr = t.const_data_ptr();
  EXPECT_NE(void_const_ptr, nullptr);

  // [DIFF] // float* mut_ptr = t.mutable_data_ptr<float>();
  //   EXPECT_NE(mut_ptr, nullptr);

  void* void_mut_ptr = t.mutable_data_ptr();
  EXPECT_NE(void_mut_ptr, nullptr);

  // We should write to file to check values
  auto file_name = g_custom_param.get();
  paddle_api_test::FileManerger file(file_name);
  file.openAppend();
  //   file << "const_ptr[0]: " + std::to_string(const_ptr[0]) + "\n";

  //   mut_ptr[0] = 5.0f;
  //   file << "mut_ptr[0]: " + std::to_string(mut_ptr[0]) + "\n";

  file.saveFile();
  // Type mismatch crash check?
  // const int* int_ptr = t.const_data_ptr<int>(); // Might throw
}
