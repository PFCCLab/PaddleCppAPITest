#include <ATen/ATen.h>
#include <gtest/gtest.h>

#include "../../src/file_manager.h"

extern paddle_api_test::ThreadSafeParam g_custom_param;

TEST(TensorBodyTest, PtrTest) {
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCPU);
  at::Tensor t = at::ones({2, 3}, options);

#if USE_PADDLE_API
  const void* void_const_ptr = t.const_data_ptr();
  EXPECT_NE(void_const_ptr, nullptr);

  void* void_mut_ptr = t.mutable_data_ptr();
  EXPECT_NE(void_mut_ptr, nullptr);
#else
  const float* const_ptr = t.const_data_ptr<float>();
  EXPECT_NE(const_ptr, nullptr);

  const void* void_const_ptr = t.const_data_ptr();
  EXPECT_NE(void_const_ptr, nullptr);

  float* mut_ptr = t.mutable_data_ptr<float>();
  EXPECT_NE(mut_ptr, nullptr);

  void* void_mut_ptr = t.mutable_data_ptr();
  EXPECT_NE(void_mut_ptr, nullptr);
#endif

  // We should write to file to check values
  auto file_name = g_custom_param.get();
  paddle_api_test::FileManerger file(file_name);
  file.openAppend();
#if USE_PADDLE_API
  file << "typed_ptr_unavailable_on_paddle_compat\n";
  file << "void_const_ptr_nonnull: "
       << std::to_string(void_const_ptr != nullptr ? 1 : 0) << "\n";
  file << "void_mut_ptr_nonnull: "
       << std::to_string(void_mut_ptr != nullptr ? 1 : 0) << "\n";
#else
  file << "const_ptr[0]: " + std::to_string(const_ptr[0]) + "\n";

  mut_ptr[0] = 5.0f;
  file << "mut_ptr[0]: " + std::to_string(mut_ptr[0]) + "\n";
#endif

  file.saveFile();
  // Type mismatch crash check?
  // const int* int_ptr = t.const_data_ptr<int>(); // Might throw
}
