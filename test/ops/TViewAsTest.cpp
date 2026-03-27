#include <ATen/ATen.h>
#include <gtest/gtest.h>

#include "../../src/file_manager.h"

extern paddle_api_test::ThreadSafeParam g_custom_param;

namespace at {
namespace test {

TEST(TensorBodyTest, TTest) {
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCPU);
  at::Tensor t1 = at::ones({2, 3}, options);

  at::Tensor t2 = t1.t();

  auto file_name = g_custom_param.get();
  paddle_api_test::FileManerger file(file_name);
  file.createFile();
  file << "TTest ";
  file << std::to_string(t2.size(0)) << " ";
  file << std::to_string(t2.size(1)) << " ";

  at::Tensor t3 = at::zeros({3, 2}, options);
  t3.t_();
  file << std::to_string(t3.size(0)) << " ";
  file << std::to_string(t3.size(1)) << " ";
  file << "\n";
  file.saveFile();
}

TEST(TensorBodyTest, ViewAsTest) {
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCPU);
  at::Tensor t1 = at::ones({2, 3}, options);
  at::Tensor t2 = at::zeros({6}, options);

  at::Tensor t3 = t2.view_as(t1);

  auto file_name = g_custom_param.get();
  paddle_api_test::FileManerger file(file_name);
  file.openAppend();
  file << "ViewAsTest ";
  file << std::to_string(t3.size(0)) << " ";
  file << std::to_string(t3.size(1)) << " ";
  file << "\n";
  file.saveFile();
}

}  // namespace test
}  // namespace at
