#include <ATen/ATen.h>
#include <ATen/core/Tensor.h>
#include <ATen/ops/eye.h>
#include <gtest/gtest.h>

#include <optional>
#include <string>

#include "../../src/file_manager.h"

extern paddle_api_test::ThreadSafeParam g_custom_param;

namespace at {
namespace test {

using paddle_api_test::FileManerger;

class EyeTest : public ::testing::Test {
 protected:
  void SetUp() override {}
};

static void write_tensor_to_file(FileManerger* file, const at::Tensor& t) {
  *file << std::to_string(t.dim()) << " ";
  *file << std::to_string(t.numel()) << " ";
  for (int64_t i = 0; i < t.dim(); ++i) {
    *file << std::to_string(t.size(i)) << " ";
  }

  at::Tensor flat = t.reshape({-1});
  if (t.scalar_type() == at::kFloat) {
    const float* data = flat.data_ptr<float>();
    for (int64_t i = 0; i < flat.numel(); ++i) {
      *file << std::to_string(data[i]) << " ";
    }
  } else if (t.scalar_type() == at::kDouble) {
    const double* data = flat.data_ptr<double>();
    for (int64_t i = 0; i < flat.numel(); ++i) {
      *file << std::to_string(data[i]) << " ";
    }
  } else if (t.scalar_type() == at::kInt) {
    const int* data = flat.data_ptr<int>();
    for (int64_t i = 0; i < flat.numel(); ++i) {
      *file << std::to_string(data[i]) << " ";
    }
  } else if (t.scalar_type() == at::kLong) {
    const int64_t* data = flat.data_ptr<int64_t>();
    for (int64_t i = 0; i < flat.numel(); ++i) {
      *file << std::to_string(data[i]) << " ";
    }
  }
}

TEST_F(EyeTest, EyeNWithOptions) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();
  file << "EyeNWithOptions ";

  at::Tensor t = at::eye(4, at::TensorOptions().dtype(at::kFloat));
  write_tensor_to_file(&file, t);

  at::Tensor zero_n = at::eye(0, at::TensorOptions().dtype(at::kFloat));
  write_tensor_to_file(&file, zero_n);

  file << "\n";
  file.saveFile();
}

TEST_F(EyeTest, EyeNMWithOptions) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "EyeNMWithOptions ";

  at::Tensor t = at::eye(3, 5, at::TensorOptions().dtype(at::kLong));
  write_tensor_to_file(&file, t);

  at::Tensor one_shape = at::eye(1, 1, at::TensorOptions().dtype(at::kLong));
  write_tensor_to_file(&file, one_shape);

  file << "\n";
  file.saveFile();
}

TEST_F(EyeTest, EyeNWithExplicitOptionalArgs) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "EyeNWithExplicitOptionalArgs ";

  at::Tensor t = at::eye(6,
                         std::optional<at::ScalarType>(at::kDouble),
                         std::nullopt,
                         std::nullopt,
                         std::nullopt);
  write_tensor_to_file(&file, t);
  file << "\n";
  file.saveFile();
}

TEST_F(EyeTest, EyeNMWithExplicitOptionalArgs) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "EyeNMWithExplicitOptionalArgs ";

  at::Tensor t = at::eye(2,
                         4,
                         std::optional<at::ScalarType>(at::kInt),
                         std::nullopt,
                         std::nullopt,
                         std::nullopt);
  write_tensor_to_file(&file, t);

  at::Tensor large_t = at::eye(64,
                               64,
                               std::optional<at::ScalarType>(at::kFloat),
                               std::nullopt,
                               std::nullopt,
                               std::nullopt);
  write_tensor_to_file(&file, large_t);

  file << "\n";
  file.saveFile();
}

}  // namespace test
}  // namespace at
