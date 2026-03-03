#include <ATen/ATen.h>
#include <ATen/core/Tensor.h>
#include <ATen/core/TensorAccessor.h>
#include <ATen/ops/zeros.h>
#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "../src/file_manager.h"

extern paddle_api_test::ThreadSafeParam g_custom_param;

namespace at {
namespace test {

using paddle_api_test::FileManerger;
using paddle_api_test::ThreadSafeParam;

class TensorAccessorTest : public ::testing::Test {
 protected:
  void SetUp() override {}
};

// ===================== TensorAccessor =====================

// 1D TensorAccessor
TEST_F(TensorAccessorTest, Accessor1DFloat) {
  at::Tensor t = at::zeros({5}, at::kFloat);
  float* data = t.data_ptr<float>();
  for (int64_t i = 0; i < 5; ++i) {
    data[i] = static_cast<float>(i) * 1.5f;
  }

  auto accessor = t.accessor<float, 1>();

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();
  file << std::to_string(accessor.size(0)) << " ";
  file << std::to_string(accessor.stride(0)) << " ";
  for (int64_t i = 0; i < accessor.size(0); ++i) {
    file << std::to_string(accessor[i]) << " ";
  }
  file.saveFile();
}

// 2D TensorAccessor
TEST_F(TensorAccessorTest, Accessor2DFloat) {
  at::Tensor t = at::zeros({3, 4}, at::kFloat);
  float* data = t.data_ptr<float>();
  for (int64_t i = 0; i < 12; ++i) {
    data[i] = static_cast<float>(i);
  }

  auto accessor = t.accessor<float, 2>();

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << std::to_string(accessor.size(0)) << " ";
  file << std::to_string(accessor.size(1)) << " ";
  file << std::to_string(accessor.stride(0)) << " ";
  file << std::to_string(accessor.stride(1)) << " ";
  // 通过 accessor[i][j] 访问
  for (int64_t i = 0; i < accessor.size(0); ++i) {
    for (int64_t j = 0; j < accessor.size(1); ++j) {
      file << std::to_string(accessor[i][j]) << " ";
    }
  }
  file.saveFile();
}

// 3D TensorAccessor
TEST_F(TensorAccessorTest, Accessor3DFloat) {
  at::Tensor t = at::zeros({2, 3, 4}, at::kFloat);
  float* data = t.data_ptr<float>();
  for (int64_t i = 0; i < 24; ++i) {
    data[i] = static_cast<float>(i) * 0.1f;
  }

  auto accessor = t.accessor<float, 3>();

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << std::to_string(accessor.size(0)) << " ";
  file << std::to_string(accessor.size(1)) << " ";
  file << std::to_string(accessor.size(2)) << " ";
  // 选取首尾值
  file << std::to_string(accessor[0][0][0]) << " ";
  file << std::to_string(accessor[1][2][3]) << " ";
  file.saveFile();
}

// Double 类型 accessor
TEST_F(TensorAccessorTest, Accessor1DDouble) {
  at::Tensor t = at::zeros({4}, at::kDouble);
  double* data = t.data_ptr<double>();
  for (int64_t i = 0; i < 4; ++i) {
    data[i] = static_cast<double>(i) * 2.5;
  }

  auto accessor = t.accessor<double, 1>();

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  for (int64_t i = 0; i < accessor.size(0); ++i) {
    file << std::to_string(accessor[i]) << " ";
  }
  file.saveFile();
}

// Int 类型 accessor
TEST_F(TensorAccessorTest, Accessor2DInt) {
  at::Tensor t = at::zeros({3, 3}, at::kInt);
  int* data = t.data_ptr<int>();
  for (int64_t i = 0; i < 9; ++i) {
    data[i] = static_cast<int>(i) - 4;
  }

  auto accessor = t.accessor<int, 2>();

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  for (int64_t i = 0; i < accessor.size(0); ++i) {
    for (int64_t j = 0; j < accessor.size(1); ++j) {
      file << std::to_string(accessor[i][j]) << " ";
    }
  }
  file.saveFile();
}

// Long 类型 accessor
TEST_F(TensorAccessorTest, Accessor1DLong) {
  at::Tensor t = at::zeros({6}, at::kLong);
  int64_t* data = t.data_ptr<int64_t>();
  for (int64_t i = 0; i < 6; ++i) {
    data[i] = i * 1000;
  }

  auto accessor = t.accessor<int64_t, 1>();

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  for (int64_t i = 0; i < accessor.size(0); ++i) {
    file << std::to_string(accessor[i]) << " ";
  }
  file.saveFile();
}

// sizes() 和 strides() 方法
TEST_F(TensorAccessorTest, AccessorSizesStrides) {
  at::Tensor t = at::zeros({4, 5, 6}, at::kFloat);
  auto accessor = t.accessor<float, 3>();

  auto sizes = accessor.sizes();
  auto strides = accessor.strides();

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  for (int64_t i = 0; i < 3; ++i) {
    file << std::to_string(sizes[i]) << " ";
  }
  for (int64_t i = 0; i < 3; ++i) {
    file << std::to_string(strides[i]) << " ";
  }
  file.saveFile();
}

// data() 方法
TEST_F(TensorAccessorTest, AccessorData) {
  at::Tensor t = at::zeros({3}, at::kFloat);
  float* data = t.data_ptr<float>();
  data[0] = 10.0f;
  data[1] = 20.0f;
  data[2] = 30.0f;

  auto accessor = t.accessor<float, 1>();
  float* acc_data = accessor.data();

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  for (int64_t i = 0; i < 3; ++i) {
    file << std::to_string(acc_data[i]) << " ";
  }
  // 验证 data() 指向同一内存
  file << std::to_string(acc_data == data ? 1 : 0) << " ";
  file.saveFile();
}

// 大 shape
TEST_F(TensorAccessorTest, AccessorLargeShape) {
  at::Tensor t = at::zeros({100, 100}, at::kFloat);
  float* data = t.data_ptr<float>();
  for (int64_t i = 0; i < 10000; ++i) {
    data[i] = static_cast<float>(i);
  }

  auto accessor = t.accessor<float, 2>();

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << std::to_string(accessor[0][0]) << " ";
  file << std::to_string(accessor[49][49]) << " ";
  file << std::to_string(accessor[99][99]) << " ";
  file.saveFile();
}

// ===================== GenericPackedTensorAccessor =====================

// PackedTensorAccessor64
TEST_F(TensorAccessorTest, PackedAccessor64_2D) {
  at::Tensor t = at::zeros({3, 4}, at::kFloat);
  float* data = t.data_ptr<float>();
  for (int64_t i = 0; i < 12; ++i) {
    data[i] = static_cast<float>(i) * 2.0f;
  }

  auto packed = t.packed_accessor64<float, 2>();

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << std::to_string(packed.size(0)) << " ";
  file << std::to_string(packed.size(1)) << " ";
  file << std::to_string(packed.stride(0)) << " ";
  file << std::to_string(packed.stride(1)) << " ";
  file.saveFile();
}

// PackedTensorAccessor32
TEST_F(TensorAccessorTest, PackedAccessor32_2D) {
  at::Tensor t = at::zeros({3, 4}, at::kFloat);
  float* data = t.data_ptr<float>();
  for (int64_t i = 0; i < 12; ++i) {
    data[i] = static_cast<float>(i);
  }

  auto packed = t.packed_accessor32<float, 2>();

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << std::to_string(packed.size(0)) << " ";
  file << std::to_string(packed.size(1)) << " ";
  file << std::to_string(packed.stride(0)) << " ";
  file << std::to_string(packed.stride(1)) << " ";
  file.saveFile();
}

// PackedAccessor transpose
TEST_F(TensorAccessorTest, PackedAccessorTranspose) {
  at::Tensor t = at::zeros({3, 4}, at::kFloat);
  float* data = t.data_ptr<float>();
  for (int64_t i = 0; i < 12; ++i) {
    data[i] = static_cast<float>(i);
  }

  auto packed = t.packed_accessor64<float, 2>();
  auto transposed = packed.transpose(0, 1);

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  // 转置后 size 和 stride 交换
  file << std::to_string(transposed.size(0)) << " ";
  file << std::to_string(transposed.size(1)) << " ";
  file << std::to_string(transposed.stride(0)) << " ";
  file << std::to_string(transposed.stride(1)) << " ";
  file.saveFile();
}

}  // namespace test
}  // namespace at
