#include <ATen/ATen.h>
#include <ATen/core/Tensor.h>
#include <ATen/ops/from_blob.h>
#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "../../src/file_manager.h"

extern paddle_api_test::ThreadSafeParam g_custom_param;

namespace at {
namespace test {

using paddle_api_test::FileManerger;
using paddle_api_test::ThreadSafeParam;

static void write_fromblob_result_to_file(FileManerger* file,
                                          const at::Tensor& result) {
  *file << std::to_string(result.dim()) << " ";
  *file << std::to_string(result.numel()) << " ";
  for (int64_t i = 0; i < result.dim(); ++i) {
    *file << std::to_string(result.sizes()[i]) << " ";
  }
}

class FromBlobTest : public ::testing::Test {
 protected:
  void SetUp() override {
    data_buffer = new float[6];
    for (int i = 0; i < 6; ++i) {
      data_buffer[i] = static_cast<float>(i);
    }
  }

  void TearDown() override { delete[] data_buffer; }

  float* data_buffer;
};

// ========== 基础功能 ==========

TEST_F(FromBlobTest, FromBlobBasic) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();
  file << "FromBlobBasic ";
  std::vector<int64_t> sizes = {2, 3};
  at::Tensor result = at::from_blob(data_buffer, sizes);
  file << std::to_string(result.data_ptr<float>() == data_buffer) << " ";
  write_fromblob_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

TEST_F(FromBlobTest, FromBlobWithOptions) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "FromBlobWithOptions ";
  std::vector<int64_t> sizes = {3, 2};
  at::Tensor result =
      at::from_blob(data_buffer, sizes, at::TensorOptions().dtype(at::kFloat));
  file << std::to_string(static_cast<int>(result.scalar_type())) << " ";
  write_fromblob_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

TEST_F(FromBlobTest, FromBlobWithStrides) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "FromBlobWithStrides ";
  std::vector<int64_t> sizes = {2, 3};
  std::vector<int64_t> strides = {3, 1};  // Row-major
  at::Tensor result = at::from_blob(data_buffer, sizes, strides);
  file << std::to_string(result.strides()[0]) << " ";
  file << std::to_string(result.strides()[1]) << " ";
  file << "\n";
  file.saveFile();
}

TEST_F(FromBlobTest, FromBlob1D) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "FromBlob1D ";
  std::vector<int64_t> sizes = {6};
  at::Tensor result = at::from_blob(data_buffer, sizes);
  file << std::to_string(result.data_ptr<float>() == data_buffer) << " ";
  write_fromblob_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// ========== Shape 覆盖 ==========

// 标量 from_blob
TEST_F(FromBlobTest, FromBlobScalar) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "FromBlobScalar ";
  float* scalar_data = new float[1];
  scalar_data[0] = 42.0f;
  std::vector<int64_t> sizes = {};  // 标量
  at::Tensor result = at::from_blob(scalar_data, sizes);
  write_fromblob_result_to_file(&file, result);
  file << std::to_string(result.data_ptr<float>()[0]) << " ";
  file << "\n";
  file.saveFile();
  delete[] scalar_data;
}

// 大 shape from_blob
TEST_F(FromBlobTest, FromBlobLargeShape) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "FromBlobLargeShape ";
  float* large_data = new float[10000];
  for (int i = 0; i < 10000; ++i) {
    large_data[i] = static_cast<float>(i % 100);
  }
  std::vector<int64_t> sizes = {100, 100};
  at::Tensor result = at::from_blob(large_data, sizes);
  file << std::to_string(result.numel()) << " ";
  write_fromblob_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
  delete[] large_data;
}

// 零维度 from_blob
TEST_F(FromBlobTest, FromBlobZeroDim) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "FromBlobZeroDim ";
  float* zero_data = new float[1];  // 实际不需要元素
  std::vector<int64_t> sizes = {2, 0, 3};
  at::Tensor result = at::from_blob(zero_data, sizes);
  write_fromblob_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
  delete[] zero_data;
}

// 全一维度 from_blob
TEST_F(FromBlobTest, FromBlobAllOneShape) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "FromBlobAllOneShape ";
  float* one_data = new float[1];
  one_data[0] = 7.0f;
  std::vector<int64_t> sizes = {1, 1, 1};
  at::Tensor result = at::from_blob(one_data, sizes);
  write_fromblob_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
  delete[] one_data;
}

// 非连续 strides from_blob
TEST_F(FromBlobTest, FromBlobNonContiguousStrides) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "FromBlobNonContiguousStrides ";
  // 创建一个更大的 buffer，然后使用非连续 strides
  float* buffer = new float[12];
  for (int i = 0; i < 12; ++i) {
    buffer[i] = static_cast<float>(i);
  }
  std::vector<int64_t> sizes = {2, 2};
  std::vector<int64_t> strides = {6, 3};  // 非连续
  at::Tensor result = at::from_blob(buffer, sizes, strides);
  file << std::to_string(result.is_contiguous()) << " ";
  write_fromblob_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
  delete[] buffer;
}

// ========== Dtype 覆盖 ==========

TEST_F(FromBlobTest, FromBlobFloat64) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "FromBlobFloat64 ";
  double* double_data = new double[4];
  for (int i = 0; i < 4; ++i) {
    double_data[i] = static_cast<double>(i) * 1.5;
  }
  std::vector<int64_t> sizes = {2, 2};
  at::Tensor result =
      at::from_blob(double_data, sizes, at::TensorOptions().dtype(at::kDouble));
  file << std::to_string(static_cast<int>(result.scalar_type())) << " ";
  write_fromblob_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
  delete[] double_data;
}

TEST_F(FromBlobTest, FromBlobInt32) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "FromBlobInt32 ";
  int32_t* int_data = new int32_t[4];
  for (int i = 0; i < 4; ++i) {
    int_data[i] = i * 10;
  }
  std::vector<int64_t> sizes = {2, 2};
  at::Tensor result =
      at::from_blob(int_data, sizes, at::TensorOptions().dtype(at::kInt));
  file << std::to_string(static_cast<int>(result.scalar_type())) << " ";
  write_fromblob_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
  delete[] int_data;
}

TEST_F(FromBlobTest, FromBlobInt64) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "FromBlobInt64 ";
  int64_t* long_data = new int64_t[4];
  for (int i = 0; i < 4; ++i) {
    long_data[i] = i * 100L;
  }
  std::vector<int64_t> sizes = {2, 2};
  at::Tensor result =
      at::from_blob(long_data, sizes, at::TensorOptions().dtype(at::kLong));
  file << std::to_string(static_cast<int>(result.scalar_type())) << " ";
  write_fromblob_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
  delete[] long_data;
}

// ========== 异常行为 ==========

// strides 与 sizes 不匹配
// TEST_F(FromBlobTest, InvalidStrides) {
//   auto file_name = g_custom_param.get();
//   FileManerger file(file_name);
//   file.openAppend();
//   file << "InvalidStrides ";
//   try {
//     std::vector<int64_t> sizes = {2, 3};
//     std::vector<int64_t> strides = {1};  // strides 数量不匹配
//     at::Tensor result = at::from_blob(data_buffer, sizes, strides);
//     write_fromblob_result_to_file(&file, result);
//   } catch (const std::exception&) {
//     file << "exception ";
//   }
//   file << "\n";
//   file.saveFile();
// }

}  // namespace test
}  // namespace at
