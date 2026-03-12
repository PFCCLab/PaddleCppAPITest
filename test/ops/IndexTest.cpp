#include <ATen/ATen.h>
#include <ATen/TensorIndexing.h>
#include <ATen/core/Tensor.h>
#include <ATen/ops/arange.h>
#include <ATen/ops/zeros.h>
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
  void SetUp() override {
    // 创建 4x5x6 tensor
    tensor = at::zeros({4, 5, 6}, at::kFloat);
    float* data = tensor.data_ptr<float>();
    for (int64_t i = 0; i < 120; ++i) {
      data[i] = static_cast<float>(i);
    }
  }

  at::Tensor tensor;
};

// 输出 shape、numel、stride、is_contiguous
static void write_index_result_to_file(FileManerger* file,
                                       const at::Tensor& result) {
  *file << std::to_string(result.dim()) << " ";
  *file << std::to_string(result.numel()) << " ";
  for (int64_t i = 0; i < result.dim(); ++i) {
    *file << std::to_string(result.sizes()[i]) << " ";
  }
  auto strides = result.strides();
  for (int64_t i = 0; i < result.dim(); ++i) {
    *file << std::to_string(strides[i]) << " ";
  }
  *file << std::to_string(static_cast<int>(result.is_contiguous())) << " ";
}

// Slice 构造测试
TEST_F(IndexTest, SliceConstruction) {
  at::indexing::Slice s1(0, 3);
  at::indexing::Slice s2(1, 5, 2);
  at::indexing::Slice s3;  // 默认构造
  (void)s3;

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();
#if USE_PADDLE_API
  file << std::to_string(s1.start()) << " ";
  file << std::to_string(s1.stop()) << " ";
  file << std::to_string(s1.step()) << " ";
  file << std::to_string(s2.start()) << " ";
  file << std::to_string(s2.stop()) << " ";
  file << std::to_string(s2.step()) << " ";
#else
  file << std::to_string(s1.start().expect_int()) << " ";
  file << std::to_string(s1.stop().expect_int()) << " ";
  file << std::to_string(s1.step().expect_int()) << " ";
  file << std::to_string(s2.start().expect_int()) << " ";
  file << std::to_string(s2.stop().expect_int()) << " ";
  file << std::to_string(s2.step().expect_int()) << " ";
#endif
  file.saveFile();
}

// 单维度 Slice 索引：tensor[1:3, :, :]
TEST_F(IndexTest, IndexSingleSlice) {
  using at::indexing::Slice;
  at::Tensor result = tensor.index({Slice(1, 3), Slice(0, 5), Slice(0, 6)});
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  write_index_result_to_file(&file, result);
  at::Tensor c = result.contiguous();
  float* data = c.data_ptr<float>();
  file << std::to_string(data[0]) << " ";
  file.saveFile();
}

// 多维度 Slice：tensor[0:2, 1:4, 2:5]
TEST_F(IndexTest, IndexMultiSlice) {
  using at::indexing::Slice;
  at::Tensor result = tensor.index({Slice(0, 2), Slice(1, 4), Slice(2, 5)});
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  write_index_result_to_file(&file, result);
  at::Tensor c = result.contiguous();
  float* data = c.data_ptr<float>();
  for (int64_t i = 0; i < c.numel(); ++i) {
    file << std::to_string(data[i]) << " ";
  }
  file.saveFile();
}

// 成员函数版本
TEST_F(IndexTest, IndexMemberFunction) {
  using at::indexing::Slice;
  at::Tensor result = tensor.index({Slice(0, 2), Slice(0, 3), Slice(0, 4)});
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  write_index_result_to_file(&file, result);
  at::Tensor c = result.contiguous();
  float* data = c.data_ptr<float>();
  file << std::to_string(data[0]) << " ";
  file << std::to_string(data[c.numel() - 1]) << " ";
  file.saveFile();
}

// 2D tensor 索引
TEST_F(IndexTest, Index2D) {
  using at::indexing::Slice;
  at::Tensor t2d = at::zeros({6, 8}, at::kFloat);
  float* data = t2d.data_ptr<float>();
  for (int64_t i = 0; i < 48; ++i) {
    data[i] = static_cast<float>(i);
  }
  at::Tensor result = t2d.index({Slice(1, 4), Slice(2, 6)});
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  write_index_result_to_file(&file, result);
  at::Tensor c = result.contiguous();
  float* rdata = c.data_ptr<float>();
  for (int64_t i = 0; i < c.numel(); ++i) {
    file << std::to_string(rdata[i]) << " ";
  }
  file.saveFile();
}

// Double 类型
TEST_F(IndexTest, IndexDouble) {
  using at::indexing::Slice;
  at::Tensor td = at::zeros({4, 5}, at::kDouble);
  double* data = td.data_ptr<double>();
  for (int64_t i = 0; i < 20; ++i) {
    data[i] = static_cast<double>(i) * 0.5;
  }
  at::Tensor result = td.index({Slice(1, 3), Slice(0, 4)});
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << std::to_string(result.dim()) << " ";
  file << std::to_string(result.numel()) << " ";
  file << std::to_string(static_cast<int>(result.scalar_type())) << " ";
  auto strides = result.strides();
  for (int64_t i = 0; i < result.dim(); ++i) {
    file << std::to_string(strides[i]) << " ";
  }
  file << std::to_string(static_cast<int>(result.is_contiguous())) << " ";
  at::Tensor c = result.contiguous();
  double* rdata = c.data_ptr<double>();
  for (int64_t i = 0; i < c.numel(); ++i) {
    file << std::to_string(rdata[i]) << " ";
  }
  file.saveFile();
}

// Int 类型
TEST_F(IndexTest, IndexInt) {
  using at::indexing::Slice;
  at::Tensor ti = at::zeros({4, 5}, at::kInt);
  int* data = ti.data_ptr<int>();
  for (int64_t i = 0; i < 20; ++i) {
    data[i] = static_cast<int>(i) - 10;
  }
  at::Tensor result = ti.index({Slice(0, 2), Slice(1, 4)});
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << std::to_string(result.dim()) << " ";
  file << std::to_string(result.numel()) << " ";
  file << std::to_string(static_cast<int>(result.scalar_type())) << " ";
  auto strides = result.strides();
  for (int64_t i = 0; i < result.dim(); ++i) {
    file << std::to_string(strides[i]) << " ";
  }
  file << std::to_string(static_cast<int>(result.is_contiguous())) << " ";
  at::Tensor c = result.contiguous();
  int* rdata = c.data_ptr<int>();
  for (int64_t i = 0; i < c.numel(); ++i) {
    file << std::to_string(rdata[i]) << " ";
  }
  file.saveFile();
}

// Long 类型
TEST_F(IndexTest, IndexLong) {
  using at::indexing::Slice;
  at::Tensor tl = at::zeros({4, 5}, at::kLong);
  int64_t* data = tl.data_ptr<int64_t>();
  for (int64_t i = 0; i < 20; ++i) {
    data[i] = i * 100;
  }
  at::Tensor result = tl.index({Slice(2, 4), Slice(0, 3)});
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << std::to_string(result.dim()) << " ";
  file << std::to_string(result.numel()) << " ";
  file << std::to_string(static_cast<int>(result.scalar_type())) << " ";
  auto strides = result.strides();
  for (int64_t i = 0; i < result.dim(); ++i) {
    file << std::to_string(strides[i]) << " ";
  }
  file << std::to_string(static_cast<int>(result.is_contiguous())) << " ";
  at::Tensor c = result.contiguous();
  int64_t* rdata = c.data_ptr<int64_t>();
  for (int64_t i = 0; i < c.numel(); ++i) {
    file << std::to_string(rdata[i]) << " ";
  }
  file.saveFile();
}

// 大 shape
TEST_F(IndexTest, IndexLargeShape) {
  using at::indexing::Slice;
  at::Tensor large = at::zeros({100, 100}, at::kFloat);
  float* data = large.data_ptr<float>();
  for (int64_t i = 0; i < 10000; ++i) {
    data[i] = static_cast<float>(i);
  }
  at::Tensor result = large.index({Slice(10, 90), Slice(20, 80)});
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  write_index_result_to_file(&file, result);
  at::Tensor c = result.contiguous();
  float* rdata = c.data_ptr<float>();
  file << std::to_string(rdata[0]) << " ";
  file << std::to_string(rdata[c.numel() - 1]) << " ";
  file.saveFile();
}

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
