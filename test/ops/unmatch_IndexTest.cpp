#include <ATen/ATen.h>
#include <ATen/core/Tensor.h>
#include <ATen/indexing.h>
#include <ATen/ops/index.h>
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

static void write_index_result_to_file(FileManerger* file,
                                       const at::Tensor& result) {
  *file << std::to_string(result.dim()) << " ";
  *file << std::to_string(result.numel()) << " ";
  for (int64_t i = 0; i < result.dim(); ++i) {
    *file << std::to_string(result.sizes()[i]) << " ";
  }
}

// Slice 构造测试
TEST_F(IndexTest, SliceConstruction) {
  at::indexing::Slice s1(0, 3);
  at::indexing::Slice s2(1, 5, 2);
  at::indexing::Slice s3;  // 默认构造

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();
  file << std::to_string(s1.start()) << " ";
  file << std::to_string(s1.stop()) << " ";
  file << std::to_string(s1.step()) << " ";
  file << std::to_string(s2.start()) << " ";
  file << std::to_string(s2.stop()) << " ";
  file << std::to_string(s2.step()) << " ";
  file.saveFile();
}

// 单维度 Slice 索引：tensor[1:3, :, :]
TEST_F(IndexTest, IndexSingleSlice) {
  std::vector<at::indexing::Slice> indices = {at::indexing::Slice(1, 3),
                                              at::indexing::Slice(0, 5),
                                              at::indexing::Slice(0, 6)};
  at::Tensor result = at::index(tensor, indices);
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  write_index_result_to_file(&file, result);
  float* data = result.data_ptr<float>();
  file << std::to_string(data[0]) << " ";
  file.saveFile();
}

// 多维度 Slice：tensor[0:2, 1:4, 2:5]
TEST_F(IndexTest, IndexMultiSlice) {
  std::vector<at::indexing::Slice> indices = {at::indexing::Slice(0, 2),
                                              at::indexing::Slice(1, 4),
                                              at::indexing::Slice(2, 5)};
  at::Tensor result = at::index(tensor, indices);
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  write_index_result_to_file(&file, result);
  float* data = result.data_ptr<float>();
  for (int64_t i = 0; i < result.numel(); ++i) {
    file << std::to_string(data[i]) << " ";
  }
  file.saveFile();
}

// 成员函数版本
TEST_F(IndexTest, IndexMemberFunction) {
  std::vector<at::indexing::Slice> indices = {at::indexing::Slice(0, 2),
                                              at::indexing::Slice(0, 3),
                                              at::indexing::Slice(0, 4)};
  at::Tensor result = tensor.index(indices);
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  write_index_result_to_file(&file, result);
  float* data = result.data_ptr<float>();
  file << std::to_string(data[0]) << " ";
  file << std::to_string(data[result.numel() - 1]) << " ";
  file.saveFile();
}

// 2D tensor 索引
TEST_F(IndexTest, Index2D) {
  at::Tensor t2d = at::zeros({6, 8}, at::kFloat);
  float* data = t2d.data_ptr<float>();
  for (int64_t i = 0; i < 48; ++i) {
    data[i] = static_cast<float>(i);
  }
  std::vector<at::indexing::Slice> indices = {at::indexing::Slice(1, 4),
                                              at::indexing::Slice(2, 6)};
  at::Tensor result = at::index(t2d, indices);
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  write_index_result_to_file(&file, result);
  float* rdata = result.data_ptr<float>();
  for (int64_t i = 0; i < result.numel(); ++i) {
    file << std::to_string(rdata[i]) << " ";
  }
  file.saveFile();
}

// Double 类型
TEST_F(IndexTest, IndexDouble) {
  at::Tensor td = at::zeros({4, 5}, at::kDouble);
  double* data = td.data_ptr<double>();
  for (int64_t i = 0; i < 20; ++i) {
    data[i] = static_cast<double>(i) * 0.5;
  }
  std::vector<at::indexing::Slice> indices = {at::indexing::Slice(1, 3),
                                              at::indexing::Slice(0, 4)};
  at::Tensor result = at::index(td, indices);
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << std::to_string(result.dim()) << " ";
  file << std::to_string(result.numel()) << " ";
  file << std::to_string(static_cast<int>(result.scalar_type())) << " ";
  double* rdata = result.data_ptr<double>();
  for (int64_t i = 0; i < result.numel(); ++i) {
    file << std::to_string(rdata[i]) << " ";
  }
  file.saveFile();
}

// Int 类型
TEST_F(IndexTest, IndexInt) {
  at::Tensor ti = at::zeros({4, 5}, at::kInt);
  int* data = ti.data_ptr<int>();
  for (int64_t i = 0; i < 20; ++i) {
    data[i] = static_cast<int>(i) - 10;
  }
  std::vector<at::indexing::Slice> indices = {at::indexing::Slice(0, 2),
                                              at::indexing::Slice(1, 4)};
  at::Tensor result = at::index(ti, indices);
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << std::to_string(result.dim()) << " ";
  file << std::to_string(result.numel()) << " ";
  file << std::to_string(static_cast<int>(result.scalar_type())) << " ";
  int* rdata = result.data_ptr<int>();
  for (int64_t i = 0; i < result.numel(); ++i) {
    file << std::to_string(rdata[i]) << " ";
  }
  file.saveFile();
}

// Long 类型
TEST_F(IndexTest, IndexLong) {
  at::Tensor tl = at::zeros({4, 5}, at::kLong);
  int64_t* data = tl.data_ptr<int64_t>();
  for (int64_t i = 0; i < 20; ++i) {
    data[i] = i * 100;
  }
  std::vector<at::indexing::Slice> indices = {at::indexing::Slice(2, 4),
                                              at::indexing::Slice(0, 3)};
  at::Tensor result = at::index(tl, indices);
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << std::to_string(result.dim()) << " ";
  file << std::to_string(result.numel()) << " ";
  file << std::to_string(static_cast<int>(result.scalar_type())) << " ";
  int64_t* rdata = result.data_ptr<int64_t>();
  for (int64_t i = 0; i < result.numel(); ++i) {
    file << std::to_string(rdata[i]) << " ";
  }
  file.saveFile();
}

// 大 shape
TEST_F(IndexTest, IndexLargeShape) {
  at::Tensor large = at::zeros({100, 100}, at::kFloat);
  float* data = large.data_ptr<float>();
  for (int64_t i = 0; i < 10000; ++i) {
    data[i] = static_cast<float>(i);
  }
  std::vector<at::indexing::Slice> indices = {at::indexing::Slice(10, 90),
                                              at::indexing::Slice(20, 80)};
  at::Tensor result = at::index(large, indices);
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  write_index_result_to_file(&file, result);
  float* rdata = result.data_ptr<float>();
  file << std::to_string(rdata[0]) << " ";
  file << std::to_string(rdata[result.numel() - 1]) << " ";
  file.saveFile();
}

}  // namespace test
}  // namespace at
