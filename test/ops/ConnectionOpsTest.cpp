#include <ATen/ATen.h>
#include <ATen/core/Tensor.h>
#include <ATen/ops/cat.h>
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

static void write_cat_result_to_file(FileManerger* file,
                                     const at::Tensor& result) {
  *file << std::to_string(result.dim()) << " ";
  *file << std::to_string(result.numel()) << " ";
  for (int64_t i = 0; i < result.dim(); ++i) {
    *file << std::to_string(result.sizes()[i]) << " ";
  }
}

class ConnectionOpsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    tensor1 = at::zeros({2, 3}, at::kFloat);
    tensor2 = at::zeros({2, 3}, at::kFloat);

    float* data1 = tensor1.data_ptr<float>();
    float* data2 = tensor2.data_ptr<float>();
    for (int64_t i = 0; i < 6; ++i) {
      data1[i] = static_cast<float>(i);
      data2[i] = static_cast<float>(i + 6);
    }
  }

  at::Tensor tensor1;
  at::Tensor tensor2;
};

// ========== 基础功能 ==========

TEST_F(ConnectionOpsTest, CatDim0) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();
  file << "CatDim0 ";
  std::vector<at::Tensor> tensors = {tensor1, tensor2};
  at::Tensor result = at::cat(tensors, 0);
  write_cat_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

TEST_F(ConnectionOpsTest, CatDim1) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "CatDim1 ";
  std::vector<at::Tensor> tensors = {tensor1, tensor2};
  at::Tensor result = at::cat(tensors, 1);
  write_cat_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

TEST_F(ConnectionOpsTest, CatThreeTensors) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "CatThreeTensors ";
  at::Tensor tensor3 = at::zeros({2, 3}, at::kFloat);
  float* data3 = tensor3.data_ptr<float>();
  for (int64_t i = 0; i < 6; ++i) {
    data3[i] = static_cast<float>(i + 12);
  }
  std::vector<at::Tensor> tensors = {tensor1, tensor2, tensor3};
  at::Tensor result = at::cat(tensors, 0);
  write_cat_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// ========== Shape 覆盖 ==========

// 大 shape cat
TEST_F(ConnectionOpsTest, CatLargeShape) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "CatLargeShape ";
  at::Tensor large1 = at::zeros({100, 100}, at::kFloat);
  at::Tensor large2 = at::zeros({100, 100}, at::kFloat);
  std::vector<at::Tensor> tensors = {large1, large2};
  at::Tensor result = at::cat(tensors, 0);
  file << std::to_string(result.numel()) << " ";
  write_cat_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// 零维度 tensor cat
TEST_F(ConnectionOpsTest, CatZeroDim) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "CatZeroDim ";
  at::Tensor zero1 = at::zeros({2, 0, 3}, at::kFloat);
  at::Tensor zero2 = at::zeros({2, 0, 3}, at::kFloat);
  std::vector<at::Tensor> tensors = {zero1, zero2};
  at::Tensor result = at::cat(tensors, 0);
  write_cat_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// 全一维度 cat
TEST_F(ConnectionOpsTest, CatAllOneShape) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "CatAllOneShape ";
  at::Tensor one1 = at::ones({1, 1, 1}, at::kFloat);
  at::Tensor one2 = at::ones({1, 1, 1}, at::kFloat);
  std::vector<at::Tensor> tensors = {one1, one2};
  at::Tensor result = at::cat(tensors, 0);
  write_cat_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// 标量 cat (需要先 unsqueeze)
TEST_F(ConnectionOpsTest, CatScalar) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "CatScalar ";
  at::Tensor scalar1 = at::ones({}, at::kFloat);
  at::Tensor scalar2 = at::ones({}, at::kFloat);
  // 标量需要 unsqueeze 后才能 cat
  at::Tensor s1 = scalar1.unsqueeze(0);
  at::Tensor s2 = scalar2.unsqueeze(0);
  std::vector<at::Tensor> tensors = {s1, s2};
  at::Tensor result = at::cat(tensors, 0);
  write_cat_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// ========== Dtype 覆盖 ==========

TEST_F(ConnectionOpsTest, CatFloat64) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "CatFloat64 ";
  at::Tensor t1 = at::zeros({2, 3}, at::kDouble);
  at::Tensor t2 = at::zeros({2, 3}, at::kDouble);
  std::vector<at::Tensor> tensors = {t1, t2};
  at::Tensor result = at::cat(tensors, 0);
  file << std::to_string(static_cast<int>(result.scalar_type())) << " ";
  write_cat_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

TEST_F(ConnectionOpsTest, CatInt32) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "CatInt32 ";
  at::Tensor t1 = at::zeros({2, 3}, at::kInt);
  at::Tensor t2 = at::zeros({2, 3}, at::kInt);
  std::vector<at::Tensor> tensors = {t1, t2};
  at::Tensor result = at::cat(tensors, 0);
  file << std::to_string(static_cast<int>(result.scalar_type())) << " ";
  write_cat_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

TEST_F(ConnectionOpsTest, CatInt64) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "CatInt64 ";
  at::Tensor t1 = at::zeros({2, 3}, at::kLong);
  at::Tensor t2 = at::zeros({2, 3}, at::kLong);
  std::vector<at::Tensor> tensors = {t1, t2};
  at::Tensor result = at::cat(tensors, 0);
  file << std::to_string(static_cast<int>(result.scalar_type())) << " ";
  write_cat_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// ========== API 变体 ==========

// 负索引 dim
TEST_F(ConnectionOpsTest, CatNegativeDim) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "CatNegativeDim ";
  std::vector<at::Tensor> tensors = {tensor1, tensor2};
  at::Tensor result = at::cat(tensors, -1);  // 使用负索引
  write_cat_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// 单个 tensor cat
TEST_F(ConnectionOpsTest, CatSingleTensor) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "CatSingleTensor ";
  std::vector<at::Tensor> tensors = {tensor1};
  at::Tensor result = at::cat(tensors, 0);
  write_cat_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// ========== 异常行为 ==========

// 空 tensor 列表
TEST_F(ConnectionOpsTest, CatEmptyList) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "CatEmptyList ";
  try {
    std::vector<at::Tensor> tensors = {};
    at::Tensor result = at::cat(tensors, 0);
    write_cat_result_to_file(&file, result);
  } catch (const std::exception&) {
    file << "exception ";
  }
  file << "\n";
  file.saveFile();
}

// 维度不匹配
TEST_F(ConnectionOpsTest, CatMismatchedDims) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "CatMismatchedDims ";
  try {
    at::Tensor t1 = at::zeros({2, 3}, at::kFloat);
    at::Tensor t2 = at::zeros({2, 4}, at::kFloat);  // 第二维不匹配
    std::vector<at::Tensor> tensors = {t1, t2};
    at::Tensor result = at::cat(tensors, 0);
    write_cat_result_to_file(&file, result);
  } catch (const std::exception&) {
    file << "exception ";
  }
  file << "\n";
  file.saveFile();
}

// 无效 dim
TEST_F(ConnectionOpsTest, CatInvalidDim) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "CatInvalidDim ";
  try {
    std::vector<at::Tensor> tensors = {tensor1, tensor2};
    at::Tensor result = at::cat(tensors, 10);  // dim 越界
    write_cat_result_to_file(&file, result);
  } catch (const std::exception&) {
    file << "exception ";
  }
  file << "\n";
  file.saveFile();
}

}  // namespace test
}  // namespace at
