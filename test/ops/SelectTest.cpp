#include <ATen/ATen.h>
#include <ATen/core/Tensor.h>
#include <ATen/ops/zeros.h>
#include <c10/core/SymInt.h>
#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "../../src/file_manager.h"

extern paddle_api_test::ThreadSafeParam g_custom_param;

namespace at {
namespace test {

using paddle_api_test::FileManerger;
using paddle_api_test::ThreadSafeParam;

class SelectTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // 创建一个 3x4x5 的三维 tensor
    test_tensor = at::zeros({3, 4, 5}, at::kFloat);
    float* data = test_tensor.data_ptr<float>();
    for (int64_t i = 0; i < 60; ++i) {
      data[i] = static_cast<float>(i);
    }
  }
  at::Tensor test_tensor;
};

static void write_select_result_to_file(FileManerger* file,
                                        const at::Tensor& result) {
  *file << std::to_string(result.dim()) << " ";
  *file << std::to_string(result.numel()) << " ";

  // 写入形状信息
  for (int64_t i = 0; i < result.dim(); ++i) {
    *file << std::to_string(result.sizes()[i]) << " ";
  }

  // 写入前几个数据值
  float* result_data = result.data_ptr<float>();
  int64_t max_elements = std::min(result.numel(), static_cast<int64_t>(10));
  for (int64_t i = 0; i < max_elements; ++i) {
    *file << std::to_string(result_data[i]) << " ";
  }
}

// 测试 select 第 0 维
TEST_F(SelectTest, SelectDim0) {
  // 从第 0 维选择索引 1，结果应该是 4x5
  at::Tensor result = test_tensor.select(0, 1);
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();
  write_select_result_to_file(&file, result);
  file.saveFile();
}

// 测试 select 第 1 维
TEST_F(SelectTest, SelectDim1) {
  // 从第 1 维选择索引 2，结果应该是 3x5
  at::Tensor result = test_tensor.select(1, 2);
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();
  write_select_result_to_file(&file, result);
  file.saveFile();
}

// 测试 select 第 2 维
TEST_F(SelectTest, SelectDim2) {
  // 从第 2 维选择索引 3，结果应该是 3x4
  at::Tensor result = test_tensor.select(2, 3);
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  write_select_result_to_file(&file, result);
  file.saveFile();
}

// 测试 select 使用负数索引
TEST_F(SelectTest, SelectNegativeIndex) {
  // 从第 0 维选择索引 -1（最后一个），结果应该是 4x5
  at::Tensor result = test_tensor.select(0, -1);
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  write_select_result_to_file(&file, result);
  file.saveFile();
}

// 测试 select 链式调用
TEST_F(SelectTest, SelectChain) {
  // 先选择第 0 维的索引 1，再选择第 0 维的索引 2，最后选择第 0 维的索引 3
  at::Tensor result = test_tensor.select(0, 1).select(0, 2).select(0, 3);
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();

  file << std::to_string(result.dim()) << " ";
  file << std::to_string(result.numel()) << " ";

  // 对于标量或一维 tensor，直接输出值
  if (result.numel() == 1) {
    file << std::to_string(result.item<float>()) << " ";
  } else {
    float* data = result.data_ptr<float>();
    for (int64_t i = 0; i < std::min(result.numel(), static_cast<int64_t>(5));
         ++i) {
      file << std::to_string(data[i]) << " ";
    }
  }
  file.saveFile();
}

// 测试 select_symint 方法
TEST_F(SelectTest, SelectSymInt) {
  // 使用 SymInt 选择第 1 维的索引 1
  c10::SymInt sym_index(1);
  at::Tensor result = test_tensor.select_symint(1, sym_index);
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  write_select_result_to_file(&file, result);
  file.saveFile();
}

// 测试 select_symint 使用不同的维度和索引
TEST_F(SelectTest, SelectSymIntVariousIndices) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();

  // 选择第 0 维的索引 0
  c10::SymInt sym_index_0(0);
  at::Tensor result1 = test_tensor.select_symint(0, sym_index_0);
  file << std::to_string(result1.dim()) << " ";
  file << std::to_string(result1.numel()) << " ";

  // 选择第 2 维的索引 4
  c10::SymInt sym_index_4(4);
  at::Tensor result2 = test_tensor.select_symint(2, sym_index_4);
  file << std::to_string(result2.dim()) << " ";
  file << std::to_string(result2.numel()) << " ";

  file.saveFile();
}

// 测试二维 tensor 的 select
TEST_F(SelectTest, Select2DTensor) {
  at::Tensor tensor_2d = at::zeros({4, 5}, at::kFloat);
  float* data = tensor_2d.data_ptr<float>();
  for (int64_t i = 0; i < 20; ++i) {
    data[i] = static_cast<float>(i * 2);
  }

  // 选择第 0 维的索引 2，结果应该是一维 tensor，大小为 5
  at::Tensor result = tensor_2d.select(0, 2);
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();

  file << std::to_string(result.dim()) << " ";
  file << std::to_string(result.numel()) << " ";

  float* result_data = result.data_ptr<float>();
  for (int64_t i = 0; i < result.numel(); ++i) {
    file << std::to_string(result_data[i]) << " ";
  }
  file.saveFile();
}

// 测试一维 tensor 的 select
TEST_F(SelectTest, Select1DTensor) {
  at::Tensor tensor_1d = at::zeros({10}, at::kFloat);
  float* data = tensor_1d.data_ptr<float>();
  for (int64_t i = 0; i < 10; ++i) {
    data[i] = static_cast<float>(i * 10);
  }

  // 选择第 0 维的索引 5，结果应该是标量
  at::Tensor result = tensor_1d.select(0, 5);
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();

  file << std::to_string(result.dim()) << " ";
  file << std::to_string(result.numel()) << " ";
  file << std::to_string(result.item<float>()) << " ";
  file.saveFile();
}

// 测试 select 返回的 view 与原始 tensor 共享存储
TEST_F(SelectTest, SelectViewSharing) {
  at::Tensor selected = test_tensor.select(0, 0);

  // 修改 selected 的数据
  float* selected_data = selected.data_ptr<float>();
  float original_value = selected_data[0];
  selected_data[0] = 999.0f;

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();

  // 验证原始 tensor 的对应位置也被修改
  float* original_data = test_tensor.data_ptr<float>();
  file << std::to_string(original_value) << " ";
  file << std::to_string(original_data[0]) << " ";
  file << std::to_string(selected_data[0]) << " ";

  // 恢复数据
  selected_data[0] = original_value;
  file.saveFile();
}

}  // namespace test
}  // namespace at
