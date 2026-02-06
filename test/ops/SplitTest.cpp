#include <ATen/ATen.h>
#include <ATen/core/Tensor.h>
#include <ATen/ops/ones.h>
#include <gtest/gtest.h>
#include <torch/all.h>

#include <string>
#include <vector>

#include "../../src/file_manager.h"

extern paddle_api_test::ThreadSafeParam g_custom_param;

namespace at {
namespace test {

using paddle_api_test::FileManerger;
using paddle_api_test::ThreadSafeParam;

class SplitTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // 创建一个 4x6x8 的 tensor 方便测试
    tensor = at::ones({4, 6, 8}, at::kFloat);
  }

  at::Tensor tensor;
};

// 测试 split - 按大小分割
TEST_F(SplitTest, SplitBySize) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();

  // 在维度0上，每2个元素分割
  std::vector<at::Tensor> splits = tensor.split(2, 0);
  file << std::to_string(splits.size()) << " ";
  for (const auto& split : splits) {
    file << std::to_string(split.sizes()[0]) << " ";
  }
  file.saveFile();
}

// 测试 split - 按大小数组分割
TEST_F(SplitTest, SplitBySizes) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();

  // 在维度1上，分割为大小 [2, 3, 1]
  std::vector<at::Tensor> splits = tensor.split({2, 3, 1}, 1);
  file << std::to_string(splits.size()) << " ";
  for (const auto& split : splits) {
    file << std::to_string(split.sizes()[1]) << " ";
  }
  file.saveFile();
}

// 测试 split_with_sizes
TEST_F(SplitTest, SplitWithSizes) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();

  // 在维度2上，分割为大小 [3, 2, 3]
  std::vector<at::Tensor> splits = tensor.split_with_sizes({3, 2, 3}, 2);
  file << std::to_string(splits.size()) << " ";
  for (const auto& split : splits) {
    file << std::to_string(split.sizes()[2]) << " ";
  }
  file.saveFile();
}

// 测试 unsafe_split
TEST_F(SplitTest, UnsafeSplit) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();

  // 在维度0上，每2个元素分割
  std::vector<at::Tensor> splits = tensor.unsafe_split(2, 0);
  file << std::to_string(splits.size()) << " ";
  for (const auto& split : splits) {
    file << std::to_string(split.sizes()[0]) << " ";
  }
  file.saveFile();
}

// 测试 unsafe_split_with_sizes
TEST_F(SplitTest, UnsafeSplitWithSizes) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();

  // 在维度1上，分割为大小 [2, 4]
  std::vector<at::Tensor> splits = tensor.unsafe_split_with_sizes({2, 4}, 1);
  file << std::to_string(splits.size()) << " ";
  for (const auto& split : splits) {
    file << std::to_string(split.sizes()[1]) << " ";
  }
  file.saveFile();
}

// 测试 tensor_split - 按节数分割
TEST_F(SplitTest, TensorSplitBySections) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();

  // 在维度0上，分割为2个部分（4能被2整除）
  std::vector<at::Tensor> splits = tensor.tensor_split(2, 0);
  file << std::to_string(splits.size()) << " ";
  for (const auto& split : splits) {
    file << std::to_string(split.sizes()[0]) << " ";
  }
  file.saveFile();
}

// 测试 tensor_split - 按索引分割
TEST_F(SplitTest, TensorSplitByIndices) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();

  // 在维度1上，在索引 [2, 4] 处分割
  std::vector<at::Tensor> splits = tensor.tensor_split({2, 4}, 1);
  file << std::to_string(splits.size()) << " ";
  for (const auto& split : splits) {
    file << std::to_string(split.sizes()[1]) << " ";
  }
  file.saveFile();
}

// 测试 tensor_split - 使用 tensor 作为索引
TEST_F(SplitTest, TensorSplitByTensorIndices) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();

  // 创建大小数组 tensor（Paddle要求总和等于维度大小8）
  std::vector<int64_t> indices_data = {2, 3, 3};
  at::Tensor indices =
      at::from_blob(indices_data.data(), {3}, at::kLong).clone();
  // 在维度2上分割
  std::vector<at::Tensor> splits = tensor.tensor_split(indices, 2);
  file << std::to_string(splits.size()) << " ";
  for (const auto& split : splits) {
    file << std::to_string(split.sizes()[2]) << " ";
  }
  file.saveFile();
}

// 测试 hsplit - 按节数水平分割
TEST_F(SplitTest, HsplitBySections) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();

  // 水平分割为3个部分
  std::vector<at::Tensor> splits = tensor.hsplit(3);
  file << std::to_string(splits.size()) << " ";
  for (const auto& split : splits) {
    file << std::to_string(split.sizes()[1]) << " ";
  }
  file.saveFile();
}

// 测试 hsplit - 按索引水平分割
TEST_F(SplitTest, HsplitByIndices) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();

  // 在索引 [2, 4] 处水平分割
  std::vector<at::Tensor> splits = tensor.hsplit({2, 4});
  file << std::to_string(splits.size()) << " ";
  for (const auto& split : splits) {
    file << std::to_string(split.sizes()[1]) << " ";
  }
  file.saveFile();
}

// 测试 vsplit - 按节数垂直分割
TEST_F(SplitTest, VsplitBySections) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();

  // 垂直分割为2个部分
  std::vector<at::Tensor> splits = tensor.vsplit(2);
  file << std::to_string(splits.size()) << " ";
  for (const auto& split : splits) {
    file << std::to_string(split.sizes()[0]) << " ";
  }
  file.saveFile();
}

// 测试 vsplit - 按索引垂直分割
TEST_F(SplitTest, VsplitByIndices) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();

  // 在索引 [1, 3] 处垂直分割
  std::vector<at::Tensor> splits = tensor.vsplit({1, 3});
  file << std::to_string(splits.size()) << " ";
  for (const auto& split : splits) {
    file << std::to_string(split.sizes()[0]) << " ";
  }
  file.saveFile();
}

// 测试 dsplit - 按节数深度分割
TEST_F(SplitTest, DsplitBySections) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();

  // 深度分割为4个部分
  std::vector<at::Tensor> splits = tensor.dsplit(4);
  file << std::to_string(splits.size()) << " ";
  for (const auto& split : splits) {
    file << std::to_string(split.sizes()[2]) << " ";
  }
  file.saveFile();
}

// 测试 dsplit - 按索引深度分割
TEST_F(SplitTest, DsplitByIndices) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();

  // 使用大小数组 [3, 5]（总和为8，等于维度2的大小）
  std::vector<at::Tensor> splits = tensor.dsplit({3, 5});
  file << std::to_string(splits.size()) << " ";
  for (const auto& split : splits) {
    file << std::to_string(split.sizes()[2]) << " ";
  }
  file.saveFile();
}

// 测试 split 不同维度
TEST_F(SplitTest, SplitDifferentDims) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();

  // 在维度0上分割
  std::vector<at::Tensor> splits0 = tensor.split(1, 0);
  file << std::to_string(splits0.size()) << " ";

  // 在维度1上分割
  std::vector<at::Tensor> splits1 = tensor.split(2, 1);
  file << std::to_string(splits1.size()) << " ";

  // 在维度2上分割
  std::vector<at::Tensor> splits2 = tensor.split(4, 2);
  file << std::to_string(splits2.size()) << " ";

  file.saveFile();
}

// 测试不均等分割
TEST_F(SplitTest, UnevenSplit) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();

  // 在维度1上，每5个元素分割（不能整除）
  std::vector<at::Tensor> splits = tensor.split(5, 1);
  file << std::to_string(splits.size()) << " ";
  for (const auto& split : splits) {
    file << std::to_string(split.sizes()[1]) << " ";
  }
  file.saveFile();
}

}  // namespace test
}  // namespace at
