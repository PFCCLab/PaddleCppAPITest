#include <ATen/ATen.h>
#include <ATen/core/Tensor.h>
#include <ATen/ops/ones.h>
#include <gtest/gtest.h>
#include <torch/all.h>

#include <string>
#include <vector>

#include "../src/file_manager.h"

extern paddle_api_test::ThreadSafeParam g_custom_param;

namespace at {
namespace test {

using paddle_api_test::FileManerger;
using paddle_api_test::ThreadSafeParam;

class LayoutTest : public ::testing::Test {
 protected:
  void SetUp() override {
    std::vector<int64_t> shape = {2, 3, 4};
    tensor = at::ones(shape, at::kFloat);
  }

  at::Tensor tensor;
};

// 测试 layout
TEST_F(LayoutTest, Layout) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();

  // 默认创建的张量应该是 strided 布局
  c10::Layout layout = tensor.layout();
  file << std::to_string(static_cast<int8_t>(layout)) << " ";
  file.saveFile();
}

// 测试 layout 常量别名
TEST_F(LayoutTest, LayoutConstants) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();

  // 测试 c10 命名空间下的常量别名
  file << std::to_string(c10::kStrided == c10::Layout::Strided) << " ";
  file << std::to_string(c10::kSparse == c10::Layout::Sparse) << " ";
  file << std::to_string(c10::kSparseCsr == c10::Layout::SparseCsr) << " ";
  file << std::to_string(c10::kSparseCsc == c10::Layout::SparseCsc) << " ";
  file << std::to_string(c10::kSparseBsr == c10::Layout::SparseBsr) << " ";
  file << std::to_string(c10::kSparseBsc == c10::Layout::SparseBsc) << " ";
  file << std::to_string(c10::kMkldnn == c10::Layout::Mkldnn) << " ";
  file << std::to_string(c10::kJagged == c10::Layout::Jagged) << " ";
  file.saveFile();
}

// 测试 at 命名空间下的 layout 常量
TEST_F(LayoutTest, LayoutConstantsInAtNamespace) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();

  file << std::to_string(at::kStrided == c10::Layout::Strided) << " ";
  file << std::to_string(at::kSparse == c10::Layout::Sparse) << " ";
  file << std::to_string(at::kSparseCsr == c10::Layout::SparseCsr) << " ";
  file << std::to_string(at::kSparseCsc == c10::Layout::SparseCsc) << " ";
  file << std::to_string(at::kSparseBsr == c10::Layout::SparseBsr) << " ";
  file << std::to_string(at::kSparseBsc == c10::Layout::SparseBsc) << " ";
  file << std::to_string(at::kMkldnn == c10::Layout::Mkldnn) << " ";
  file << std::to_string(at::kJagged == c10::Layout::Jagged) << " ";
  file.saveFile();
}

// 测试 torch 命名空间下的 layout 常量
TEST_F(LayoutTest, LayoutConstantsInTorchNamespace) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();

  file << std::to_string(torch::kStrided == c10::Layout::Strided) << " ";
  file << std::to_string(torch::kSparse == c10::Layout::Sparse) << " ";
  file << std::to_string(torch::kSparseCsr == c10::Layout::SparseCsr) << " ";
  file << std::to_string(torch::kSparseCsc == c10::Layout::SparseCsc) << " ";
  file << std::to_string(torch::kSparseBsr == c10::Layout::SparseBsr) << " ";
  file << std::to_string(torch::kSparseBsc == c10::Layout::SparseBsc) << " ";
  file << std::to_string(torch::kMkldnn == c10::Layout::Mkldnn) << " ";
  file << std::to_string(torch::kJagged == c10::Layout::Jagged) << " ";
  file.saveFile();
}

// 测试 layout 枚举值
TEST_F(LayoutTest, LayoutEnumValues) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();

  // 测试 Layout 枚举的底层值
  file << std::to_string(static_cast<int8_t>(c10::Layout::Strided)) << " ";
  file << std::to_string(static_cast<int8_t>(c10::Layout::Sparse)) << " ";
  file << std::to_string(static_cast<int8_t>(c10::Layout::SparseCsr)) << " ";
  file << std::to_string(static_cast<int8_t>(c10::Layout::Mkldnn)) << " ";
  file << std::to_string(static_cast<int8_t>(c10::Layout::SparseCsc)) << " ";
  file << std::to_string(static_cast<int8_t>(c10::Layout::SparseBsr)) << " ";
  file << std::to_string(static_cast<int8_t>(c10::Layout::SparseBsc)) << " ";
  file << std::to_string(static_cast<int8_t>(c10::Layout::Jagged)) << " ";
  file << std::to_string(static_cast<int8_t>(c10::Layout::NumOptions)) << " ";
  file.saveFile();
}

// 测试 layout 输出流操作符
TEST_F(LayoutTest, LayoutOutputStream) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();

  std::ostringstream oss;

  oss.str("");
  oss << c10::Layout::Strided;
  file << oss.str() << " ";

  oss.str("");
  oss << c10::Layout::Sparse;
  file << oss.str() << " ";

  oss.str("");
  oss << c10::Layout::SparseCsr;
  file << oss.str() << " ";

  oss.str("");
  oss << c10::Layout::SparseCsc;
  file << oss.str() << " ";

  oss.str("");
  oss << c10::Layout::SparseBsr;
  file << oss.str() << " ";

  oss.str("");
  oss << c10::Layout::SparseBsc;
  file << oss.str() << " ";

  oss.str("");
  oss << c10::Layout::Mkldnn;
  file << oss.str() << " ";

  oss.str("");
  oss << c10::Layout::Jagged;
  file << oss.str() << " ";

  file.saveFile();
}

// 测试使用 kStrided 常量与 tensor.layout() 比较
TEST_F(LayoutTest, LayoutWithConstant) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();

  // 使用常量别名进行比较
  file << std::to_string(tensor.layout() == at::kStrided) << " ";
  file << std::to_string(tensor.layout() == torch::kStrided) << " ";
  file << std::to_string(tensor.layout() == c10::kStrided) << " ";

  // 确保不是其他布局类型
  file << std::to_string(tensor.layout() != at::kSparse) << " ";
  file << std::to_string(tensor.layout() != at::kSparseCsr) << " ";
  file << std::to_string(tensor.layout() != at::kMkldnn) << " ";
  file.saveFile();
}

}  // namespace test
}  // namespace at
