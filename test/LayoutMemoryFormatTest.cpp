#include <ATen/ATen.h>
#include <c10/core/Layout.h>
#include <c10/core/MemoryFormat.h>
#include <gtest/gtest.h>

#include <sstream>
#include <string>

#include "../src/file_manager.h"

extern paddle_api_test::ThreadSafeParam g_custom_param;

namespace at {
namespace test {

using paddle_api_test::FileManerger;
using paddle_api_test::ThreadSafeParam;

class LayoutMemoryFormatTest : public ::testing::Test {
 protected:
  void SetUp() override {}
};

// ===================== Layout =====================

// Layout 枚举值
TEST_F(LayoutMemoryFormatTest, LayoutEnumValues) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();
  file << "LayoutEnumValues ";
  file << std::to_string(static_cast<int>(c10::kStrided)) << " ";
  file << std::to_string(static_cast<int>(c10::kSparse)) << " ";
  file << std::to_string(static_cast<int>(c10::kSparseCsr)) << " ";
  file << std::to_string(static_cast<int>(c10::kMkldnn)) << " ";
  file << std::to_string(static_cast<int>(c10::kSparseCsc)) << " ";
  file << std::to_string(static_cast<int>(c10::kSparseBsr)) << " ";
  file << std::to_string(static_cast<int>(c10::kSparseBsc)) << " ";
  file << std::to_string(static_cast<int>(c10::kJagged)) << " ";
  file << "\n";
  file.saveFile();
}

// Layout ostream 输出
TEST_F(LayoutMemoryFormatTest, LayoutOutputStream) {
  std::ostringstream oss;
  oss << c10::kStrided;
  std::string strided_str = oss.str();

  oss.str("");
  oss << c10::kSparse;
  std::string sparse_str = oss.str();

  oss.str("");
  oss << c10::kSparseCsr;
  std::string sparse_csr_str = oss.str();

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "LayoutOutputStream ";
  file << strided_str << " ";
  file << sparse_str << " ";
  file << sparse_csr_str << " ";
  file << "\n";
  file.saveFile();
}

// at 命名空间别名
TEST_F(LayoutMemoryFormatTest, LayoutAtNamespace) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "LayoutAtNamespace ";
  file << std::to_string(static_cast<int>(at::kStrided)) << " ";
  file << std::to_string(static_cast<int>(at::kSparse)) << " ";
  file << std::to_string(static_cast<int>(at::kSparseCsr)) << " ";
  file << "\n";
  file.saveFile();
}

// c10 命名空间别名
TEST_F(LayoutMemoryFormatTest, LayoutTorchNamespace) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "LayoutTorchNamespace ";
  file << std::to_string(static_cast<int>(c10::kStrided)) << " ";
  file << std::to_string(static_cast<int>(c10::kSparse)) << " ";
  file << std::to_string(static_cast<int>(c10::kSparseCsr)) << " ";
  file << "\n";
  file.saveFile();
}

// Layout 比较
TEST_F(LayoutMemoryFormatTest, LayoutComparison) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "LayoutComparison ";
  file << std::to_string(c10::kStrided == at::kStrided ? 1 : 0) << " ";
  file << std::to_string(c10::kSparse == at::kSparse ? 1 : 0) << " ";
  file << std::to_string(c10::kStrided == c10::kSparse ? 1 : 0) << " ";
  file << "\n";
  file.saveFile();
}

// ===================== MemoryFormat =====================

// MemoryFormat 枚举值
TEST_F(LayoutMemoryFormatTest, MemoryFormatEnumValues) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "MemoryFormatEnumValues ";
  file << std::to_string(static_cast<int>(c10::MemoryFormat::Contiguous))
       << " ";
  file << std::to_string(static_cast<int>(c10::MemoryFormat::Preserve)) << " ";
  file << std::to_string(static_cast<int>(c10::MemoryFormat::ChannelsLast))
       << " ";
  file << std::to_string(static_cast<int>(c10::MemoryFormat::ChannelsLast3d))
       << " ";
  file << "\n";
  file.saveFile();
}

// c10 命名空间别名
TEST_F(LayoutMemoryFormatTest, MemoryFormatNamespaces) {
  c10::MemoryFormat mf1 = c10::MemoryFormat::Contiguous;
  c10::MemoryFormat mf2 = c10::MemoryFormat::ChannelsLast;
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "MemoryFormatNamespaces ";
  file << std::to_string(static_cast<int>(mf1)) << " ";
  file << std::to_string(static_cast<int>(mf2)) << " ";
  file << "\n";
  file.saveFile();
}

// MemoryFormat 比较
TEST_F(LayoutMemoryFormatTest, MemoryFormatComparison) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "MemoryFormatComparison ";
  file << std::to_string(c10::MemoryFormat::Contiguous ==
                                 c10::MemoryFormat::Contiguous
                             ? 1
                             : 0)
       << " ";
  file << std::to_string(c10::MemoryFormat::Contiguous ==
                                 c10::MemoryFormat::ChannelsLast
                             ? 1
                             : 0)
       << " ";
  file << "\n";
  file.saveFile();
}

}  // namespace test
}  // namespace at
