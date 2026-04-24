#include <ATen/core/jit_type.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cctype>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include "src/file_manager.h"

extern paddle_api_test::ThreadSafeParam g_custom_param;

namespace at {
namespace test {

using paddle_api_test::FileManerger;

namespace {

std::string to_lower_ascii(std::string value) {
  std::transform(
      value.begin(), value.end(), value.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
      });
  return value;
}

bool contains_token_ci(const std::string& text, const std::string& token) {
  return to_lower_ascii(text).find(to_lower_ascii(token)) != std::string::npos;
}

}  // namespace

class JitTypeBaseTest : public ::testing::Test {};

TEST_F(JitTypeBaseTest, AnnotationReprAndSubtype) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();
  file << "AnnotationReprAndSubtype ";

  auto int_type = c10::IntType::get();
  auto float_type = c10::FloatType::get();
  auto number_type = c10::NumberType::get();
  auto string_type = c10::StringType::get();
  auto tensor_type = c10::TensorType::get();

  c10::TypePrinter printer =
      [](const c10::Type& type) -> std::optional<std::string> {
    if (type.kind() == c10::TypeKind::IntType) {
      return std::string("renamed_int");
    }
    return std::nullopt;
  };

  file << std::to_string(int_type->kind() == c10::TypeKind::IntType) << " ";
  file << std::to_string(float_type->kind() == c10::TypeKind::FloatType) << " ";
  file << std::to_string(contains_token_ci(int_type->annotation_str(printer),
                                           "renamed_int")
                             ? 1
                             : 0)
       << " ";
  file << std::to_string(
              contains_token_ci(float_type->annotation_str(printer), "float")
                  ? 1
                  : 0)
       << " ";
  file << std::to_string(!string_type->repr_str().empty()) << " ";
  file << std::to_string(contains_token_ci(tensor_type->str(), "tensor") ? 1
                                                                         : 0)
       << " ";
  file << std::to_string(int_type->isSubtypeOf(*number_type)) << " ";
  file << std::to_string(number_type->isSubtypeOf(*int_type)) << " ";
  file << std::to_string(*int_type == *c10::IntType::get()) << " ";
  file << std::to_string(*int_type == *float_type) << " ";

  std::ostringstream out;
  out << *tensor_type;
  file << std::to_string(!out.str().empty()) << " ";
  file << "\n";
  file.saveFile();
}

TEST_F(JitTypeBaseTest, TypePtrAndContainedTypes) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "TypePtrAndContainedTypes ";

  c10::TypePtr int_ptr = c10::IntType::get();
  c10::TypePtr tensor_ptr = c10::TensorType::get();
  bool create_failed = false;

  try {
    (void)int_ptr->createWithContained({tensor_ptr});
  } catch (const std::exception&) {
    create_failed = true;
  }

  file << std::to_string(static_cast<bool>(int_ptr) ? 1 : 0) << " ";
  file << std::to_string(int_ptr.get() != nullptr ? 1 : 0) << " ";
  file << std::to_string((*int_ptr).kind() == c10::TypeKind::IntType) << " ";
  file << std::to_string(int_ptr->containedTypes().empty() ? 1 : 0) << " ";
  file << std::to_string(create_failed ? 1 : 0) << " ";
  file << "\n";
  file.saveFile();
}

}  // namespace test
}  // namespace at
