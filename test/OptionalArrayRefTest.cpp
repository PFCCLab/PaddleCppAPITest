#include <ATen/ATen.h>
#include <c10/util/OptionalArrayRef.h>
#include <gtest/gtest.h>

#include <optional>
#include <vector>

#include "../../src/file_manager.h"

extern paddle_api_test::ThreadSafeParam g_custom_param;

namespace at {
namespace test {

using paddle_api_test::FileManerger;
using paddle_api_test::ThreadSafeParam;

class OptionalArrayRefTest : public ::testing::Test {
 protected:
  void SetUp() override {}
};

// 默认构造（空）
TEST_F(OptionalArrayRefTest, DefaultConstruction) {
  c10::OptionalArrayRef<int64_t> arr;
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();
  file << std::to_string(arr.has_value() ? 1 : 0) << " ";
  file << std::to_string(arr.has_value() ? arr->size() : -1) << " ";
  file.saveFile();
}

// 从 std::nullopt 构造
TEST_F(OptionalArrayRefTest, NulloptConstruction) {
  c10::OptionalArrayRef<int64_t> arr(std::nullopt);
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << std::to_string(arr.has_value() ? 1 : 0) << " ";
  file.saveFile();
}

// 从单个元素构造
TEST_F(OptionalArrayRefTest, SingleElement) {
  c10::OptionalArrayRef<int64_t> arr(42);
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << std::to_string(arr.has_value() ? 1 : 0) << " ";
  file << std::to_string(arr->size()) << " ";
  file << std::to_string(arr->front()) << " ";
  file << std::to_string(arr->back()) << " ";
  file.saveFile();
}

// 从 std::vector 构造
TEST_F(OptionalArrayRefTest, FromVector) {
  std::vector<int64_t> vec = {10, 20, 30, 40, 50};
  c10::OptionalArrayRef<int64_t> arr(vec);
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << std::to_string(arr.has_value() ? 1 : 0) << " ";
  file << std::to_string(arr->size()) << " ";
  for (size_t i = 0; i < arr->size(); ++i) {
    file << std::to_string((*arr)[i]) << " ";
  }
  file.saveFile();
}

// 从 initializer_list 构造
TEST_F(OptionalArrayRefTest, FromInitializerList) {
  c10::OptionalArrayRef<int64_t> arr({1, 2, 3, 4});
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << std::to_string(arr.has_value() ? 1 : 0) << " ";
  file << std::to_string(arr->size()) << " ";
  for (const auto& v : *arr) {
    file << std::to_string(v) << " ";
  }
  file.saveFile();
}

// 从 std::optional<ArrayRef> 构造
TEST_F(OptionalArrayRefTest, FromOptionalArrayRef) {
  std::optional<c10::ArrayRef<int64_t>> opt_arr(std::vector<int64_t>{5, 6, 7});
  c10::OptionalArrayRef<int64_t> arr(opt_arr);
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << std::to_string(arr.has_value() ? 1 : 0) << " ";
  file << std::to_string(arr->size()) << " ";
  file << std::to_string(arr->front()) << " ";
  file.saveFile();
}

// 从 pointer + length 构造
TEST_F(OptionalArrayRefTest, PointerLength) {
  int64_t data[] = {100, 200, 300};
  c10::OptionalArrayRef<int64_t> arr(data, 3);
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << std::to_string(arr.has_value() ? 1 : 0) << " ";
  file << std::to_string(arr->size()) << " ";
  for (size_t i = 0; i < arr->size(); ++i) {
    file << std::to_string(arr->at(i)) << " ";
  }
  file.saveFile();
}

// operator= 赋值
TEST_F(OptionalArrayRefTest, AssignmentOperator) {
  c10::OptionalArrayRef<int64_t> arr;
  arr = std::nullopt;
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << std::to_string(arr.has_value() ? 1 : 0) << " ";
  arr = std::vector<int64_t>{1, 2, 3};
  file << std::to_string(arr.has_value() ? 1 : 0) << " ";
  file << std::to_string(arr->size()) << " ";
  file.saveFile();
}

// operator bool
TEST_F(OptionalArrayRefTest, BoolOperator) {
  c10::OptionalArrayRef<int64_t> empty_arr;
  c10::OptionalArrayRef<int64_t> valid_arr({1, 2});
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << std::to_string(static_cast<bool>(empty_arr) ? 1 : 0) << " ";
  file << std::to_string(static_cast<bool>(valid_arr) ? 1 : 0) << " ";
  file.saveFile();
}

// value() 方法
TEST_F(OptionalArrayRefTest, ValueMethod) {
  c10::OptionalArrayRef<int64_t> arr({7, 8, 9});
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  auto& ref = arr.value();
  file << std::to_string(ref.size()) << " ";
  file << std::to_string(ref.front()) << " ";
  file << std::to_string(ref.back()) << " ";
  file.saveFile();
}

// value_or 方法
TEST_F(OptionalArrayRefTest, ValueOrMethod) {
  c10::OptionalArrayRef<int64_t> empty_arr;
  c10::OptionalArrayRef<int64_t> valid_arr({1, 2, 3});
  std::vector<int64_t> default_vec = {9, 9, 9};

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();

  auto& empty_result = empty_arr.value_or(default_vec);
  file << std::to_string(empty_result.size()) << " ";
  for (size_t i = 0; i < empty_result.size(); ++i) {
    file << std::to_string(empty_result[i]) << " ";
  }

  auto& valid_result = valid_arr.value_or(default_vec);
  file << std::to_string(valid_result.size()) << " ";
  for (size_t i = 0; i < valid_result.size(); ++i) {
    file << std::to_string(valid_result[i]) << " ";
  }
  file.saveFile();
}

// reset 方法
TEST_F(OptionalArrayRefTest, ResetMethod) {
  c10::OptionalArrayRef<int64_t> arr({1, 2, 3});
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << std::to_string(arr.has_value() ? 1 : 0) << " ";
  arr.reset();
  file << std::to_string(arr.has_value() ? 1 : 0) << " ";
  file.saveFile();
}

// swap 方法
TEST_F(OptionalArrayRefTest, SwapMethod) {
  c10::OptionalArrayRef<int64_t> arr1({1, 2});
  c10::OptionalArrayRef<int64_t> arr2({3, 4, 5});
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << std::to_string(arr1->size()) << " " << std::to_string(arr2->size())
       << " ";
  arr1.swap(arr2);
  file << std::to_string(arr1->size()) << " " << std::to_string(arr2->size())
       << " ";
  file.saveFile();
}

// emplace 方法
TEST_F(OptionalArrayRefTest, EmplaceMethod) {
  c10::OptionalArrayRef<int64_t> arr;
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  arr.emplace(1, 2, 3, 4);
  file << std::to_string(arr.has_value() ? 1 : 0) << " ";
  file << std::to_string(arr->size()) << " ";
  for (const auto& v : *arr) {
    file << std::to_string(v) << " ";
  }
  file.saveFile();
}

// slice 方法
TEST_F(OptionalArrayRefTest, SliceMethod) {
  std::vector<int64_t> vec = {0, 1, 2, 3, 4, 5};
  c10::OptionalArrayRef<int64_t> arr(vec);
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  auto sliced = arr->slice(1, 3);  // [1, 2, 3]
  file << std::to_string(sliced.size()) << " ";
  for (size_t i = 0; i < sliced.size(); ++i) {
    file << std::to_string(sliced[i]) << " ";
  }
  file.saveFile();
}

// operator==
TEST_F(OptionalArrayRefTest, EqualityOperator) {
  c10::OptionalArrayRef<int64_t> arr1({1, 2, 3});
  c10::OptionalArrayRef<int64_t> arr2({1, 2, 3});
  c10::OptionalArrayRef<int64_t> arr3({1, 2, 4});
  c10::IntArrayRef direct_ref({1, 2, 3});

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << std::to_string(arr1 == arr2 ? 1 : 0) << " ";
  file << std::to_string(arr1 == arr3 ? 1 : 0) << " ";
  file << std::to_string(arr1 == direct_ref ? 1 : 0) << " ";
  file << std::to_string(direct_ref == arr1 ? 1 : 0) << " ";
  file.saveFile();
}

// OptionalIntArrayRef 别名测试
TEST_F(OptionalArrayRefTest, OptionalIntArrayRef) {
  c10::OptionalIntArrayRef arr({10, 20, 30});
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << std::to_string(arr.has_value() ? 1 : 0) << " ";
  file << std::to_string(arr->size()) << " ";
  for (size_t i = 0; i < arr->size(); ++i) {
    file << std::to_string((*arr)[i]) << " ";
  }
  file.saveFile();
}

// float 类型 OptionalArrayRef
TEST_F(OptionalArrayRefTest, FloatOptionalArrayRef) {
  c10::OptionalArrayRef<float> arr({1.5f, 2.5f, 3.5f});
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << std::to_string(arr.has_value() ? 1 : 0) << " ";
  file << std::to_string(arr->size()) << " ";
  for (const auto& v : *arr) {
    file << std::to_string(v) << " ";
  }
  file.saveFile();
}

// copy 构造
TEST_F(OptionalArrayRefTest, CopyConstruction) {
  c10::OptionalArrayRef<int64_t> arr1({5, 6, 7});
  c10::OptionalArrayRef<int64_t> arr2(arr1);
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << std::to_string(arr2.has_value() ? 1 : 0) << " ";
  file << std::to_string(arr2->size()) << " ";
  file << std::to_string(arr2->front()) << " ";
  file.saveFile();
}

// move 构造
TEST_F(OptionalArrayRefTest, MoveConstruction) {
  c10::OptionalArrayRef<int64_t> arr1({8, 9, 10});
  c10::OptionalArrayRef<int64_t> arr2(std::move(arr1));
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << std::to_string(arr2.has_value() ? 1 : 0) << " ";
  file << std::to_string(arr2->size()) << " ";
  file << std::to_string(arr2->front()) << " ";
  file.saveFile();
}

// empty array
TEST_F(OptionalArrayRefTest, EmptyArray) {
  std::vector<int64_t> empty_vec;
  c10::OptionalArrayRef<int64_t> arr(empty_vec);
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << std::to_string(arr.has_value() ? 1 : 0) << " ";
  file << std::to_string(arr->empty() ? 1 : 0) << " ";
  file << std::to_string(arr->size()) << " ";
  file.saveFile();
}

// from std::in_place_t
TEST_F(OptionalArrayRefTest, InPlaceConstruction) {
  c10::OptionalArrayRef<int64_t> arr(std::in_place, 1, 2, 3, 4, 5);
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << std::to_string(arr.has_value() ? 1 : 0) << " ";
  file << std::to_string(arr->size()) << " ";
  for (const auto& v : *arr) {
    file << std::to_string(v) << " ";
  }
  file.saveFile();
}

}  // namespace test
}  // namespace at
