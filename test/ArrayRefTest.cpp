#include <ATen/ATen.h>
#include <c10/util/ArrayRef.h>
#include <gtest/gtest.h>

#include <array>
#include <string>
#include <vector>

#include "../src/file_manager.h"

extern paddle_api_test::ThreadSafeParam g_custom_param;

namespace at {
namespace test {

using paddle_api_test::FileManerger;
using paddle_api_test::ThreadSafeParam;

class ArrayRefTest : public ::testing::Test {
 protected:
  void SetUp() override {}
};

// 默认构造（空）
TEST_F(ArrayRefTest, DefaultConstruction) {
  c10::ArrayRef<int64_t> arr;
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();
  file << std::to_string(arr.empty() ? 1 : 0) << " ";
  file << std::to_string(arr.size()) << " ";
  file.saveFile();
}

// 从单个元素构造
TEST_F(ArrayRefTest, SingleElement) {
  int64_t val = 42;
  c10::ArrayRef<int64_t> arr(val);
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << std::to_string(arr.size()) << " ";
  file << std::to_string(arr[0]) << " ";
  file << std::to_string(arr.front()) << " ";
  file << std::to_string(arr.back()) << " ";
  file.saveFile();
}

// 从 pointer + length 构造
TEST_F(ArrayRefTest, PointerLength) {
  int64_t data[] = {10, 20, 30, 40, 50};
  c10::ArrayRef<int64_t> arr(data, 5);
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << std::to_string(arr.size()) << " ";
  for (size_t i = 0; i < arr.size(); ++i) {
    file << std::to_string(arr[i]) << " ";
  }
  file.saveFile();
}

// 从 begin/end 指针构造
TEST_F(ArrayRefTest, BeginEnd) {
  int64_t data[] = {1, 2, 3};
  c10::ArrayRef<int64_t> arr(data, data + 3);
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << std::to_string(arr.size()) << " ";
  for (auto it = arr.begin(); it != arr.end(); ++it) {
    file << std::to_string(*it) << " ";
  }
  file.saveFile();
}

// 从 std::vector 构造
TEST_F(ArrayRefTest, FromVector) {
  std::vector<int64_t> vec = {100, 200, 300, 400};
  c10::ArrayRef<int64_t> arr(vec);
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << std::to_string(arr.size()) << " ";
  file << std::to_string(arr.front()) << " ";
  file << std::to_string(arr.back()) << " ";
  file.saveFile();
}

// 从 std::array 构造
TEST_F(ArrayRefTest, FromStdArray) {
  std::array<int64_t, 3> arr_data = {7, 8, 9};
  c10::ArrayRef<int64_t> arr(arr_data);
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << std::to_string(arr.size()) << " ";
  for (size_t i = 0; i < arr.size(); ++i) {
    file << std::to_string(arr[i]) << " ";
  }
  file.saveFile();
}

// 从 C 数组构造
TEST_F(ArrayRefTest, FromCArray) {
  int64_t data[] = {11, 22, 33, 44};
  c10::ArrayRef<int64_t> arr(data);
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << std::to_string(arr.size()) << " ";
  for (size_t i = 0; i < arr.size(); ++i) {
    file << std::to_string(arr[i]) << " ";
  }
  file.saveFile();
}

// 从 initializer_list 构造
TEST_F(ArrayRefTest, FromInitializerList) {
  c10::ArrayRef<int64_t> arr({5, 10, 15});
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << std::to_string(arr.size()) << " ";
  for (size_t i = 0; i < arr.size(); ++i) {
    file << std::to_string(arr[i]) << " ";
  }
  file.saveFile();
}

// slice 方法
TEST_F(ArrayRefTest, Slice) {
  std::vector<int64_t> vec = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  c10::ArrayRef<int64_t> arr(vec);

  auto sliced1 = arr.slice(2, 3);  // [2, 3, 4]
  auto sliced2 = arr.slice(5);     // [5, 6, 7, 8, 9]

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << std::to_string(sliced1.size()) << " ";
  for (size_t i = 0; i < sliced1.size(); ++i) {
    file << std::to_string(sliced1[i]) << " ";
  }
  file << std::to_string(sliced2.size()) << " ";
  for (size_t i = 0; i < sliced2.size(); ++i) {
    file << std::to_string(sliced2[i]) << " ";
  }
  file.saveFile();
}

// equals 方法
TEST_F(ArrayRefTest, Equals) {
  std::vector<int64_t> vec1 = {1, 2, 3};
  std::vector<int64_t> vec2 = {1, 2, 3};
  std::vector<int64_t> vec3 = {1, 2, 4};

  c10::ArrayRef<int64_t> arr1(vec1);
  c10::ArrayRef<int64_t> arr2(vec2);
  c10::ArrayRef<int64_t> arr3(vec3);

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << std::to_string(arr1.equals(arr2) ? 1 : 0) << " ";
  file << std::to_string(arr1.equals(arr3) ? 1 : 0) << " ";
  // 运算符重载
  file << std::to_string(arr1 == arr2 ? 1 : 0) << " ";
  file << std::to_string(arr1 != arr3 ? 1 : 0) << " ";
  file.saveFile();
}

// at() 方法
TEST_F(ArrayRefTest, At) {
  std::vector<int64_t> vec = {10, 20, 30};
  c10::ArrayRef<int64_t> arr(vec);

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << std::to_string(arr.at(0)) << " ";
  file << std::to_string(arr.at(1)) << " ";
  file << std::to_string(arr.at(2)) << " ";
  file.saveFile();
}

// vec() 方法
TEST_F(ArrayRefTest, Vec) {
  int64_t data[] = {5, 10, 15};
  c10::ArrayRef<int64_t> arr(data);
  auto vec = arr.vec();

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << std::to_string(vec.size()) << " ";
  for (const auto& v : vec) {
    file << std::to_string(v) << " ";
  }
  file.saveFile();
}

// reverse_iterator
TEST_F(ArrayRefTest, ReverseIterator) {
  std::vector<int64_t> vec = {1, 2, 3, 4, 5};
  c10::ArrayRef<int64_t> arr(vec);

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  for (auto it = arr.rbegin(); it != arr.rend(); ++it) {
    file << std::to_string(*it) << " ";
  }
  file.saveFile();
}

// float 类型 ArrayRef
TEST_F(ArrayRefTest, FloatArrayRef) {
  std::vector<float> vec = {1.1f, 2.2f, 3.3f};
  c10::ArrayRef<float> arr(vec);

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << std::to_string(arr.size()) << " ";
  for (size_t i = 0; i < arr.size(); ++i) {
    file << std::to_string(arr[i]) << " ";
  }
  file.saveFile();
}

// IntArrayRef 别名
TEST_F(ArrayRefTest, IntArrayRef) {
  std::vector<int64_t> vec = {3, 4, 5};
  c10::IntArrayRef arr(vec);

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << std::to_string(arr.size()) << " ";
  for (size_t i = 0; i < arr.size(); ++i) {
    file << std::to_string(arr[i]) << " ";
  }
  file.saveFile();
}

// vector 和 ArrayRef 的比较运算符
TEST_F(ArrayRefTest, VectorArrayRefComparison) {
  std::vector<int64_t> vec = {1, 2, 3};
  c10::ArrayRef<int64_t> arr({1, 2, 3});

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << std::to_string(vec == arr ? 1 : 0) << " ";
  file << std::to_string(arr == vec ? 1 : 0) << " ";
  file.saveFile();
}

}  // namespace test
}  // namespace at
