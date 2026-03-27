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

template <typename T>
static bool arrayref_begin_api_probe(c10::ArrayRef<T> arr) {
  return arr.begin() == arr.data();
}

template <typename T>
static size_t arrayref_end_api_probe(c10::ArrayRef<T> arr) {
  return static_cast<size_t>(arr.end() - arr.begin());
}

template <typename T>
static bool arrayref_cbegin_api_probe(const c10::ArrayRef<T>& arr) {
  return arr.cbegin() == arr.data();
}

template <typename T>
static size_t arrayref_cend_api_probe(const c10::ArrayRef<T>& arr) {
  return static_cast<size_t>(arr.cend() - arr.cbegin());
}

template <typename T, typename Predicate>
static bool arrayref_allmatch_api_probe(const c10::ArrayRef<T>& arr,
                                        Predicate pred) {
  return arr.allMatch(pred);
}

template <typename T>
static bool arrayref_empty_api_probe(const c10::ArrayRef<T>& arr) {
  return arr.empty();
}

// 默认构造（空）
TEST_F(ArrayRefTest, DefaultConstruction) {
  c10::ArrayRef<int64_t> arr;
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();
  file << "DefaultConstruction ";
  file << std::to_string(arr.empty() ? 1 : 0) << " ";
  file << std::to_string(arr.size()) << " ";
  file << "\n";
  file.saveFile();
}

// 从单个元素构造
TEST_F(ArrayRefTest, SingleElement) {
  int64_t val = 42;
  c10::ArrayRef<int64_t> arr(val);
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "SingleElement ";
  file << std::to_string(arr.size()) << " ";
  file << std::to_string(arr[0]) << " ";
  file << std::to_string(arr.front()) << " ";
  file << std::to_string(arr.back()) << " ";
  file << "\n";
  file.saveFile();
}

// 从 pointer + length 构造
TEST_F(ArrayRefTest, PointerLength) {
  int64_t data[] = {10, 20, 30, 40, 50};
  c10::ArrayRef<int64_t> arr(data, 5);
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "PointerLength ";
  file << std::to_string(arr.size()) << " ";
  for (size_t i = 0; i < arr.size(); ++i) {
    file << std::to_string(arr[i]) << " ";
  }
  file << "\n";
  file.saveFile();
}

// 从 begin/end 指针构造
TEST_F(ArrayRefTest, BeginEnd) {
  int64_t data[] = {1, 2, 3};
  c10::ArrayRef<int64_t> arr(data, data + 3);
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "BeginEnd ";
  file << std::to_string(arr.size()) << " ";
  for (auto it = arr.begin(); it != arr.end(); ++it) {
    file << std::to_string(*it) << " ";
  }
  file << "\n";
  file.saveFile();
}

// begin()/end() 方法
TEST_F(ArrayRefTest, IteratorMethods) {
  std::vector<int64_t> vec = {2, 4, 6, 8};
  c10::ArrayRef<int64_t> arr(vec);

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "IteratorMethods ";
  file << std::to_string(arrayref_begin_api_probe(arr) ? 1 : 0) << " ";
  file << std::to_string(arrayref_end_api_probe(arr)) << " ";
  for (auto it = arr.begin(); it != arr.end(); ++it) {
    file << std::to_string(*it) << " ";
  }
  file << "\n";
  file.saveFile();
}

// cbegin()/cend() 方法
TEST_F(ArrayRefTest, ConstIteratorMethods) {
  const std::vector<int64_t> vec = {3, 6, 9};
  const c10::ArrayRef<int64_t> arr(vec);

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "ConstIteratorMethods ";
  file << std::to_string(arrayref_cbegin_api_probe(arr) ? 1 : 0) << " ";
  file << std::to_string(arrayref_cend_api_probe(arr)) << " ";
  for (auto it = arr.cbegin(); it != arr.cend(); ++it) {
    file << std::to_string(*it) << " ";
  }
  file << "\n";
  file.saveFile();
}

// allMatch()/empty() 方法
TEST_F(ArrayRefTest, AllMatchAndEmpty) {
  c10::ArrayRef<int64_t> empty_arr;
  std::vector<int64_t> even_vec = {2, 4, 6, 8};
  std::vector<int64_t> mixed_vec = {2, -1, 6, 8};
  c10::ArrayRef<int64_t> even_arr(even_vec);
  c10::ArrayRef<int64_t> mixed_arr(mixed_vec);

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "AllMatchAndEmpty ";
  file << std::to_string(arrayref_empty_api_probe(empty_arr) ? 1 : 0) << " ";
  file << std::to_string(arrayref_empty_api_probe(even_arr) ? 1 : 0) << " ";
  file << std::to_string(
              arrayref_allmatch_api_probe(
                  empty_arr, [](const int64_t& value) { return value >= 0; })
                  ? 1
                  : 0)
       << " ";
  file << std::to_string(
              arrayref_allmatch_api_probe(
                  even_arr, [](const int64_t& value) { return value % 2 == 0; })
                  ? 1
                  : 0)
       << " ";
  file << std::to_string(
              arrayref_allmatch_api_probe(
                  mixed_arr, [](const int64_t& value) { return value >= 0; })
                  ? 1
                  : 0)
       << " ";
  file << "\n";
  file.saveFile();
}

// 从 std::vector 构造
TEST_F(ArrayRefTest, FromVector) {
  std::vector<int64_t> vec = {100, 200, 300, 400};
  c10::ArrayRef<int64_t> arr(vec);
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "FromVector ";
  file << std::to_string(arr.size()) << " ";
  file << std::to_string(arr.front()) << " ";
  file << std::to_string(arr.back()) << " ";
  file << "\n";
  file.saveFile();
}

// 从 std::array 构造
TEST_F(ArrayRefTest, FromStdArray) {
  std::array<int64_t, 3> arr_data = {7, 8, 9};
  c10::ArrayRef<int64_t> arr(arr_data);
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "FromStdArray ";
  file << std::to_string(arr.size()) << " ";
  for (size_t i = 0; i < arr.size(); ++i) {
    file << std::to_string(arr[i]) << " ";
  }
  file << "\n";
  file.saveFile();
}

// 从 C 数组构造
TEST_F(ArrayRefTest, FromCArray) {
  int64_t data[] = {11, 22, 33, 44};
  c10::ArrayRef<int64_t> arr(data);
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "FromCArray ";
  file << std::to_string(arr.size()) << " ";
  for (size_t i = 0; i < arr.size(); ++i) {
    file << std::to_string(arr[i]) << " ";
  }
  file << "\n";
  file.saveFile();
}

// 从 initializer_list 构造
TEST_F(ArrayRefTest, FromInitializerList) {
  c10::ArrayRef<int64_t> arr({5, 10, 15});
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "FromInitializerList ";
  file << std::to_string(arr.size()) << " ";
  for (size_t i = 0; i < arr.size(); ++i) {
    file << std::to_string(arr[i]) << " ";
  }
  file << "\n";
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
  file << "Slice ";
  file << std::to_string(sliced1.size()) << " ";
  for (size_t i = 0; i < sliced1.size(); ++i) {
    file << std::to_string(sliced1[i]) << " ";
  }
  file << std::to_string(sliced2.size()) << " ";
  for (size_t i = 0; i < sliced2.size(); ++i) {
    file << std::to_string(sliced2[i]) << " ";
  }
  file << "\n";
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
  file << "Equals ";
  file << std::to_string(arr1.equals(arr2) ? 1 : 0) << " ";
  file << std::to_string(arr1.equals(arr3) ? 1 : 0) << " ";
  // 运算符重载
  file << std::to_string(arr1 == arr2 ? 1 : 0) << " ";
  file << std::to_string(arr1 != arr3 ? 1 : 0) << " ";
  file << "\n";
  file.saveFile();
}

// at() 方法
TEST_F(ArrayRefTest, At) {
  std::vector<int64_t> vec = {10, 20, 30};
  c10::ArrayRef<int64_t> arr(vec);

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "At ";
  file << std::to_string(arr.at(0)) << " ";
  file << std::to_string(arr.at(1)) << " ";
  file << std::to_string(arr.at(2)) << " ";
  file << "\n";
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
  file << "Vec ";
  file << std::to_string(vec.size()) << " ";
  for (const auto& v : vec) {
    file << std::to_string(v) << " ";
  }
  file << "\n";
  file.saveFile();
}

// reverse_iterator
TEST_F(ArrayRefTest, ReverseIterator) {
  std::vector<int64_t> vec = {1, 2, 3, 4, 5};
  c10::ArrayRef<int64_t> arr(vec);

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "ReverseIterator ";
  for (auto it = arr.rbegin(); it != arr.rend(); ++it) {
    file << std::to_string(*it) << " ";
  }
  file << "\n";
  file.saveFile();
}

// float 类型 ArrayRef
TEST_F(ArrayRefTest, FloatArrayRef) {
  std::vector<float> vec = {1.1f, 2.2f, 3.3f};
  c10::ArrayRef<float> arr(vec);

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "FloatArrayRef ";
  file << std::to_string(arr.size()) << " ";
  for (size_t i = 0; i < arr.size(); ++i) {
    file << std::to_string(arr[i]) << " ";
  }
  file << "\n";
  file.saveFile();
}

// IntArrayRef 别名
TEST_F(ArrayRefTest, IntArrayRef) {
  std::vector<int64_t> vec = {3, 4, 5};
  c10::IntArrayRef arr(vec);

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "IntArrayRef ";
  file << std::to_string(arr.size()) << " ";
  for (size_t i = 0; i < arr.size(); ++i) {
    file << std::to_string(arr[i]) << " ";
  }
  file << "\n";
  file.saveFile();
}

// vector 和 ArrayRef 的比较运算符
TEST_F(ArrayRefTest, VectorArrayRefComparison) {
  std::vector<int64_t> vec = {1, 2, 3};
  c10::ArrayRef<int64_t> arr({1, 2, 3});

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "VectorArrayRefComparison ";
  file << std::to_string(vec == arr ? 1 : 0) << " ";
  file << std::to_string(arr == vec ? 1 : 0) << " ";
  file << "\n";
  file.saveFile();
}

}  // namespace test
}  // namespace at
