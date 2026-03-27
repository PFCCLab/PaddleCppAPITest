#include <ATen/ATen.h>
#include <gtest/gtest.h>
#include <torch/python.h>

#include <string>
#include <type_traits>
#include <utility>

#include "../src/file_manager.h"

extern paddle_api_test::ThreadSafeParam g_custom_param;

namespace at {
namespace test {

using paddle_api_test::FileManerger;
using paddle_api_test::ThreadSafeParam;

class PythonCompatTest : public ::testing::Test {
 protected:
  void SetUp() override {}
};

// py_object_to_dtype
TEST_F(PythonCompatTest, PyObjectToDtypeTypeCheck) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();
  file << "PyObjectToDtypeTypeCheck ";

  using ResultType = decltype(torch::python::detail::py_object_to_dtype(
      std::declval<py::object>()));
  constexpr bool returns_dtype = std::is_same<ResultType, torch::Dtype>::value;

  file << std::to_string(returns_dtype ? 1 : 0) << " ";
  file << "\n";
  file.saveFile();
}

}  // namespace test
}  // namespace at
