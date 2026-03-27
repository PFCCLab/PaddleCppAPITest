#include <ATen/ATen.h>
#include <ATen/native/RangeUtils.h>
#include <c10/util/Exception.h>
#include <gtest/gtest.h>

#include <string>

#include "../src/file_manager.h"

extern paddle_api_test::ThreadSafeParam g_custom_param;

namespace at {
namespace test {

using paddle_api_test::FileManerger;
using paddle_api_test::ThreadSafeParam;

class RangeUtilsTest : public ::testing::Test {
 protected:
  void SetUp() override {}
};

static void write_bool_result(FileManerger* file, bool value) {
  *file << (value ? "1 " : "0 ");
}

static bool bounds_check_succeeds(const c10::Scalar& start,
                                  const c10::Scalar& end,
                                  const c10::Scalar& step) {
  try {
    at::native::arange_check_bounds(start, end, step);
    return true;
  } catch (const std::exception&) {
    return false;
  }
}

TEST_F(RangeUtilsTest, ArangeCheckBoundsValidInt32) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();
  file << "ArangeCheckBoundsValidInt32 ";

  write_bool_result(&file, bounds_check_succeeds(0, 5, 1));
  file << "\n";
  file.saveFile();
}

TEST_F(RangeUtilsTest, ArangeCheckBoundsValidDoubleNegativeStep) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "ArangeCheckBoundsValidDoubleNegativeStep ";

  write_bool_result(&file, bounds_check_succeeds(5.0, -1.0, -1.5));
  file << "\n";
  file.saveFile();
}

TEST_F(RangeUtilsTest, ArangeCheckBoundsZeroStepThrows) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "ArangeCheckBoundsZeroStepThrows ";

  write_bool_result(&file, !bounds_check_succeeds(0.0f, 5.0f, 0.0f));
  file << "\n";
  file.saveFile();
}

TEST_F(RangeUtilsTest, ArangeCheckBoundsWrongDirectionThrows) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "ArangeCheckBoundsWrongDirectionThrows ";

  write_bool_result(&file,
                    !bounds_check_succeeds(static_cast<int64_t>(0),
                                           static_cast<int64_t>(5),
                                           static_cast<int64_t>(-1)));
  file << "\n";
  file.saveFile();
}

TEST_F(RangeUtilsTest, ComputeArangeSizeInt32) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "ComputeArangeSizeInt32 ";

  file << std::to_string(at::native::compute_arange_size<int>(2, 12, 3)) << " ";
  file << "\n";
  file.saveFile();
}

TEST_F(RangeUtilsTest, ComputeArangeSizeInt64NegativeStep) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "ComputeArangeSizeInt64NegativeStep ";

  file << std::to_string(at::native::compute_arange_size<int64_t>(
              static_cast<int64_t>(5),
              static_cast<int64_t>(-1),
              static_cast<int64_t>(-2)))
       << " ";
  file << "\n";
  file.saveFile();
}

TEST_F(RangeUtilsTest, ComputeArangeSizeFloat) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "ComputeArangeSizeFloat ";

  file << std::to_string(
              at::native::compute_arange_size<float>(0.0f, 1.0f, 0.25f))
       << " ";
  file << "\n";
  file.saveFile();
}

TEST_F(RangeUtilsTest, ComputeArangeSizeDoubleZeroLength) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "ComputeArangeSizeDoubleZeroLength ";

  file << std::to_string(at::native::compute_arange_size<double>(5.0, 5.0, 1.0))
       << " ";
  file << "\n";
  file.saveFile();
}

}  // namespace test
}  // namespace at
