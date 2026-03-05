#include <ATen/ATen.h>
#include <c10/util/Exception.h>
#include <gtest/gtest.h>

#include <string>

#include "../src/file_manager.h"

extern paddle_api_test::ThreadSafeParam g_custom_param;

namespace at {
namespace test {

using paddle_api_test::FileManerger;
using paddle_api_test::ThreadSafeParam;

class ExceptionTest : public ::testing::Test {
 protected:
  void SetUp() override {}
};

// TORCH_CHECK 成功（条件为 true）
TEST_F(ExceptionTest, TorchCheckSuccess) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();
  bool passed = true;
  try {
    TORCH_CHECK(1 == 1, "This should not throw");
  } catch (...) {
    passed = false;
  }
  file << std::to_string(passed ? 1 : 0) << " ";
  file.saveFile();
}

// TORCH_CHECK 失败（条件为 false）
TEST_F(ExceptionTest, TorchCheckFailure) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  bool caught = false;
  try {
    TORCH_CHECK(false, "Expected failure");
  } catch (...) {
    caught = true;
  }
  file << std::to_string(caught ? 1 : 0) << " ";
  file.saveFile();
}

// TORCH_INTERNAL_ASSERT 成功
TEST_F(ExceptionTest, TorchInternalAssertSuccess) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  bool passed = true;
  try {
    TORCH_INTERNAL_ASSERT(2 > 1, "Should not throw");
  } catch (...) {
    passed = false;
  }
  file << std::to_string(passed ? 1 : 0) << " ";
  file.saveFile();
}

// TORCH_INTERNAL_ASSERT 失败
TEST_F(ExceptionTest, TorchInternalAssertFailure) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  bool caught = false;
  try {
    TORCH_INTERNAL_ASSERT(false, "Expected failure");
  } catch (...) {
    caught = true;
  }
  file << std::to_string(caught ? 1 : 0) << " ";
  file.saveFile();
}

// TORCH_CHECK_EQ 成功
TEST_F(ExceptionTest, TorchCheckEqSuccess) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  bool passed = true;
  try {
    TORCH_CHECK_EQ(3, 3);
  } catch (...) {
    passed = false;
  }
  file << std::to_string(passed ? 1 : 0) << " ";
  file.saveFile();
}

// TORCH_CHECK_EQ 失败
// LibTorch: 失败时调用 abort()，使用 EXPECT_DEATH 捕获进程终止。
// Paddle:   失败时抛出 C++ 异常，使用 try-catch 捕获。
TEST_F(ExceptionTest, TorchCheckEqFailure) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
#if USE_PADDLE_API
  bool caught = false;
  try {
    TORCH_CHECK_EQ(3, 4);
  } catch (...) {
    caught = true;
  }
  file << std::to_string(caught ? 1 : 0) << " ";
#else
  EXPECT_DEATH({ TORCH_CHECK_EQ(3, 4); }, ".*");
  file << "1 ";
#endif
  file.saveFile();
}

// TORCH_CHECK_NE
// LibTorch: 失败时调用 abort()，使用 EXPECT_DEATH 捕获进程终止。
// Paddle:   失败时抛出 C++ 异常，使用 try-catch 捕获。
TEST_F(ExceptionTest, TorchCheckNe) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  bool passed = true;
  try {
    TORCH_CHECK_NE(3, 4);
  } catch (...) {
    passed = false;
  }
  file << std::to_string(passed ? 1 : 0) << " ";

#if USE_PADDLE_API
  bool caught = false;
  try {
    TORCH_CHECK_NE(3, 3);
  } catch (...) {
    caught = true;
  }
  file << std::to_string(caught ? 1 : 0) << " ";
#else
  EXPECT_DEATH({ TORCH_CHECK_NE(3, 3); }, ".*");
  file << "1 ";
#endif
  file.saveFile();
}

// TORCH_CHECK_LT / LE / GT / GE
TEST_F(ExceptionTest, TorchCheckComparisons) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();

  bool lt_pass = true;
  try {
    TORCH_CHECK_LT(3, 4);
  } catch (...) {
    lt_pass = false;
  }
  file << std::to_string(lt_pass ? 1 : 0) << " ";

  bool le_pass = true;
  try {
    TORCH_CHECK_LE(3, 3);
  } catch (...) {
    le_pass = false;
  }
  file << std::to_string(le_pass ? 1 : 0) << " ";

  bool gt_pass = true;
  try {
    TORCH_CHECK_GT(4, 3);
  } catch (...) {
    gt_pass = false;
  }
  file << std::to_string(gt_pass ? 1 : 0) << " ";

  bool ge_pass = true;
  try {
    TORCH_CHECK_GE(3, 3);
  } catch (...) {
    ge_pass = false;
  }
  file << std::to_string(ge_pass ? 1 : 0) << " ";
  file.saveFile();
}

// C10_THROW_ERROR
TEST_F(ExceptionTest, C10ThrowError) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();

  bool caught_not_impl = false;
  try {
    C10_THROW_ERROR(NotImplementedError, "not implemented");
  } catch (...) {
    caught_not_impl = true;
  }
  file << std::to_string(caught_not_impl ? 1 : 0) << " ";

  bool caught_error = false;
  try {
    C10_THROW_ERROR(Error, "generic error");
  } catch (...) {
    caught_error = true;
  }
  file << std::to_string(caught_error ? 1 : 0) << " ";
  file.saveFile();
}

}  // namespace test
}  // namespace at
