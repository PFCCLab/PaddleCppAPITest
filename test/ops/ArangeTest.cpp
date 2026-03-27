#include <ATen/ATen.h>
#include <ATen/core/Tensor.h>
#include <ATen/ops/arange.h>
#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <string>
#include <vector>

#include "../../src/file_manager.h"

extern paddle_api_test::ThreadSafeParam g_custom_param;

namespace at {
namespace test {

using paddle_api_test::FileManerger;
using paddle_api_test::ThreadSafeParam;

// 按 dtype 分发写出 tensor 内容
static void write_arange_result_to_file(FileManerger* file,
                                        const at::Tensor& result) {
  *file << std::to_string(result.dim()) << " ";
  *file << std::to_string(result.numel()) << " ";
  for (int64_t i = 0; i < result.dim(); ++i) {
    *file << std::to_string(result.sizes()[i]) << " ";
  }
  switch (result.scalar_type()) {
    case at::kFloat: {
      float* data = result.data_ptr<float>();
      // 只输出前10个和后10个元素（避免大 tensor 输出过多）
      int64_t print_count = std::min(result.numel(), static_cast<int64_t>(10));
      for (int64_t i = 0; i < print_count; ++i) {
        *file << std::to_string(data[i]) << " ";
      }
      if (result.numel() > 10) {
        *file << "... ";
        for (int64_t i = result.numel() - 10; i < result.numel(); ++i) {
          *file << std::to_string(data[i]) << " ";
        }
      }
      break;
    }
    case at::kDouble: {
      double* data = result.data_ptr<double>();
      int64_t print_count = std::min(result.numel(), static_cast<int64_t>(10));
      for (int64_t i = 0; i < print_count; ++i) {
        *file << std::to_string(data[i]) << " ";
      }
      if (result.numel() > 10) {
        *file << "... ";
        for (int64_t i = result.numel() - 10; i < result.numel(); ++i) {
          *file << std::to_string(data[i]) << " ";
        }
      }
      break;
    }
    case at::kLong: {
      int64_t* data = result.data_ptr<int64_t>();
      int64_t print_count = std::min(result.numel(), static_cast<int64_t>(10));
      for (int64_t i = 0; i < print_count; ++i) {
        *file << std::to_string(data[i]) << " ";
      }
      if (result.numel() > 10) {
        *file << "... ";
        for (int64_t i = result.numel() - 10; i < result.numel(); ++i) {
          *file << std::to_string(data[i]) << " ";
        }
      }
      break;
    }
    case at::kInt: {
      int32_t* data = result.data_ptr<int32_t>();
      int64_t print_count = std::min(result.numel(), static_cast<int64_t>(10));
      for (int64_t i = 0; i < print_count; ++i) {
        *file << std::to_string(data[i]) << " ";
      }
      if (result.numel() > 10) {
        *file << "... ";
        for (int64_t i = result.numel() - 10; i < result.numel(); ++i) {
          *file << std::to_string(data[i]) << " ";
        }
      }
      break;
    }
    default: {
      *file << "unsupported_dtype ";
      break;
    }
  }
}

class ArangeTest : public ::testing::Test {
 protected:
  void SetUp() override {}
};

// ========== 基础功能 ==========

TEST_F(ArangeTest, BasicArangeWithEnd) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();
  file << "BasicArangeWithEnd ";
  at::Tensor result = at::arange(5, at::TensorOptions().dtype(at::kLong));
  write_arange_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

TEST_F(ArangeTest, ArangeWithStartEnd) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "ArangeWithStartEnd ";
  at::Tensor result = at::arange(2, 7, at::TensorOptions().dtype(at::kLong));
  write_arange_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

TEST_F(ArangeTest, ArangeWithStartEndStep) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "ArangeWithStartEndStep ";
  at::Tensor result =
      at::arange(1, 10, 2, at::TensorOptions().dtype(at::kLong));
  write_arange_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// ========== Shape 覆盖 ==========

// 大 shape {10000}
TEST_F(ArangeTest, LargeShape) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "LargeShape ";
  at::Tensor result = at::arange(10000, at::TensorOptions().dtype(at::kLong));
  write_arange_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// 空 arange (start == end)
TEST_F(ArangeTest, EmptyArange) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "EmptyArange ";
  at::Tensor result = at::arange(5, 5, at::TensorOptions().dtype(at::kLong));
  write_arange_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// 单元素 arange
TEST_F(ArangeTest, SingleElement) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "SingleElement ";
  at::Tensor result = at::arange(0, 1, at::TensorOptions().dtype(at::kLong));
  write_arange_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// ========== Dtype 覆盖 ==========

TEST_F(ArangeTest, Float32Dtype) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "Float32Dtype ";
  at::Tensor result = at::arange(4, at::TensorOptions().dtype(at::kFloat));
  file << std::to_string(result.dim()) << " ";
  file << std::to_string(result.numel()) << " ";
  file << std::to_string(static_cast<int>(result.scalar_type())) << " ";
  write_arange_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

TEST_F(ArangeTest, Float64Dtype) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "Float64Dtype ";
  at::Tensor result = at::arange(0, 5, at::TensorOptions().dtype(at::kDouble));
  file << std::to_string(static_cast<int>(result.scalar_type())) << " ";
  write_arange_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

TEST_F(ArangeTest, Int32Dtype) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "Int32Dtype ";
  at::Tensor result = at::arange(6, at::TensorOptions().dtype(at::kInt));
  file << std::to_string(static_cast<int>(result.scalar_type())) << " ";
  write_arange_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// ========== 值域覆盖 ==========

// 负值范围
TEST_F(ArangeTest, NegativeValues) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "NegativeValues ";
  at::Tensor result = at::arange(-3, 3, at::TensorOptions().dtype(at::kLong));
  write_arange_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// 负步长
TEST_F(ArangeTest, NegativeStep) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "NegativeStep ";
  at::Tensor result =
      at::arange(10, 0, -1, at::TensorOptions().dtype(at::kLong));
  write_arange_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// 浮点步长
TEST_F(ArangeTest, FloatStep) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "FloatStep ";
  at::Tensor result =
      at::arange(0.0, 1.0, 0.1, at::TensorOptions().dtype(at::kFloat));
  write_arange_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// 负浮点步长
TEST_F(ArangeTest, NegativeFloatStep) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "NegativeFloatStep ";
  at::Tensor result =
      at::arange(1.0, 0.0, -0.2, at::TensorOptions().dtype(at::kFloat));
  write_arange_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// 大步长
TEST_F(ArangeTest, LargeStep) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "LargeStep ";
  at::Tensor result =
      at::arange(0, 100, 25, at::TensorOptions().dtype(at::kLong));
  write_arange_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// ========== 异常行为 ==========

// 无效步长 (step=0 会抛异常或产生空 tensor)
TEST_F(ArangeTest, ZeroStep) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "ZeroStep ";
  try {
    at::Tensor result =
        at::arange(0, 5, 0, at::TensorOptions().dtype(at::kLong));
    write_arange_result_to_file(&file, result);
  } catch (const std::exception&) {
    file << "exception ";
  }
  file << "\n";
  file.saveFile();
}

// 方向不一致 (start < end 但 step < 0)
TEST_F(ArangeTest, WrongDirection) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "WrongDirection ";
  try {
    at::Tensor result =
        at::arange(0, 5, -1, at::TensorOptions().dtype(at::kLong));
    write_arange_result_to_file(&file, result);
  } catch (const std::exception&) {
    file << "exception ";
  }
  file << "\n";
  file.saveFile();
}

}  // namespace test
}  // namespace at
