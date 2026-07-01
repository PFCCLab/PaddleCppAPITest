#include <ATen/ATen.h>
#include <ATen/core/Tensor.h>
#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "src/file_manager.h"

#if USE_PADDLE_API
#include "paddle/common/flags.h"
COMMON_DECLARE_bool(use_stride_kernel);
#endif

extern paddle_api_test::ThreadSafeParam g_custom_param;

namespace at {
namespace test {

using paddle_api_test::FileManerger;

class BroadcastToTest : public ::testing::Test {
 protected:
  void SetUp() override {}
};

class UseStrideKernelGuard {
 public:
  explicit UseStrideKernelGuard(bool value) {
#if USE_PADDLE_API
    previous_ = FLAGS_use_stride_kernel;
    FLAGS_use_stride_kernel = value;
#else
    (void)value;
#endif
  }

  ~UseStrideKernelGuard() {
#if USE_PADDLE_API
    FLAGS_use_stride_kernel = previous_;
#endif
  }

 private:
#if USE_PADDLE_API
  bool previous_{true};
#endif
};

// Compute element offset from flat index using strides (strides-aware access)
static inline int64_t compute_offset_from_flat_index(int64_t flat_idx,
                                                     const at::Tensor& tensor) {
  int64_t offset = 0;
  int64_t remainder = flat_idx;
  for (int64_t d = tensor.dim() - 1; d >= 0; --d) {
    int64_t coord = remainder % tensor.sizes()[d];
    remainder /= tensor.sizes()[d];
    offset += coord * tensor.strides()[d];
  }
  return offset;
}

// Write tensor metadata (dim, numel, sizes, strides) and all element values
// Uses strides-aware access to faithfully reflect the underlying layout.
// If Paddle and PyTorch produce different strides, result_cmp will DIFFER,
// and the difference should be recorded as a known mismatch.
static void write_broadcast_to_result_to_file(FileManerger* file,
                                              const at::Tensor& result) {
  *file << std::to_string(result.dim()) << " ";
  *file << std::to_string(result.numel()) << " ";
  for (int64_t i = 0; i < result.dim(); ++i) {
    *file << std::to_string(result.sizes()[i]) << " ";
  }
  // Record strides so layout differences are detected by result_cmp
  for (int64_t i = 0; i < result.dim(); ++i) {
    *file << std::to_string(result.strides()[i]) << " ";
  }
  if (result.numel() == 0) {
    *file << "empty ";
    return;
  }
  switch (result.scalar_type()) {
    case at::kFloat: {
      float* data = result.data_ptr<float>();
      for (int64_t i = 0; i < result.numel(); ++i) {
        int64_t offset = compute_offset_from_flat_index(i, result);
        *file << std::to_string(data[offset]) << " ";
      }
      break;
    }
    case at::kDouble: {
      double* data = result.data_ptr<double>();
      for (int64_t i = 0; i < result.numel(); ++i) {
        int64_t offset = compute_offset_from_flat_index(i, result);
        *file << std::to_string(data[offset]) << " ";
      }
      break;
    }
    case at::kInt: {
      int32_t* data = result.data_ptr<int32_t>();
      for (int64_t i = 0; i < result.numel(); ++i) {
        int64_t offset = compute_offset_from_flat_index(i, result);
        *file << std::to_string(data[offset]) << " ";
      }
      break;
    }
    case at::kLong: {
      int64_t* data = result.data_ptr<int64_t>();
      for (int64_t i = 0; i < result.numel(); ++i) {
        int64_t offset = compute_offset_from_flat_index(i, result);
        *file << std::to_string(data[offset]) << " ";
      }
      break;
    }
    default:
      *file << "unsupported_dtype ";
      break;
  }
}

// Values-only writer for cases where Paddle intentionally materializes a
// broadcast while PyTorch returns a stride-0 view.
static void write_broadcast_values_only_to_file(FileManerger* file,
                                                const at::Tensor& result) {
  *file << std::to_string(result.dim()) << " ";
  *file << std::to_string(result.numel()) << " ";
  for (int64_t i = 0; i < result.dim(); ++i) {
    *file << std::to_string(result.sizes()[i]) << " ";
  }
  if (result.numel() == 0) {
    *file << "empty ";
    return;
  }
  float* data = result.data_ptr<float>();
  for (int64_t i = 0; i < result.numel(); ++i) {
    int64_t offset = compute_offset_from_flat_index(i, result);
    *file << std::to_string(data[offset]) << " ";
  }
}

// ======================== Shape coverage ========================

// Small shape test
TEST_F(BroadcastToTest, BroadcastToSmall) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();
  file << "BroadcastToSmall ";
  at::Tensor small = at::ones({1, 3}, at::kFloat);
  at::Tensor result = small.broadcast_to({2, 3});
  write_broadcast_to_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// Large shape test
TEST_F(BroadcastToTest, BroadcastToLarge) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "BroadcastToLarge ";
  at::Tensor large = at::ones({1, 1, 128}, at::kFloat);
  at::Tensor result = large.broadcast_to({64, 32, 128});
  write_broadcast_to_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// Boundary: empty tensor
TEST_F(BroadcastToTest, BroadcastToBoundaryEmpty) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "BroadcastToBoundaryEmpty ";
  at::Tensor t = at::ones({1, 0}, at::kFloat);
  at::Tensor result = t.broadcast_to({2, 0});
  file << std::to_string(result.dim()) << " ";
  file << std::to_string(result.numel()) << " ";
  file << "empty ";
  file << "\n";
  file.saveFile();
}

// Boundary: rank less (input rank < target rank)
TEST_F(BroadcastToTest, BroadcastToBoundaryRankLess) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "BroadcastToBoundaryRankLess ";
  at::Tensor t = at::ones({1}, at::kFloat);
  at::Tensor result = t.broadcast_to({2, 3});
  write_broadcast_to_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// Boundary: scalar (0-d tensor)
TEST_F(BroadcastToTest, BroadcastToScalar) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "BroadcastToScalar ";
  at::Tensor t = at::full({}, 5.0f, at::kFloat);
  at::Tensor result = t.broadcast_to({2, 3});
  write_broadcast_to_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// ======================== Dtype coverage ========================

TEST_F(BroadcastToTest, BroadcastToDtypeFloat) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "BroadcastToDtypeFloat ";
  at::Tensor t = at::ones({1, 2}, at::kFloat);
  at::Tensor result = t.broadcast_to({3, 2});
  file << std::to_string(static_cast<int>(result.scalar_type())) << " ";
  write_broadcast_to_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

TEST_F(BroadcastToTest, BroadcastToDtypeDouble) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "BroadcastToDtypeDouble ";
  at::Tensor t = at::ones({1, 2}, at::kDouble);
  at::Tensor result = t.broadcast_to({3, 2});
  file << std::to_string(static_cast<int>(result.scalar_type())) << " ";
  write_broadcast_to_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

TEST_F(BroadcastToTest, BroadcastToDtypeInt) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "BroadcastToDtypeInt ";
  at::Tensor t = at::ones({1, 2}, at::kInt);
  at::Tensor result = t.broadcast_to({3, 2});
  file << std::to_string(static_cast<int>(result.scalar_type())) << " ";
  write_broadcast_to_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

TEST_F(BroadcastToTest, BroadcastToDtypeLong) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "BroadcastToDtypeLong ";
  at::Tensor t = at::ones({1, 2}, at::kLong);
  at::Tensor result = t.broadcast_to({3, 2});
  file << std::to_string(static_cast<int>(result.scalar_type())) << " ";
  write_broadcast_to_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// ======================== Exception coverage ========================

TEST_F(BroadcastToTest, BroadcastToInvalidNonSingleton) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "BroadcastToInvalidNonSingleton ";

  try {
    at::Tensor t = at::ones({2, 3}, at::kFloat);
    at::Tensor result = t.broadcast_to({2, 4});
    write_broadcast_to_result_to_file(&file, result);
  } catch (const std::exception&) {
    file << "exception ";
  }

  file << "\n";
  file.saveFile();
}

TEST_F(BroadcastToTest, BroadcastToHighRankToLowRank) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "BroadcastToHighRankToLowRank ";

  try {
    at::Tensor t = at::ones({2, 3, 4}, at::kFloat);
    at::Tensor result = t.broadcast_to({3, 4});
    write_broadcast_to_result_to_file(&file, result);
  } catch (const std::exception&) {
    file << "exception ";
  }

  file << "\n";
  file.saveFile();
}

// C++ ATen broadcast_to follows expand-style -1 keep-dim behavior.
TEST_F(BroadcastToTest, BroadcastToNegativeOne) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "BroadcastToNegativeOne ";

  at::Tensor t = at::ones({3}, at::kFloat);
  at::Tensor result = t.broadcast_to({-1});
  write_broadcast_to_result_to_file(&file, result);

  file << "\n";
  file.saveFile();
}

TEST_F(BroadcastToTest, BroadcastToNegativeLessThanMinusOne) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "BroadcastToNegativeLessThanMinusOne ";

  try {
    at::Tensor t = at::ones({1}, at::kFloat);
    at::Tensor result = t.broadcast_to({-2});
    write_broadcast_to_result_to_file(&file, result);
  } catch (const std::exception&) {
    file << "exception ";
  }

  file << "\n";
  file.saveFile();
}

TEST_F(BroadcastToTest, ExpandNegativeLessThanMinusOne) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "ExpandNegativeLessThanMinusOne ";

  try {
    at::Tensor t = at::ones({1}, at::kFloat);
    at::Tensor result = t.expand({-2});
    write_broadcast_to_result_to_file(&file, result);
  } catch (const std::exception&) {
    file << "exception ";
  }

  file << "\n";
  file.saveFile();
}

TEST_F(BroadcastToTest, BroadcastToStrideZeroValues) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "BroadcastToStrideZeroValues ";

  at::Tensor t = at::zeros({1, 2}, at::kFloat);
  t.data_ptr<float>()[0] = 3.0f;
  t.data_ptr<float>()[1] = 7.0f;
  at::Tensor result = t.broadcast_to({3, 2});
  write_broadcast_to_result_to_file(&file, result);

  file << "\n";
  file.saveFile();
}

TEST_F(BroadcastToTest, BroadcastToStrideKernelDisabledValuesOnly) {
  UseStrideKernelGuard guard(false);
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "BroadcastToStrideKernelDisabledValuesOnly ";

  at::Tensor t = at::zeros({1, 2}, at::kFloat);
  t.data_ptr<float>()[0] = 3.0f;
  t.data_ptr<float>()[1] = 7.0f;
  at::Tensor result = t.broadcast_to({3, 2});
  write_broadcast_values_only_to_file(&file, result);

  file << "\n";
  file.saveFile();
}

TEST_F(BroadcastToTest, ExpandStrideKernelDisabledValuesOnly) {
  UseStrideKernelGuard guard(false);
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "ExpandStrideKernelDisabledValuesOnly ";

  at::Tensor t = at::ones({1}, at::kFloat);
  t.data_ptr<float>()[0] = 5.0f;
  at::Tensor result = t.expand({2, 3});
  write_broadcast_values_only_to_file(&file, result);

  file << "\n";
  file.saveFile();
}

// ======================== Function form ========================

TEST_F(BroadcastToTest, BroadcastToFunction) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "BroadcastToFunction ";
  at::Tensor t = at::ones({1, 2}, at::kFloat);
  at::Tensor result = at::broadcast_to(t, {3, 2});
  write_broadcast_to_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

}  // namespace test
}  // namespace at
