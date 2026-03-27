#include <ATen/ATen.h>
#include <ATen/core/Tensor.h>
#include <c10/util/Exception.h>
#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "../../src/file_manager.h"

extern paddle_api_test::ThreadSafeParam g_custom_param;

namespace at {
namespace test {

using paddle_api_test::FileManerger;

class SplitTest : public ::testing::Test {};

static void write_tensor_to_file(FileManerger* file, const at::Tensor& result) {
  at::Tensor contiguous = result.contiguous();
  *file << std::to_string(result.dim()) << " ";
  *file << std::to_string(result.numel()) << " ";
  *file << std::to_string(static_cast<int>(result.scalar_type())) << " ";
  for (int64_t i = 0; i < result.dim(); ++i) {
    *file << std::to_string(result.size(i)) << " ";
  }

  if (result.scalar_type() == at::kFloat) {
    float* data = contiguous.data_ptr<float>();
    for (int64_t i = 0; i < contiguous.numel(); ++i) {
      *file << std::to_string(data[i]) << " ";
    }
  } else if (result.scalar_type() == at::kDouble) {
    double* data = contiguous.data_ptr<double>();
    for (int64_t i = 0; i < contiguous.numel(); ++i) {
      *file << std::to_string(data[i]) << " ";
    }
  } else if (result.scalar_type() == at::kInt) {
    int* data = contiguous.data_ptr<int>();
    for (int64_t i = 0; i < contiguous.numel(); ++i) {
      *file << std::to_string(data[i]) << " ";
    }
  } else if (result.scalar_type() == at::kLong) {
    int64_t* data = contiguous.data_ptr<int64_t>();
    for (int64_t i = 0; i < contiguous.numel(); ++i) {
      *file << std::to_string(data[i]) << " ";
    }
  }
}

static void write_tensor_list_to_file(FileManerger* file,
                                      const std::vector<at::Tensor>& results) {
  *file << std::to_string(results.size()) << " ";
  for (const auto& result : results) {
    write_tensor_to_file(file, result);
  }
}

template <typename Func>
static void write_tensor_list_or_exception(FileManerger* file, Func&& func) {
  try {
    std::vector<at::Tensor> results = func();
    *file << "0 ";
    write_tensor_list_to_file(file, results);
  } catch (const std::exception&) {
    *file << "1 ";
  }
}

TEST_F(SplitTest, SplitAndTensorSplitFamilies) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();
  file << "SplitAndTensorSplitFamilies ";

  at::Tensor base =
      at::arange(24, at::TensorOptions().dtype(at::kFloat)).reshape({2, 3, 4});

  write_tensor_list_or_exception(&file, [&]() { return base.split(2, 1); });
  write_tensor_list_or_exception(&file,
                                 [&]() { return base.unsafe_split(2, 2); });
  write_tensor_list_or_exception(&file, [&]() {
    return base.split_with_sizes({1, 2}, 1);
  });
  write_tensor_list_or_exception(&file, [&]() {
    return base.unsafe_split_with_sizes({1, 3}, 2);
  });
  write_tensor_list_or_exception(&file,
                                 [&]() { return base.tensor_split(3, 2); });

  at::Tensor matrix =
      at::arange(12, at::TensorOptions().dtype(at::kFloat)).reshape({3, 4});
  write_tensor_list_or_exception(&file, [&]() { return matrix.hsplit(2); });
  write_tensor_list_or_exception(&file, [&]() { return matrix.vsplit(3); });

  at::Tensor cube =
      at::arange(24, at::TensorOptions().dtype(at::kFloat)).reshape({2, 3, 4});
  write_tensor_list_or_exception(&file, [&]() { return cube.dsplit(2); });

  file << "\n";
  file.saveFile();
}

TEST_F(SplitTest, SymintSectionAndChunkSize) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "SymintSectionAndChunkSize ";

  at::Tensor float_tensor =
      at::arange(18, at::TensorOptions().dtype(at::kFloat)).reshape({3, 6});
  c10::SymInt tensor_sections(4);
  c10::SymInt split_size(2);
  c10::SymInt unsafe_split_size(5);

  write_tensor_list_or_exception(&file, [&]() {
    return float_tensor.tensor_split_symint(tensor_sections, 1);
  });
  write_tensor_list_or_exception(
      &file, [&]() { return float_tensor.split_symint(split_size, 1); });

  at::Tensor large_long_tensor =
      at::arange(128, at::TensorOptions().dtype(at::kLong));
  write_tensor_list_or_exception(&file, [&]() {
    return large_long_tensor.unsafe_split_symint(unsafe_split_size, 0);
  });

  file << "\n";
  file.saveFile();
}

TEST_F(SplitTest, SymintIndicesAndSplitSizes) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "SymintIndicesAndSplitSizes ";

  at::Tensor non_contiguous =
      at::arange(12, at::TensorOptions().dtype(at::kDouble))
          .reshape({3, 4})
          .transpose(0, 1);
  std::vector<c10::SymInt> tensor_split_indices = {c10::SymInt(1),
                                                   c10::SymInt(3)};
  std::vector<c10::SymInt> split_sizes = {c10::SymInt(2), c10::SymInt(2)};
  std::vector<c10::SymInt> sized_splits = {c10::SymInt(1), c10::SymInt(3)};
  std::vector<c10::SymInt> unsafe_sized_splits = {
      c10::SymInt(1), c10::SymInt(1), c10::SymInt(2)};

  write_tensor_list_or_exception(&file, [&]() {
    return non_contiguous.tensor_split_symint(
        c10::SymIntArrayRef(tensor_split_indices), 0);
  });
  write_tensor_list_or_exception(&file, [&]() {
    return non_contiguous.split_symint(c10::SymIntArrayRef(split_sizes), 0);
  });
  write_tensor_list_or_exception(&file, [&]() {
    return non_contiguous.split_with_sizes_symint(
        c10::SymIntArrayRef(sized_splits), 0);
  });
  write_tensor_list_or_exception(&file, [&]() {
    return non_contiguous.unsafe_split_with_sizes_symint(
        c10::SymIntArrayRef(unsafe_sized_splits), 0);
  });

  file << "\n";
  file.saveFile();
}

TEST_F(SplitTest, BoundaryShapesAndDtypes) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "BoundaryShapesAndDtypes ";

  at::Tensor scalar =
      at::ones({}, at::TensorOptions().dtype(at::kDouble)).fill_(7.0);
  c10::SymInt scalar_sections(1);
  write_tensor_list_or_exception(
      &file, [&]() { return scalar.tensor_split_symint(scalar_sections, 0); });

  at::Tensor empty_long = at::empty({0}, at::TensorOptions().dtype(at::kLong));
  std::vector<c10::SymInt> empty_split_sizes = {c10::SymInt(0)};
  write_tensor_list_or_exception(&file, [&]() {
    return empty_long.split_with_sizes_symint(
        c10::SymIntArrayRef(empty_split_sizes), 0);
  });

  at::Tensor ones_int =
      at::ones({1, 1, 1}, at::TensorOptions().dtype(at::kInt));
  c10::SymInt unit_split_size(1);
  write_tensor_list_or_exception(
      &file, [&]() { return ones_int.split_symint(unit_split_size, 2); });

  file << "\n";
  file.saveFile();
}

}  // namespace test
}  // namespace at
