#include <ATen/ATen.h>
#include <ATen/core/Tensor.h>
#include <ATen/ops/as_strided.h>
#include <ATen/ops/full.h>
#include <ATen/ops/take.h>
#include <ATen/ops/tensor.h>
#include <ATen/ops/zeros.h>
#include <c10/util/Exception.h>
#include <c10/util/complex.h>
#include <gtest/gtest.h>

#include <complex>
#include <cstdint>
#include <string>
#include <vector>

#include "src/file_manager.h"

extern paddle_api_test::ThreadSafeParam g_custom_param;

namespace at {
namespace test {

using paddle_api_test::FileManerger;

class TakeTest : public ::testing::Test {
 protected:
  void SetUp() override {}
};

static void write_result_to_file(FileManerger* file, const at::Tensor& result) {
  *file << std::to_string(result.dim()) << " ";
  *file << std::to_string(result.numel()) << " ";
  if (result.scalar_type() == at::kFloat) {
    float* data = result.data_ptr<float>();
    for (int64_t i = 0; i < result.numel(); ++i) {
      *file << std::to_string(data[i]) << " ";
    }
  } else if (result.scalar_type() == at::kDouble) {
    double* data = result.data_ptr<double>();
    for (int64_t i = 0; i < result.numel(); ++i) {
      *file << std::to_string(data[i]) << " ";
    }
  } else if (result.scalar_type() == at::kInt) {
    int* data = result.data_ptr<int>();
    for (int64_t i = 0; i < result.numel(); ++i) {
      *file << std::to_string(data[i]) << " ";
    }
  } else if (result.scalar_type() == at::kLong) {
    int64_t* data = result.data_ptr<int64_t>();
    for (int64_t i = 0; i < result.numel(); ++i) {
      *file << std::to_string(data[i]) << " ";
    }
  } else if (result.scalar_type() == at::kUInt16) {
    uint16_t* data = result.data_ptr<uint16_t>();
    for (int64_t i = 0; i < result.numel(); ++i) {
      *file << std::to_string(data[i]) << " ";
    }
  } else if (result.scalar_type() == at::kUInt32) {
    uint32_t* data = result.data_ptr<uint32_t>();
    for (int64_t i = 0; i < result.numel(); ++i) {
      *file << std::to_string(data[i]) << " ";
    }
  } else if (result.scalar_type() == at::kBool) {
    bool* data = result.data_ptr<bool>();
    for (int64_t i = 0; i < result.numel(); ++i) {
      *file << std::to_string(static_cast<int>(data[i])) << " ";
    }
  } else if (result.scalar_type() == at::kChar) {
    int8_t* data = result.data_ptr<int8_t>();
    for (int64_t i = 0; i < result.numel(); ++i) {
      *file << std::to_string(static_cast<int>(data[i])) << " ";
    }
  } else if (result.scalar_type() == at::kHalf) {
    at::Half* data = result.data_ptr<at::Half>();
    for (int64_t i = 0; i < result.numel(); ++i) {
      *file << std::to_string(static_cast<float>(data[i])) << " ";
    }
  } else if (result.scalar_type() == at::kBFloat16) {
    at::BFloat16* data = result.data_ptr<at::BFloat16>();
    for (int64_t i = 0; i < result.numel(); ++i) {
      *file << std::to_string(static_cast<float>(data[i])) << " ";
    }
  } else if (result.scalar_type() == at::kFloat8_e5m2 ||
             result.scalar_type() == at::kFloat8_e4m3fn) {
    at::Tensor result_float = result.to(at::kFloat);
    float* data = result_float.data_ptr<float>();
    for (int64_t i = 0; i < result_float.numel(); ++i) {
      *file << std::to_string(data[i]) << " ";
    }
  } else if (result.scalar_type() == at::kComplexFloat) {
    c10::complex<float>* data = result.data_ptr<c10::complex<float>>();
    for (int64_t i = 0; i < result.numel(); ++i) {
      auto value = static_cast<std::complex<float>>(data[i]);
      *file << std::to_string(value.real()) << " ";
      *file << std::to_string(value.imag()) << " ";
    }
  } else if (result.scalar_type() == at::kComplexDouble) {
    c10::complex<double>* data = result.data_ptr<c10::complex<double>>();
    for (int64_t i = 0; i < result.numel(); ++i) {
      auto value = static_cast<std::complex<double>>(data[i]);
      *file << std::to_string(value.real()) << " ";
      *file << std::to_string(value.imag()) << " ";
    }
  }
}

static at::Tensor make_test_tensor(at::ScalarType dtype) {
  if (dtype == at::kFloat) {
    auto t = at::zeros({3, 4}, at::kFloat);
    float* data = t.data_ptr<float>();
    for (int i = 0; i < 12; ++i) data[i] = static_cast<float>(i);
    return t;
  } else if (dtype == at::kDouble) {
    auto t = at::zeros({3, 4}, at::kDouble);
    double* data = t.data_ptr<double>();
    for (int i = 0; i < 12; ++i) data[i] = static_cast<double>(i);
    return t;
  } else if (dtype == at::kInt) {
    auto t = at::zeros({3, 4}, at::kInt);
    int* data = t.data_ptr<int>();
    for (int i = 0; i < 12; ++i) data[i] = i;
    return t;
  } else if (dtype == at::kLong) {
    auto t = at::zeros({3, 4}, at::kLong);
    int64_t* data = t.data_ptr<int64_t>();
    for (int i = 0; i < 12; ++i) data[i] = i;
    return t;
  }
  return at::zeros({3, 4}, at::kFloat);
}

static at::Tensor make_index_tensor(const std::vector<int64_t>& values) {
  return at::tensor(at::ArrayRef<int64_t>(values),
                    at::TensorOptions().dtype(at::kLong));
}

static at::Tensor make_float_tensor(const std::vector<float>& values) {
  return at::tensor(at::ArrayRef<float>(values),
                    at::TensorOptions().dtype(at::kFloat));
}

static at::Tensor make_int_index_tensor(const std::vector<int32_t>& values) {
  return at::tensor(at::ArrayRef<int32_t>(values),
                    at::TensorOptions().dtype(at::kInt));
}

// Shape: small 1D index
TEST_F(TakeTest, TakeFloatSmall) {
  at::Tensor t = make_test_tensor(at::kFloat);
  at::Tensor index = make_index_tensor({0, 5, 11});
  at::Tensor result = at::take(t, index);

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();
  file << "TakeFloatSmall ";
  write_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

TEST_F(TakeTest, TakeDoubleSmall) {
  at::Tensor t = make_test_tensor(at::kDouble);
  at::Tensor index = make_index_tensor({0, 5, 11});
  at::Tensor result = at::take(t, index);

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "TakeDoubleSmall ";
  write_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

TEST_F(TakeTest, TakeIntSmall) {
  at::Tensor t = make_test_tensor(at::kInt);
  at::Tensor index = make_index_tensor({0, 5, 11});
  at::Tensor result = at::take(t, index);

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "TakeIntSmall ";
  write_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

TEST_F(TakeTest, TakeLongSmall) {
  at::Tensor t = make_test_tensor(at::kLong);
  at::Tensor index = make_index_tensor({0, 5, 11});
  at::Tensor result = at::take(t, index);

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "TakeLongSmall ";
  write_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

TEST_F(TakeTest, TakeBoolSmall) {
  at::Tensor t = make_index_tensor({0, 1, 0, 1}).to(at::kBool);
  at::Tensor index = make_index_tensor({0, 3, 1});
  at::Tensor result = at::take(t, index);

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "TakeBoolSmall ";
  write_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

TEST_F(TakeTest, TakeCharSmall) {
  std::vector<int8_t> values = {-4, -2, 3, 7};
  at::Tensor t = at::tensor(at::ArrayRef<int8_t>(values),
                            at::TensorOptions().dtype(at::kChar));
  at::Tensor index = make_index_tensor({0, 3, 1});
  at::Tensor result = at::take(t, index);

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "TakeCharSmall ";
  write_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

TEST_F(TakeTest, TakeHalfSmall) {
  at::Tensor t = make_float_tensor({1.5f, -2.0f, 0.5f, 4.0f}).to(at::kHalf);
  at::Tensor index = make_index_tensor({0, 3, 1});
  at::Tensor result = at::take(t, index);

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "TakeHalfSmall ";
  write_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

TEST_F(TakeTest, TakeBFloat16Small) {
  at::Tensor t =
      make_float_tensor({1.25f, -3.5f, 2.0f, 8.0f}).to(at::kBFloat16);
  at::Tensor index = make_index_tensor({0, 3, 1});
  at::Tensor result = at::take(t, index);

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "TakeBFloat16Small ";
  write_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

TEST_F(TakeTest, TakeUInt16Throws) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "TakeUInt16Throws ";
  try {
    at::Tensor t = make_index_tensor({2, 2, 2, 2}).to(at::kUInt16);
    at::Tensor index = make_index_tensor({0, 3, 1});
    at::Tensor result = at::take(t, index);
    write_result_to_file(&file, result);
  } catch (const std::exception&) {
    file << "exception ";
  }
  file << "\n";
  file.saveFile();
}

TEST_F(TakeTest, TakeUInt32Throws) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "TakeUInt32Throws ";
  try {
    at::Tensor t = make_index_tensor({2, 2, 2, 2}).to(at::kUInt32);
    at::Tensor index = make_index_tensor({0, 3, 1});
    at::Tensor result = at::take(t, index);
    write_result_to_file(&file, result);
  } catch (const std::exception&) {
    file << "exception ";
  }
  file << "\n";
  file.saveFile();
}

TEST_F(TakeTest, TakeFloat8E5M2Small) {
  at::Tensor t =
      make_float_tensor({1.0f, 2.0f, 4.0f, 8.0f}).to(at::kFloat8_e5m2);
  at::Tensor index = make_index_tensor({0, 3, 1});

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "TakeFloat8E5M2Small ";
  try {
    at::Tensor result = at::take(t, index);
    write_result_to_file(&file, result);
  } catch (const std::exception&) {
    file << "exception ";
  }
  file << "\n";
  file.saveFile();
}

TEST_F(TakeTest, TakeFloat8E4M3FNSmall) {
  at::Tensor t =
      make_float_tensor({1.0f, 2.0f, 4.0f, 8.0f}).to(at::kFloat8_e4m3fn);
  at::Tensor index = make_index_tensor({0, 3, 1});

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "TakeFloat8E4M3FNSmall ";
  try {
    at::Tensor result = at::take(t, index);
    write_result_to_file(&file, result);
  } catch (const std::exception&) {
    file << "exception ";
  }
  file << "\n";
  file.saveFile();
}

TEST_F(TakeTest, TakeComplexFloatSmall) {
  std::vector<c10::complex<float>> values = {
      {1.0f, 2.0f}, {3.0f, -4.0f}, {-5.0f, 6.0f}, {7.0f, 8.0f}};
  at::Tensor t = at::tensor(at::ArrayRef<c10::complex<float>>(values),
                            at::TensorOptions().dtype(at::kComplexFloat));
  at::Tensor index = make_index_tensor({0, 3, 1});
  at::Tensor result = at::take(t, index);

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "TakeComplexFloatSmall ";
  write_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

TEST_F(TakeTest, TakeComplexDoubleSmall) {
  std::vector<c10::complex<double>> values = {
      {1.0, -2.0}, {-3.0, 4.0}, {5.0, -6.0}, {-7.0, -8.0}};
  at::Tensor t = at::tensor(at::ArrayRef<c10::complex<double>>(values),
                            at::TensorOptions().dtype(at::kComplexDouble));
  at::Tensor index = make_index_tensor({0, 3, 1});
  at::Tensor result = at::take(t, index);

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "TakeComplexDoubleSmall ";
  write_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

TEST_F(TakeTest, TakeFloatNonContiguousInput) {
  at::Tensor base = at::arange(6, at::TensorOptions().dtype(at::kFloat));
  at::Tensor t = base.as_strided({3}, {2});
  at::Tensor index = make_index_tensor({1});
  at::Tensor result = at::take(t, index);

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "TakeFloatNonContiguousInput ";
  write_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

TEST_F(TakeTest, TakeFloatNonContiguousIndex) {
  at::Tensor t = at::arange(6, at::TensorOptions().dtype(at::kFloat));
  at::Tensor index_base = make_index_tensor({0, 1, 2, 3, 4, 5});
  at::Tensor index = index_base.as_strided({3}, {2});
  at::Tensor result = at::take(t, index);

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "TakeFloatNonContiguousIndex ";
  write_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// Shape: multi-dimensional index
TEST_F(TakeTest, TakeFloatMultiDimIndex) {
  at::Tensor t = make_test_tensor(at::kFloat);
  at::Tensor index = make_index_tensor({0, 3, 7, 10}).reshape({2, 2});
  at::Tensor result = at::take(t, index);

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "TakeFloatMultiDimIndex ";
  write_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// Shape: scalar index (0-dim)
TEST_F(TakeTest, TakeFloatScalarIndex) {
  at::Tensor t = make_test_tensor(at::kFloat);
  at::Tensor index = at::full({}, 7, at::kLong);
  at::Tensor result = at::take(t, index);

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "TakeFloatScalarIndex ";
  write_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// Shape: empty index (boundary)
TEST_F(TakeTest, TakeFloatEmptyIndex) {
  at::Tensor t = make_test_tensor(at::kFloat);
  at::Tensor index = at::empty({0}, at::TensorOptions().dtype(at::kLong));
  at::Tensor result = at::take(t, index);

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "TakeFloatEmptyIndex ";
  write_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

TEST_F(TakeTest, TakeEmptyInputNonEmptyIndexThrows) {
  at::Tensor t = at::empty({0}, at::TensorOptions().dtype(at::kFloat));
  at::Tensor index = make_index_tensor({0});

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "TakeEmptyInputNonEmptyIndexThrows ";
  try {
    at::Tensor result = at::take(t, index);
    write_result_to_file(&file, result);
  } catch (const std::exception&) {
    file << "exception ";
  }
  file << "\n";
  file.saveFile();
}

// Shape: duplicate indices
TEST_F(TakeTest, TakeFloatDuplicateIndices) {
  at::Tensor t = make_test_tensor(at::kFloat);
  at::Tensor index = make_index_tensor({1, 1, 3, 1});
  at::Tensor result = at::take(t, index);

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "TakeFloatDuplicateIndices ";
  write_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// Exception: index out of range
TEST_F(TakeTest, TakeExceptionOutOfRange) {
  at::Tensor t = make_test_tensor(at::kFloat);
  at::Tensor index = make_index_tensor({100});  // out of range

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "TakeExceptionOutOfRange ";
  try {
    at::Tensor result = at::take(t, index);
    write_result_to_file(&file, result);
  } catch (const std::exception&) {
    file << "exception ";
  }
  file << "\n";
  file.saveFile();
}

TEST_F(TakeTest, TakeFloatNegativeIndex) {
  at::Tensor t = make_test_tensor(at::kFloat);
  at::Tensor index = make_index_tensor({-1, 0});

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "TakeFloatNegativeIndex ";
  try {
    at::Tensor result = at::take(t, index);
    write_result_to_file(&file, result);
  } catch (const std::exception&) {
    file << "exception ";
  }
  file << "\n";
  file.saveFile();
}

TEST_F(TakeTest, TakeIntIndexThrows) {
  at::Tensor t = make_test_tensor(at::kFloat);
  at::Tensor index = make_int_index_tensor({0, 1});

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "TakeIntIndexThrows ";
  try {
    at::Tensor result = at::take(t, index);
    write_result_to_file(&file, result);
  } catch (const std::exception&) {
    file << "exception ";
  }
  file << "\n";
  file.saveFile();
}

TEST_F(TakeTest, TakeExceptionAtNumel) {
  at::Tensor t = make_test_tensor(at::kFloat);
  at::Tensor index = make_index_tensor({t.numel()});

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "TakeExceptionAtNumel ";
  try {
    at::Tensor result = at::take(t, index);
    write_result_to_file(&file, result);
  } catch (const std::exception&) {
    file << "exception ";
  }
  file << "\n";
  file.saveFile();
}

TEST_F(TakeTest, TakeExceptionBelowNegativeNumel) {
  at::Tensor t = make_test_tensor(at::kFloat);
  at::Tensor index = make_index_tensor({-t.numel() - 1});

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "TakeExceptionBelowNegativeNumel ";
  try {
    at::Tensor result = at::take(t, index);
    write_result_to_file(&file, result);
  } catch (const std::exception&) {
    file << "exception ";
  }
  file << "\n";
  file.saveFile();
}

TEST_F(TakeTest, TakeCudaNegativeIndex) {
  at::Tensor t = make_test_tensor(at::kFloat);
  at::Tensor index = make_index_tensor({-1});

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "TakeCudaNegativeIndex ";
  try {
    at::Tensor result = at::take(t.cuda(), index.cuda());
    write_result_to_file(&file, result.cpu());
  } catch (const std::exception&) {
    file << "exception ";
  }
  file << "\n";
  file.saveFile();
}

TEST_F(TakeTest, TakeCudaExceptionAtNumel) {
  at::Tensor t = make_test_tensor(at::kFloat);
  at::Tensor index = make_index_tensor({t.numel()});

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "TakeCudaExceptionAtNumel ";
  try {
    at::Tensor result = at::take(t.cuda(), index.cuda());
    write_result_to_file(&file, result.cpu());
  } catch (const std::exception&) {
    file << "exception ";
  }
  file << "\n";
  file.saveFile();
}

TEST_F(TakeTest, TakeCudaExceptionBelowNegativeNumel) {
  at::Tensor t = make_test_tensor(at::kFloat);
  at::Tensor index = make_index_tensor({-t.numel() - 1});

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "TakeCudaExceptionBelowNegativeNumel ";
  try {
    at::Tensor result = at::take(t.cuda(), index.cuda());
    write_result_to_file(&file, result.cpu());
  } catch (const std::exception&) {
    file << "exception ";
  }
  file << "\n";
  file.saveFile();
}

TEST_F(TakeTest, TakeCudaUInt16Throws) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "TakeCudaUInt16Throws ";
  try {
    at::Tensor t = make_index_tensor({2, 2, 2, 2}).to(at::kUInt16);
    at::Tensor index = make_index_tensor({0, 3, 1});
    at::Tensor result = at::take(t.cuda(), index.cuda());
    write_result_to_file(&file, result.cpu());
  } catch (const std::exception&) {
    file << "exception ";
  }
  file << "\n";
  file.saveFile();
}

TEST_F(TakeTest, TakeCudaUInt32Throws) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "TakeCudaUInt32Throws ";
  try {
    at::Tensor t = make_index_tensor({2, 2, 2, 2}).to(at::kUInt32);
    at::Tensor index = make_index_tensor({0, 3, 1});
    at::Tensor result = at::take(t.cuda(), index.cuda());
    write_result_to_file(&file, result.cpu());
  } catch (const std::exception&) {
    file << "exception ";
  }
  file << "\n";
  file.saveFile();
}

}  // namespace test
}  // namespace at
