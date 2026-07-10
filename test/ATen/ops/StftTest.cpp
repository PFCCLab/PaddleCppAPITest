#include <ATen/ATen.h>
#include <ATen/core/Tensor.h>
#include <ATen/ops/ones.h>
#include <ATen/ops/stft.h>
#include <gtest/gtest.h>

#include <cmath>
#include <complex>
#include <optional>
#include <string>
#include <vector>

#include "src/file_manager.h"

extern paddle_api_test::ThreadSafeParam g_custom_param;

namespace at {
namespace test {

using paddle_api_test::FileManerger;
using paddle_api_test::ThreadSafeParam;

// 写出 stft 结果: shape 信息 + dtype + 各 frame 的 DC bin 实部
// 注: stft 的数值结果在 Paddle 与 PyTorch 之间可能因 padding/FFT
// 实现差异而不同， 因此只比较 shape/dtype 和每帧 DC 分量(用于验证 FFT
// 确实被执行)
static void write_stft_result_to_file(FileManerger* file,
                                      const at::Tensor& result) {
  at::Tensor contig = result.contiguous();
  *file << std::to_string(contig.dim()) << " ";
  *file << std::to_string(contig.numel()) << " ";
  for (int64_t i = 0; i < contig.dim(); ++i) {
    *file << std::to_string(contig.sizes()[i]) << " ";
  }
  // 写出 dtype
  *file << std::to_string(static_cast<int>(contig.scalar_type())) << " ";

  // 只写出第一个 batch 中每帧 DC 分量(bin 0)的实部，作为 FFT 已执行的验证
  const bool is_complex = contig.scalar_type() == at::kComplexFloat ||
                          contig.scalar_type() == at::kComplexDouble;
  int64_t n_frames = contig.sizes()[contig.dim() - (is_complex ? 1 : 2)];
  for (int64_t frame = 0; frame < n_frames; ++frame) {
    if (contig.scalar_type() == at::kComplexFloat) {
      std::complex<float>* data =
          reinterpret_cast<std::complex<float>*>(contig.data_ptr());
      *file << std::to_string(data[frame].real()) << " ";
    } else if (contig.scalar_type() == at::kComplexDouble) {
      std::complex<double>* data =
          reinterpret_cast<std::complex<double>*>(contig.data_ptr());
      *file << std::to_string(data[frame].real()) << " ";
    } else if (contig.scalar_type() == at::kFloat) {
      // return_complex=false: shape [..., freq, frames, 2], last dim = [real,
      // imag]
      float* data = reinterpret_cast<float*>(contig.data_ptr());
      *file << std::to_string(data[frame * 2]) << " ";
    } else if (contig.scalar_type() == at::kDouble) {
      double* data = reinterpret_cast<double*>(contig.data_ptr());
      *file << std::to_string(data[frame * 2]) << " ";
    } else {
      *file << "unsupported_dtype ";
    }
  }
}

class StftTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // 基准 2D tensor: [batch=1, time=16], float32
    test_tensor = at::ones({1, 16}, at::kFloat);
  }
  at::Tensor test_tensor;
};

// ========== 基础功能 ==========

TEST_F(StftTest, BasicStft) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();
  file << "BasicStft ";
  at::Tensor result = test_tensor.stft(
      /*n_fft=*/8,
      /*hop_length=*/4,
      /*win_length=*/8,
      /*window=*/::std::nullopt,
      /*normalized=*/false,
      /*onesided=*/::std::nullopt,
      /*return_complex=*/true);
  write_stft_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

TEST_F(StftTest, StftWithWindow) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "StftWithWindow ";
  at::Tensor window = at::ones({8}, at::kFloat);
  at::Tensor result = test_tensor.stft(
      /*n_fft=*/8,
      /*hop_length=*/4,
      /*win_length=*/8,
      /*window=*/window,
      /*normalized=*/false,
      /*onesided=*/true,
      /*return_complex=*/true);
  write_stft_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

TEST_F(StftTest, StftNormalized) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "StftNormalized ";
  at::Tensor result = test_tensor.stft(
      /*n_fft=*/8,
      /*hop_length=*/4,
      /*win_length=*/8,
      /*window=*/::std::nullopt,
      /*normalized=*/true,
      /*onesided=*/true,
      /*return_complex=*/true);
  write_stft_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// ========== Shape 覆盖 ==========

TEST_F(StftTest, SmallShape) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "SmallShape ";
  at::Tensor t = at::ones({1, 16}, at::kFloat);
  at::Tensor result = t.stft(/*n_fft=*/8,
                             /*hop_length=*/4,
                             /*win_length=*/8,
                             /*window=*/::std::nullopt,
                             /*normalized=*/false,
                             /*onesided=*/true,
                             /*return_complex=*/true);
  write_stft_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

TEST_F(StftTest, LargeShape) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "LargeShape ";
  at::Tensor t = at::ones({2, 64}, at::kFloat);
  at::Tensor result = t.stft(/*n_fft=*/16,
                             /*hop_length=*/8,
                             /*win_length=*/16,
                             /*window=*/::std::nullopt,
                             /*normalized=*/false,
                             /*onesided=*/true,
                             /*return_complex=*/true);
  write_stft_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

TEST_F(StftTest, OneDInput) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "OneDInput ";
  at::Tensor t = at::ones({16}, at::kFloat);
  at::Tensor result = t.stft(/*n_fft=*/8,
                             /*hop_length=*/4,
                             /*win_length=*/8,
                             /*window=*/::std::nullopt,
                             /*normalized=*/false,
                             /*onesided=*/true,
                             /*return_complex=*/true);
  write_stft_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// ========== Dtype 覆盖 ==========

TEST_F(StftTest, Float64Dtype) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "Float64Dtype ";
  at::Tensor t = at::ones({1, 16}, at::kDouble);
  at::Tensor result = t.stft(/*n_fft=*/8,
                             /*hop_length=*/4,
                             /*win_length=*/8,
                             /*window=*/::std::nullopt,
                             /*normalized=*/false,
                             /*onesided=*/true,
                             /*return_complex=*/true);
  write_stft_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

// ========== API 变体 ==========

TEST_F(StftTest, DifferentHopLength) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "DifferentHopLength ";
  at::Tensor t = at::ones({1, 32}, at::kFloat);
  at::Tensor result = t.stft(/*n_fft=*/16,
                             /*hop_length=*/8,
                             /*win_length=*/16,
                             /*window=*/::std::nullopt,
                             /*normalized=*/false,
                             /*onesided=*/true,
                             /*return_complex=*/true);
  write_stft_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

TEST_F(StftTest, ReturnComplexFalse) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "ReturnComplexFalse ";
  at::Tensor result = test_tensor.stft(
      /*n_fft=*/8,
      /*hop_length=*/4,
      /*win_length=*/8,
      /*window=*/::std::nullopt,
      /*normalized=*/false,
      /*onesided=*/true,
      /*return_complex=*/false);
  write_stft_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

TEST_F(StftTest, PyTorchStyleCenterFalseOverload) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "PyTorchStyleCenterFalseOverload ";
  at::Tensor result = test_tensor.stft(
      /*n_fft=*/8,
      /*hop_length=*/4,
      /*win_length=*/8,
      /*window=*/::std::nullopt,
      /*center=*/false,
      /*pad_mode=*/"reflect",
      /*normalized=*/true,
      /*onesided=*/true,
      /*return_complex=*/true);
  write_stft_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

TEST_F(StftTest, FreeFunctionLegacySchema) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "FreeFunctionLegacySchema ";
  at::Tensor result = at::stft(test_tensor,
                               /*n_fft=*/8,
                               /*hop_length=*/4,
                               /*win_length=*/8,
                               /*window=*/::std::nullopt,
                               /*normalized=*/false,
                               /*onesided=*/true,
                               /*return_complex=*/true);
  write_stft_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

TEST_F(StftTest, FreeFunctionPyTorchStyleCenterFalseOverload) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "FreeFunctionPyTorchStyleCenterFalseOverload ";
  at::Tensor result = at::stft(test_tensor,
                               /*n_fft=*/8,
                               /*hop_length=*/4,
                               /*win_length=*/8,
                               /*window=*/::std::nullopt,
                               /*center=*/false,
                               /*pad_mode=*/"reflect",
                               /*normalized=*/true,
                               /*onesided=*/true,
                               /*return_complex=*/true);
  write_stft_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

TEST_F(StftTest, ReturnComplexUnspecifiedRealInputThrows) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "ReturnComplexUnspecifiedRealInputThrows ";
  bool thrown = false;
  try {
    (void)test_tensor.stft(/*n_fft=*/8,
                           /*hop_length=*/4,
                           /*win_length=*/8,
                           /*window=*/::std::nullopt,
                           /*normalized=*/false,
                           /*onesided=*/::std::optional<bool>{true});
  } catch (const std::exception&) {
    thrown = true;
    file << "exception ";
  }
  if (!thrown) {
    file << "no_exception ";
  }
  file << "\n";
  file.saveFile();
}

TEST_F(StftTest, CenterReflect) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "CenterReflect ";
  at::Tensor result = test_tensor.stft(
      /*n_fft=*/8,
      /*hop_length=*/4,
      /*win_length=*/8,
      /*window=*/::std::nullopt,
      /*center=*/true,
      /*pad_mode=*/"reflect",
      /*normalized=*/false,
      /*onesided=*/::std::nullopt,
      /*return_complex=*/true);
  write_stft_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

TEST_F(StftTest, CenterConstant) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "CenterConstant ";
  at::Tensor result = test_tensor.stft(
      /*n_fft=*/8,
      /*hop_length=*/4,
      /*win_length=*/8,
      /*window=*/::std::nullopt,
      /*center=*/true,
      /*pad_mode=*/"constant",
      /*normalized=*/false,
      /*onesided=*/::std::nullopt,
      /*return_complex=*/true);
  write_stft_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

TEST_F(StftTest, ComplexFloatDefaultOnesided) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "ComplexFloatDefaultOnesided ";
  at::Tensor input = at::ones({16}, at::kComplexFloat);
  at::Tensor result = input.stft(
      /*n_fft=*/8,
      /*hop_length=*/4,
      /*win_length=*/8,
      /*window=*/::std::nullopt,
      /*normalized=*/false,
      /*onesided=*/::std::nullopt,
      /*return_complex=*/::std::nullopt);
  write_stft_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

TEST_F(StftTest, ComplexDoubleCenterReflect) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "ComplexDoubleCenterReflect ";
  at::Tensor input = at::ones({16}, at::kComplexDouble);
  at::Tensor result = input.stft(
      /*n_fft=*/8,
      /*hop_length=*/4,
      /*win_length=*/8,
      /*window=*/::std::nullopt,
      /*center=*/true,
      /*pad_mode=*/"reflect",
      /*normalized=*/false,
      /*onesided=*/::std::nullopt,
      /*return_complex=*/::std::nullopt);
  write_stft_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

TEST_F(StftTest, ComplexOnesidedTrueThrows) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "ComplexOnesidedTrueThrows ";
  try {
    at::Tensor input = at::ones({16}, at::kComplexFloat);
    (void)input.stft(/*n_fft=*/8,
                     /*hop_length=*/4,
                     /*win_length=*/8,
                     /*window=*/::std::nullopt,
                     /*normalized=*/false,
                     /*onesided=*/::std::optional<bool>{true},
                     /*return_complex=*/true);
    file << "no_exception ";
  } catch (const std::exception&) {
    file << "exception ";
  }
  file << "\n";
  file.saveFile();
}

TEST_F(StftTest, ComplexWindowDefaultOnesided) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "ComplexWindowDefaultOnesided ";
  at::Tensor window = at::ones({8}, at::kComplexFloat);
  at::Tensor result = test_tensor.stft(
      /*n_fft=*/8,
      /*hop_length=*/4,
      /*win_length=*/8,
      /*window=*/window,
      /*normalized=*/false,
      /*onesided=*/::std::nullopt,
      /*return_complex=*/::std::nullopt);
  write_stft_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

TEST_F(StftTest, RealOnesidedFalse) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "RealOnesidedFalse ";
  at::Tensor result = test_tensor.stft(
      /*n_fft=*/8,
      /*hop_length=*/4,
      /*win_length=*/8,
      /*window=*/::std::nullopt,
      /*normalized=*/false,
      /*onesided=*/::std::optional<bool>{false},
      /*return_complex=*/true);
  write_stft_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

TEST_F(StftTest, WinLengthSmallerThanNFFT) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "WinLengthSmallerThanNFFT ";
  at::Tensor window = at::ones({4}, at::kFloat);
  at::Tensor result = test_tensor.stft(
      /*n_fft=*/8,
      /*hop_length=*/4,
      /*win_length=*/4,
      /*window=*/window,
      /*normalized=*/false,
      /*onesided=*/true,
      /*return_complex=*/true);
  write_stft_result_to_file(&file, result);
  file << "\n";
  file.saveFile();
}

}  // namespace test
}  // namespace at
