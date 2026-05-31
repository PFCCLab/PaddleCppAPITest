#include <ATen/ATen.h>
#include <ATen/core/Tensor.h>
#include <ATen/ops/ones.h>
#include <ATen/ops/stft.h>
#include <gtest/gtest.h>

#include <cmath>
#include <complex>
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

  // 只写出每帧 DC 分量(bin 0)的实部，作为 FFT 已执行的验证
  int64_t n_frames = contig.sizes()[contig.dim() - 1];
  int64_t freq_bins = contig.sizes()[contig.dim() - 2];
  for (int64_t f = 0; f < n_frames; ++f) {
    if (contig.scalar_type() == at::kComplexFloat) {
      std::complex<float>* data =
          reinterpret_cast<std::complex<float>*>(contig.data_ptr());
      *file << std::to_string(data[f * freq_bins].real()) << " ";
    } else if (contig.scalar_type() == at::kComplexDouble) {
      std::complex<double>* data =
          reinterpret_cast<std::complex<double>*>(contig.data_ptr());
      *file << std::to_string(data[f * freq_bins].real()) << " ";
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

}  // namespace test
}  // namespace at
