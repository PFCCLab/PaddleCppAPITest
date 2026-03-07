#include <ATen/ATen.h>
#include <ATen/core/Tensor.h>
#include <ATen/ops/zeros.h>
#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "../../src/file_manager.h"

extern paddle_api_test::ThreadSafeParam g_custom_param;

namespace at {
namespace test {

using paddle_api_test::FileManerger;
using paddle_api_test::ThreadSafeParam;

class DetachTest : public ::testing::Test {
 protected:
  void SetUp() override {
    std::vector<int64_t> shape = {3, 4};
    test_tensor = at::zeros(shape, at::kFloat);
    float* data = test_tensor.data_ptr<float>();
    for (int64_t i = 0; i < 12; ++i) {
      data[i] = static_cast<float>(i + 1);
    }
  }
  at::Tensor test_tensor;
};

static void write_detach_result_to_file(FileManerger* file,
                                        const at::Tensor& result,
                                        const at::Tensor& original) {
  *file << std::to_string(result.dim()) << " ";
  *file << std::to_string(result.numel()) << " ";

  // 写入形状信息
  for (int64_t i = 0; i < result.dim(); ++i) {
    *file << std::to_string(result.sizes()[i]) << " ";
  }

  // 写入数据内容
  float* result_data = result.data_ptr<float>();
  for (int64_t i = 0; i < result.numel(); ++i) {
    *file << std::to_string(result_data[i]) << " ";
  }

  // 验证数据指针是否相同（共享存储）
  *file << std::to_string(result.data_ptr<float>() ==
                          original.data_ptr<float>())
        << " ";
}

// 测试 detach() 方法 - 创建新的 tensor，不跟踪梯度
TEST_F(DetachTest, BasicDetach) {
  at::Tensor detached = test_tensor.detach();
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();
  write_detach_result_to_file(&file, detached, test_tensor);
  file.saveFile();
}

// 测试 detach_() in-place 方法
TEST_F(DetachTest, InplaceDetach) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();

  // 保存原始指针
  float* original_ptr = test_tensor.data_ptr<float>();

  // 调用 in-place 版本
  at::Tensor& result = test_tensor.detach_();

  // 验证返回的是同一个 tensor
  file << std::to_string(result.data_ptr<float>() == original_ptr) << " ";

  // 写入数据
  float* data = result.data_ptr<float>();
  for (int64_t i = 0; i < result.numel(); ++i) {
    file << std::to_string(data[i]) << " ";
  }
  file.saveFile();
}

// 测试 detach 后修改数据
TEST_F(DetachTest, DetachAndModify) {
  at::Tensor detached = test_tensor.detach();

  // 修改 detached tensor 的数据
  float* detached_data = detached.data_ptr<float>();
  detached_data[0] = 99.0f;
  detached_data[1] = 88.0f;

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();

  // 验证原始 tensor 的数据也被修改了（因为共享存储）
  float* original_data = test_tensor.data_ptr<float>();
  file << std::to_string(original_data[0]) << " ";
  file << std::to_string(original_data[1]) << " ";
  file << std::to_string(detached_data[0]) << " ";
  file << std::to_string(detached_data[1]) << " ";
  file.saveFile();
}

// 测试不同类型 tensor 的 detach
TEST_F(DetachTest, DetachDifferentTensor) {
  at::Tensor different_tensor = at::zeros({2, 2}, at::kFloat);
  float* data = different_tensor.data_ptr<float>();
  data[0] = 1.0f;
  data[1] = 2.0f;
  data[2] = 3.0f;
  data[3] = 4.0f;

  at::Tensor detached = different_tensor.detach();

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();

  file << std::to_string(detached.numel()) << " ";
  file << std::to_string(detached.dim()) << " ";

  float* detached_data = detached.data_ptr<float>();
  for (int64_t i = 0; i < detached.numel(); ++i) {
    file << std::to_string(detached_data[i]) << " ";
  }
  file.saveFile();
}

// 测试多维 tensor 的 detach
TEST_F(DetachTest, MultiDimensionalDetach) {
  at::Tensor multi_tensor = at::zeros({2, 3, 4}, at::kFloat);
  float* data = multi_tensor.data_ptr<float>();
  for (int64_t i = 0; i < 24; ++i) {
    data[i] = static_cast<float>(i);
  }

  at::Tensor detached = multi_tensor.detach();

  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();

  file << std::to_string(detached.dim()) << " ";
  file << std::to_string(detached.sizes()[0]) << " ";
  file << std::to_string(detached.sizes()[1]) << " ";
  file << std::to_string(detached.sizes()[2]) << " ";
  file << std::to_string(detached.numel()) << " ";

  // 验证数据共享
  file << std::to_string(detached.data_ptr<float>() ==
                         multi_tensor.data_ptr<float>())
       << " ";
  file.saveFile();
}

}  // namespace test
}  // namespace at
