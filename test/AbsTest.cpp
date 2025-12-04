#include <ATen/ATen.h>
#include <ATen/core/Tensor.h>
#include <ATen/ops/abs.h>
#include <ATen/ops/zeros.h>
#include <gtest/gtest.h>
#include <torch/all.h>

#include <vector>

namespace at {
namespace test {

class AbsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // 创建包含正数、负数、零的测试张量
    std::vector<int64_t> shape = {4};
    test_tensor = at::zeros(shape, at::kFloat);

    // 设置测试数据: [1.0, -2.0, 0.0, -3.5]
    float* data = test_tensor.data_ptr<float>();
    data[0] = 1.0f;
    data[1] = -2.0f;
    data[2] = 0.0f;
    data[3] = -3.5f;
  }

  at::Tensor test_tensor;
};

// 测试所有元素的绝对值计算
TEST_F(AbsTest, BasicAbs) {
  at::Tensor result = at::abs(test_tensor);

  // 验证结果张量的形状
  EXPECT_EQ(result.sizes(), test_tensor.sizes());

  // 验证元素绝对值
  float* result_data = result.data_ptr<float>();
  EXPECT_FLOAT_EQ(result_data[0], 1.0f);  // abs(1.0) = 1.0
  EXPECT_FLOAT_EQ(result_data[1], 2.0f);  // abs(-2.0) = 2.0
  EXPECT_FLOAT_EQ(result_data[2], 0.0f);  // abs(0.0) = 0.0
  EXPECT_FLOAT_EQ(result_data[3], 3.5f);  // abs(-3.5) = 3.5
}

// 测试全正数张量的绝对值
TEST_F(AbsTest, PositiveTensor) {
  at::Tensor positive_tensor = at::zeros({3}, at::kFloat);
  float* data = positive_tensor.data_ptr<float>();
  data[0] = 1.5f;
  data[1] = 3.0f;
  data[2] = 7.2f;

  at::Tensor result = at::abs(positive_tensor);

  float* result_data = result.data_ptr<float>();
  EXPECT_FLOAT_EQ(result_data[0], 1.5f);
  EXPECT_FLOAT_EQ(result_data[1], 3.0f);
  EXPECT_FLOAT_EQ(result_data[2], 7.2f);
}

// 测试全负数张量的绝对值
TEST_F(AbsTest, NegativeTensor) {
  at::Tensor negative_tensor = at::zeros({3}, at::kFloat);
  float* data = negative_tensor.data_ptr<float>();
  data[0] = -1.5f;
  data[1] = -3.0f;
  data[2] = -7.2f;

  at::Tensor result = at::abs(negative_tensor);

  float* result_data = result.data_ptr<float>();
  EXPECT_FLOAT_EQ(result_data[0], 1.5f);
  EXPECT_FLOAT_EQ(result_data[1], 3.0f);
  EXPECT_FLOAT_EQ(result_data[2], 7.2f);
}

// 测试返回张量与原张量的独立性
TEST_F(AbsTest, ResultIndependence) {
  at::Tensor result = at::abs(test_tensor);

  // 修改原张量，验证结果张量不受影响
  float* original_data = test_tensor.data_ptr<float>();
  original_data[0] = 10.0f;

  float* result_data = result.data_ptr<float>();
  EXPECT_FLOAT_EQ(result_data[0], 1.0f);     // 结果应该保持不变
  EXPECT_FLOAT_EQ(original_data[0], 10.0f);  // 原张量已被修改
}

// 测试二维张量的绝对值
TEST_F(AbsTest, TwoDimensionalTensor) {
  at::Tensor tensor_2d = at::zeros({2, 2}, at::kFloat);
  float* data = tensor_2d.data_ptr<float>();
  data[0] = -1.0f;
  data[1] = 2.0f;
  data[2] = 0.0f;
  data[3] = -1.5f;

  at::Tensor result = at::abs(tensor_2d);

  // 验证形状
  EXPECT_EQ(result.dim(), 2);
  EXPECT_EQ(result.sizes()[0], 2);
  EXPECT_EQ(result.sizes()[1], 2);

  // 验证数值
  float* result_data = result.data_ptr<float>();
  EXPECT_FLOAT_EQ(result_data[0], 1.0f);
  EXPECT_FLOAT_EQ(result_data[1], 2.0f);
  EXPECT_FLOAT_EQ(result_data[2], 0.0f);
  EXPECT_FLOAT_EQ(result_data[3], 1.5f);
}

// 测试不同数据类型的绝对值（整数类型）
TEST_F(AbsTest, IntTensor) {
  at::Tensor int_tensor = at::zeros({3}, at::kInt);
  int* data = int_tensor.data_ptr<int>();
  data[0] = -10;
  data[1] = 20;
  data[2] = -5;

  at::Tensor result = at::abs(int_tensor);

  int* result_data = result.data_ptr<int>();
  EXPECT_EQ(result_data[0], 10);
  EXPECT_EQ(result_data[1], 20);
  EXPECT_EQ(result_data[2], 5);
}

}  // namespace test
}  // namespace at
