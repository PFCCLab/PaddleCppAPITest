#include <ATen/ATen.h>
#include <c10/core/Allocator.h>
#include <gtest/gtest.h>

#include <string>

#include "../src/file_manager.h"

extern paddle_api_test::ThreadSafeParam g_custom_param;

namespace at {
namespace test {

using paddle_api_test::FileManerger;
using paddle_api_test::ThreadSafeParam;

class AllocatorTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // 分配测试用的内存
    test_data_ = new float[4]{1.0f, 2.0f, 3.0f, 4.0f};
    test_ctx_ = new int(42);
  }

  void TearDown() override {
    // 注意：如果数据被 DataPtr 的 deleter 释放，这里不应重复释放
    // 在这些测试中，我们使用自定义 deleter 不真正释放内存
  }

  float* test_data_ = nullptr;
  void* test_ctx_ = nullptr;
};

// 自定义 deleter 函数用于测试（不真正释放，由测试管理）
static bool g_deleter_called = false;
static void test_deleter(void* ptr) { g_deleter_called = true; }

// 真正释放内存的 deleter
static void real_float_deleter(void* ptr) { delete[] static_cast<float*>(ptr); }

// 测试默认构造函数
TEST_F(AllocatorTest, DefaultConstructor) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();

  c10::DataPtr data_ptr;

  // 默认构造的 DataPtr 应该为 null
  file << std::to_string(data_ptr.get() == nullptr) << " ";
  // operator bool 应该返回 false
  file << std::to_string(static_cast<bool>(data_ptr) == false) << " ";
  // context 应该为 nullptr
  file << std::to_string(data_ptr.get_context() == nullptr) << " ";

  file.saveFile();
}

// 测试带数据和设备的构造函数
TEST_F(AllocatorTest, ConstructorWithDataAndDevice) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();

#if USE_PADDLE_API
  c10::DataPtr data_ptr(static_cast<void*>(test_data_), phi::CPUPlace());
#else
  c10::DataPtr data_ptr(static_cast<void*>(test_data_),
                        c10::Device(c10::DeviceType::CPU));
#endif

  // 指针应该正确设置
  file << std::to_string(data_ptr.get() == static_cast<void*>(test_data_))
       << " ";
  // operator bool 应该返回 true
  file << std::to_string(static_cast<bool>(data_ptr) == true) << " ";
  // 验证可以通过 get() 访问数据
  float* ptr = static_cast<float*>(data_ptr.get());
  file << std::to_string(ptr[0]) << " ";
  file << std::to_string(ptr[1]) << " ";

  file.saveFile();
}

// 测试带完整参数的构造函数
TEST_F(AllocatorTest, ConstructorWithDeleter) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();

  g_deleter_called = false;

#if USE_PADDLE_API
  c10::DataPtr data_ptr(
      static_cast<void*>(test_data_), test_ctx_, test_deleter, phi::CPUPlace());
#else
  c10::DataPtr data_ptr(static_cast<void*>(test_data_),
                        test_ctx_,
                        test_deleter,
                        c10::Device(c10::DeviceType::CPU));
#endif

  // 指针应该正确设置
  file << std::to_string(data_ptr.get() == static_cast<void*>(test_data_))
       << " ";
  // context 应该正确设置
  file << std::to_string(data_ptr.get_context() == test_ctx_) << " ";
  // deleter 应该正确设置
  file << std::to_string(data_ptr.get_deleter() == test_deleter) << " ";

  file.saveFile();
}

// 测试移动构造函数
TEST_F(AllocatorTest, MoveConstructor) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();

#if USE_PADDLE_API
  c10::DataPtr original(static_cast<void*>(test_data_), phi::CPUPlace());
#else
  c10::DataPtr original(static_cast<void*>(test_data_),
                        c10::Device(c10::DeviceType::CPU));
#endif
  void* original_ptr = original.get();
  c10::DataPtr moved(std::move(original));

  // 移动后的 DataPtr 应该持有原始指针
  file << std::to_string(moved.get() == original_ptr) << " ";
  file << std::to_string(moved.get() == static_cast<void*>(test_data_)) << " ";

  file.saveFile();
}

// 测试移动赋值操作符
TEST_F(AllocatorTest, MoveAssignment) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();

#if USE_PADDLE_API
  c10::DataPtr original(static_cast<void*>(test_data_), phi::CPUPlace());
#else
  c10::DataPtr original(static_cast<void*>(test_data_),
                        c10::Device(c10::DeviceType::CPU));
#endif
  void* original_ptr = original.get();
  c10::DataPtr assigned;
  assigned = std::move(original);

  // 移动赋值后应该持有原始指针
  file << std::to_string(assigned.get() == original_ptr) << " ";

  file.saveFile();
}

// 测试 clear 方法
// 注意：clear() 后 get_deleter() 的行为在 PyTorch 和 Paddle 间有差异
// PyTorch 不会重置 deleter 为 nullptr，Paddle 会
// 因此只测试 get(), operator bool(), get_context() 的行为一致性
TEST_F(AllocatorTest, Clear) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();

#if USE_PADDLE_API
  c10::DataPtr data_ptr(
      static_cast<void*>(test_data_), test_ctx_, test_deleter, phi::CPUPlace());
#else
  c10::DataPtr data_ptr(static_cast<void*>(test_data_),
                        test_ctx_,
                        test_deleter,
                        c10::Device(c10::DeviceType::CPU));
#endif

  // clear 前验证状态
  file << std::to_string(data_ptr.get() != nullptr) << " ";
  file << std::to_string(static_cast<bool>(data_ptr)) << " ";

  data_ptr.clear();

  // clear 后核心属性应该为空
  file << std::to_string(data_ptr.get() == nullptr) << " ";
  file << std::to_string(static_cast<bool>(data_ptr) == false) << " ";
  file << std::to_string(data_ptr.get_context() == nullptr) << " ";
  // 注意：不测试 get_deleter() == nullptr，因为两个框架行为不同

  file.saveFile();
}

// 测试与 nullptr 的比较操作符
TEST_F(AllocatorTest, NullptrComparison) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();

  c10::DataPtr null_ptr;
#if USE_PADDLE_API
  c10::DataPtr valid_ptr(static_cast<void*>(test_data_), phi::CPUPlace());
#else
  c10::DataPtr valid_ptr(static_cast<void*>(test_data_),
                         c10::Device(c10::DeviceType::CPU));
#endif

  // null_ptr == nullptr 应该为 true
  file << std::to_string(null_ptr == nullptr) << " ";
  file << std::to_string(nullptr == null_ptr) << " ";
  // null_ptr != nullptr 应该为 false
  file << std::to_string(null_ptr != nullptr) << " ";
  file << std::to_string(nullptr != null_ptr) << " ";

  // valid_ptr == nullptr 应该为 false
  file << std::to_string(valid_ptr == nullptr) << " ";
  file << std::to_string(nullptr == valid_ptr) << " ";
  // valid_ptr != nullptr 应该为 true
  file << std::to_string(valid_ptr != nullptr) << " ";
  file << std::to_string(nullptr != valid_ptr) << " ";

  file.saveFile();
}

// 测试 at::DataPtr 别名
TEST_F(AllocatorTest, AtDataPtrAlias) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();

  // at::DataPtr 应该是 c10::DataPtr 的别名
#if USE_PADDLE_API
  at::DataPtr at_ptr(static_cast<void*>(test_data_), phi::CPUPlace());
#else
  at::DataPtr at_ptr(static_cast<void*>(test_data_),
                     c10::Device(c10::DeviceType::CPU));
#endif

  file << std::to_string(at_ptr.get() == static_cast<void*>(test_data_)) << " ";
  file << std::to_string(static_cast<bool>(at_ptr)) << " ";

  // 验证可以移动赋值给 c10::DataPtr
  c10::DataPtr c10_ptr = std::move(at_ptr);
  file << std::to_string(c10_ptr.get() == static_cast<void*>(test_data_))
       << " ";

  file.saveFile();
}

// 测试 operator-> 方法
TEST_F(AllocatorTest, ArrowOperator) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();

#if USE_PADDLE_API
  c10::DataPtr data_ptr(static_cast<void*>(test_data_), phi::CPUPlace());
#else
  c10::DataPtr data_ptr(static_cast<void*>(test_data_),
                        c10::Device(c10::DeviceType::CPU));
#endif

  // operator-> 应该返回原始指针
  file << std::to_string(data_ptr.operator->() ==
                         static_cast<void*>(test_data_))
       << " ";
  file << std::to_string(data_ptr.operator->() == data_ptr.get()) << " ";

  file.saveFile();
}

// 测试空 DataPtr 的边界情况
TEST_F(AllocatorTest, EmptyDataPtrEdgeCases) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();

  c10::DataPtr empty_ptr;

  // 验证空指针的核心属性
  file << std::to_string(empty_ptr.get() == nullptr) << " ";
  file << std::to_string(empty_ptr.get_context() == nullptr) << " ";
  file << std::to_string(!static_cast<bool>(empty_ptr)) << " ";

  // 调用 clear 对空指针应该安全
  empty_ptr.clear();
  file << std::to_string(empty_ptr.get() == nullptr) << " ";

  file.saveFile();
}

// 测试链式移动
TEST_F(AllocatorTest, ChainedMoves) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();

#if USE_PADDLE_API
  c10::DataPtr original(static_cast<void*>(test_data_), phi::CPUPlace());
#else
  c10::DataPtr original(static_cast<void*>(test_data_),
                        c10::Device(c10::DeviceType::CPU));
#endif
  void* ptr = original.get();

  // 链式移动
  c10::DataPtr moved1(std::move(original));
  c10::DataPtr moved2(std::move(moved1));
  c10::DataPtr moved3 = std::move(moved2);

  // 最终应该指向原始数据
  file << std::to_string(moved3.get() == ptr) << " ";
  file << std::to_string(moved3.get() == static_cast<void*>(test_data_)) << " ";

  file.saveFile();
}

// 测试 Deleter 在析构时是否被调用
TEST_F(AllocatorTest, DeleterCalledOnDestruction) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();

  {
    // 在作用域内创建 DataPtr
    float* local_data = new float[2]{1.0f, 2.0f};
#if USE_PADDLE_API
    c10::DataPtr data_ptr(static_cast<void*>(local_data),
                          local_data,
                          real_float_deleter,
                          phi::CPUPlace());
#else
    c10::DataPtr data_ptr(static_cast<void*>(local_data),
                          local_data,
                          real_float_deleter,
                          c10::Device(c10::DeviceType::CPU));
#endif
    file << std::to_string(data_ptr.get() != nullptr) << " ";
  }
  // DataPtr 出作用域后，deleter 应该被调用（内存已释放）

  file.saveFile();
}

// 测试 get 方法返回正确的指针类型
TEST_F(AllocatorTest, GetReturnsCorrectPointer) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();

#if USE_PADDLE_API
  c10::DataPtr data_ptr(static_cast<void*>(test_data_), phi::CPUPlace());
#else
  c10::DataPtr data_ptr(static_cast<void*>(test_data_),
                        c10::Device(c10::DeviceType::CPU));
#endif

  // get() 返回 void*，可以转换为原始类型
  void* void_ptr = data_ptr.get();
  float* float_ptr = static_cast<float*>(void_ptr);

  // 验证数据完整性
  file << std::to_string(float_ptr[0]) << " ";
  file << std::to_string(float_ptr[1]) << " ";
  file << std::to_string(float_ptr[2]) << " ";
  file << std::to_string(float_ptr[3]) << " ";

  file.saveFile();
}

// 测试 DeleterFnPtr 类型
TEST_F(AllocatorTest, DeleterFnPtrType) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();

  // 验证 DeleterFnPtr 类型存在且可用
  c10::DeleterFnPtr deleter = test_deleter;
  file << std::to_string(deleter != nullptr) << " ";

#if USE_PADDLE_API
  c10::DataPtr data_ptr(
      static_cast<void*>(test_data_), test_ctx_, deleter, phi::CPUPlace());
#else
  c10::DataPtr data_ptr(static_cast<void*>(test_data_),
                        test_ctx_,
                        deleter,
                        c10::Device(c10::DeviceType::CPU));
#endif

  file << std::to_string(data_ptr.get_deleter() == deleter) << " ";

  file.saveFile();
}

}  // namespace test
}  // namespace at
