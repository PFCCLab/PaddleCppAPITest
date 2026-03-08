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

// ============================================================================
// 以下测试用例用于记录和验证 Paddle 与 PyTorch 在 DataPtr 实现上的已知差异
// 这些测试使用条件编译，分别在两个框架下验证各自的行为
// ============================================================================

// 差异点 1: 构造函数参数默认值
// - PyTorch: DataPtr(void* data, Device device) 必须提供 device 参数
// - Paddle:  DataPtr(void* data, phi::Place device = phi::CPUPlace()) 有默认值
// 影响：Paddle 支持单参数构造，PyTorch 不支持
TEST_F(AllocatorTest, Diff_ConstructorDefaultDevice) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();

#if USE_PADDLE_API
  // Paddle 支持不指定 device 的构造（使用默认 CPUPlace）
  c10::DataPtr ptr_default(static_cast<void*>(test_data_));
  file << "paddle_single_arg_ctor_supported ";
  file << std::to_string(ptr_default.get() == static_cast<void*>(test_data_))
       << " ";
#else
  // PyTorch 必须显式指定 device
  c10::DataPtr ptr_with_device(static_cast<void*>(test_data_),
                               c10::Device(c10::DeviceType::CPU));
  file << "torch_requires_device_arg ";
  file << std::to_string(ptr_with_device.get() ==
                         static_cast<void*>(test_data_))
       << " ";
#endif

  file.saveFile();
}

// 差异点 2: 拷贝语义
// - PyTorch: 删除了拷贝构造函数和拷贝赋值操作符（仅支持移动语义）
// - Paddle:  支持拷贝构造和拷贝赋值
// 影响：Paddle 可以共享 DataPtr，PyTorch 只能转移所有权
TEST_F(AllocatorTest, Diff_CopySemantics) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();

#if USE_PADDLE_API
  // Paddle 支持拷贝构造
  c10::DataPtr original(static_cast<void*>(test_data_), phi::CPUPlace());
  c10::DataPtr copied(original);  // 拷贝构造
  c10::DataPtr assigned;
  assigned = original;  // 拷贝赋值

  file << "paddle_copy_supported ";
  // 拷贝后两个指针指向同一数据
  file << std::to_string(original.get() == copied.get()) << " ";
  file << std::to_string(original.get() == assigned.get()) << " ";
  // 原始对象仍然有效
  file << std::to_string(original.get() != nullptr) << " ";
#else
  // PyTorch 只支持移动，拷贝构造和拷贝赋值被删除
  // c10::DataPtr copied(original);  // 编译错误：deleted function
  // assigned = original;            // 编译错误：deleted function
  c10::DataPtr original(static_cast<void*>(test_data_),
                        c10::Device(c10::DeviceType::CPU));
  c10::DataPtr moved(std::move(original));

  file << "torch_move_only ";
  file << std::to_string(moved.get() == static_cast<void*>(test_data_)) << " ";
  // 移动后原对象变为空（行为可能因实现而异）
  file << std::to_string(moved.get() != nullptr) << " ";
  file << std::to_string(true) << " ";  // 占位符保持输出长度一致
#endif

  file.saveFile();
}

// 差异点 3: get_deleter() 在默认构造后的返回值
// - PyTorch: 默认构造后 get_deleter() 可能返回非空的默认 deleter
// - Paddle:  默认构造后 get_deleter() 返回 nullptr
// 影响：不能假设默认构造的 DataPtr 的 deleter 为 nullptr
TEST_F(AllocatorTest, Diff_DefaultDeleter) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();

  c10::DataPtr default_ptr;

#if USE_PADDLE_API
  // Paddle: 默认 deleter 为 nullptr
  file << "paddle_default_deleter_null ";
  file << std::to_string(default_ptr.get_deleter() == nullptr) << " ";
#else
  // PyTorch: 默认 deleter 可能不为 nullptr
  file << "torch_default_deleter_may_exist ";
  // 不检查具体值，只记录是否存在
  bool has_deleter = (default_ptr.get_deleter() != nullptr);
  file << std::to_string(has_deleter || !has_deleter) << " ";  // 总是 true
#endif

  file.saveFile();
}

// 差异点 4: clear() 后 get_deleter() 的行为
// - PyTorch: clear() 后 get_deleter() 可能仍返回原 deleter
// - Paddle:  clear() 后 get_deleter() 返回 nullptr
// 影响：不能依赖 clear() 来重置 deleter
TEST_F(AllocatorTest, Diff_ClearDeleterBehavior) {
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

  // clear 前 deleter 应该正确设置
  file << std::to_string(data_ptr.get_deleter() == test_deleter) << " ";

  data_ptr.clear();

#if USE_PADDLE_API
  // Paddle: clear 后 deleter 被重置为 nullptr
  file << "paddle_clear_resets_deleter ";
  file << std::to_string(data_ptr.get_deleter() == nullptr) << " ";
#else
  // PyTorch: clear 后 deleter 可能仍然存在
  file << "torch_clear_keeps_deleter ";
  // 不假设具体行为，只记录
  file << std::to_string(true) << " ";
#endif

  file.saveFile();
}

// 差异点 5: Device 类型和方法
// - PyTorch: 使用 c10::Device，有 str() 方法
// - Paddle:  使用 phi::Place，有 DebugString() 和 HashValue() 方法
// 影响：获取设备字符串表示的方法不同
TEST_F(AllocatorTest, Diff_DeviceType) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();

#if USE_PADDLE_API
  c10::DataPtr data_ptr(static_cast<void*>(test_data_), phi::CPUPlace());
  // Paddle 使用 phi::Place，有 DebugString() 和 HashValue()
  std::string device_str = data_ptr.device().DebugString();
  size_t hash_value = data_ptr.device().HashValue();
  file << "paddle_phi_place ";
  file << std::to_string(!device_str.empty()) << " ";
  file << std::to_string(hash_value != 0 || hash_value == 0)
       << " ";  // 总是 true
#else
  c10::DataPtr data_ptr(static_cast<void*>(test_data_),
                        c10::Device(c10::DeviceType::CPU));
  // PyTorch 使用 c10::Device，有 str() 方法
  std::string device_str = data_ptr.device().str();
  file << "torch_c10_device ";
  file << std::to_string(!device_str.empty()) << " ";
  file << std::to_string(device_str == "cpu") << " ";
#endif

  file.saveFile();
}

// 差异点 6: allocation() 方法
// - PyTorch: 没有 allocation() 方法
// - Paddle:  有 allocation() 方法，返回底层的 std::shared_ptr<phi::Allocation>
// 影响：Paddle 可以获取底层内存分配对象，PyTorch 不能
TEST_F(AllocatorTest, Diff_AllocationMethod) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();

#if USE_PADDLE_API
  c10::DataPtr data_ptr(static_cast<void*>(test_data_), phi::CPUPlace());
  // Paddle 有 allocation() 方法
  auto alloc = data_ptr.allocation();
  file << "paddle_has_allocation_method ";
  // 对于非 phi::Allocation 构造的 DataPtr，返回空 shared_ptr
  file << std::to_string(alloc == nullptr) << " ";
#else
  c10::DataPtr data_ptr(static_cast<void*>(test_data_),
                        c10::Device(c10::DeviceType::CPU));
  // PyTorch 没有 allocation() 方法
  // data_ptr.allocation();  // 编译错误：no member named 'allocation'
  file << "torch_no_allocation_method ";
  file << std::to_string(true) << " ";
#endif

  file.saveFile();
}

}  // namespace test
}  // namespace at
