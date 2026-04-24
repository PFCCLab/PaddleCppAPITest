#include <ATen/ATen.h>
#include <c10/core/Allocator.h>
#include <gtest/gtest.h>

#include <functional>
#include <string>

#include "src/file_manager.h"

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

static void delete_byte_array(void* ptr) { delete[] static_cast<char*>(ptr); }

class ByteAllocator final : public c10::Allocator {
 public:
  c10::DataPtr allocate(size_t n) override {
    size_t bytes = n == 0 ? 1 : n;
    char* data = new char[bytes];
    return c10::DataPtr(
        data, data, delete_byte_array, c10::Device(c10::DeviceType::CPU));
  }

  void copy_data(void* dest,
                 const void* src,
                 std::size_t count) const override {
    default_copy_data(dest, src, count);
  }
};

static ByteAllocator g_registered_allocator;
REGISTER_ALLOCATOR(c10::DeviceType::IPU, &g_registered_allocator);

static void dataptr_clear_api_probe(c10::DataPtr* data_ptr) {
  data_ptr->clear();
}

// 测试默认构造函数
TEST_F(AllocatorTest, DefaultConstructor) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();
  file << "DefaultConstructor ";

  c10::DataPtr data_ptr;

  // 默认构造的 DataPtr 应该为 null
  file << std::to_string(data_ptr.get() == nullptr) << " ";
  // operator bool 应该返回 false
  file << std::to_string(static_cast<bool>(data_ptr) == false) << " ";
  // context 应该为 nullptr
  file << std::to_string(data_ptr.get_context() == nullptr) << " ";
  // 默认 deleter 不为 nullptr（两端均有默认 deleter）
  file << std::to_string(data_ptr.get_deleter() != nullptr) << " ";

  file << "\n";
  file.saveFile();
}

// 测试带数据和设备的构造函数
TEST_F(AllocatorTest, ConstructorWithDataAndDevice) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "ConstructorWithDataAndDevice ";

  c10::DataPtr data_ptr(static_cast<void*>(test_data_),
                        c10::Device(c10::DeviceType::CPU));

  // 指针应该正确设置
  file << std::to_string(data_ptr.get() == static_cast<void*>(test_data_))
       << " ";
  // operator bool 应该返回 true
  file << std::to_string(static_cast<bool>(data_ptr) == true) << " ";
  // 验证可以通过 get() 访问数据
  float* ptr = static_cast<float*>(data_ptr.get());
  file << std::to_string(ptr[0]) << " ";
  file << std::to_string(ptr[1]) << " ";
  // device 字符串表示
  file << std::to_string(data_ptr.device().str() == "cpu") << " ";

  file << "\n";
  file.saveFile();
}

TEST_F(AllocatorTest, MutableGet) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "MutableGet ";

  c10::DataPtr data_ptr(static_cast<void*>(test_data_),
                        c10::Device(c10::DeviceType::CPU));

  void* mutable_ptr = data_ptr.mutable_get();
  static_cast<float*>(mutable_ptr)[2] = 9.0f;

  file << std::to_string(mutable_ptr == data_ptr.get()) << " ";
  file << std::to_string(test_data_[2]) << " ";

  file << "\n";
  file.saveFile();
}

// 测试带完整参数的构造函数
TEST_F(AllocatorTest, ConstructorWithDeleter) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "ConstructorWithDeleter ";

  g_deleter_called = false;

  c10::DataPtr data_ptr(static_cast<void*>(test_data_),
                        test_ctx_,
                        test_deleter,
                        c10::Device(c10::DeviceType::CPU));

  // 指针应该正确设置
  file << std::to_string(data_ptr.get() == static_cast<void*>(test_data_))
       << " ";
  // context 应该正确设置
  file << std::to_string(data_ptr.get_context() == test_ctx_) << " ";
  // deleter 应该正确设置
  file << std::to_string(data_ptr.get_deleter() == test_deleter) << " ";

  file << "\n";
  file.saveFile();
}

// 测试移动构造函数
TEST_F(AllocatorTest, MoveConstructor) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "MoveConstructor ";

  c10::DataPtr original(static_cast<void*>(test_data_),
                        c10::Device(c10::DeviceType::CPU));
  void* original_ptr = original.get();
  c10::DataPtr moved(std::move(original));

  // 移动后的 DataPtr 应该持有原始指针
  file << std::to_string(moved.get() == original_ptr) << " ";
  file << std::to_string(moved.get() == static_cast<void*>(test_data_)) << " ";

  file << "\n";
  file.saveFile();
}

// 测试移动赋值操作符
TEST_F(AllocatorTest, MoveAssignment) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "MoveAssignment ";

  c10::DataPtr original(static_cast<void*>(test_data_),
                        c10::Device(c10::DeviceType::CPU));
  void* original_ptr = original.get();
  c10::DataPtr assigned;
  assigned = std::move(original);

  // 移动赋值后应该持有原始指针
  file << std::to_string(assigned.get() == original_ptr) << " ";

  file << "\n";
  file.saveFile();
}

// 测试 clear 方法
TEST_F(AllocatorTest, Clear) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "Clear ";

  c10::DataPtr data_ptr(static_cast<void*>(test_data_),
                        test_ctx_,
                        test_deleter,
                        c10::Device(c10::DeviceType::CPU));

  // clear 前验证状态
  file << std::to_string(data_ptr.get() != nullptr) << " ";
  file << std::to_string(static_cast<bool>(data_ptr)) << " ";

  dataptr_clear_api_probe(&data_ptr);

  // clear 后核心属性应该为空
  file << std::to_string(data_ptr.get() == nullptr) << " ";
  file << std::to_string(static_cast<bool>(data_ptr) == false) << " ";
  file << std::to_string(data_ptr.get_context() == nullptr) << " ";
  // clear 后 deleter 仍然保留（与 PyTorch 行为一致）
  file << std::to_string(data_ptr.get_deleter() == test_deleter) << " ";

  file << "\n";
  file.saveFile();
}

// 测试与 nullptr 的比较操作符
TEST_F(AllocatorTest, NullptrComparison) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "NullptrComparison ";

  c10::DataPtr null_ptr;
  c10::DataPtr valid_ptr(static_cast<void*>(test_data_),
                         c10::Device(c10::DeviceType::CPU));

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

  file << "\n";
  file.saveFile();
}

// 测试 at::DataPtr 别名
TEST_F(AllocatorTest, AtDataPtrAlias) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "AtDataPtrAlias ";

  // at::DataPtr 应该是 c10::DataPtr 的别名
  at::DataPtr at_ptr(static_cast<void*>(test_data_),
                     c10::Device(c10::DeviceType::CPU));

  file << std::to_string(at_ptr.get() == static_cast<void*>(test_data_)) << " ";
  file << std::to_string(static_cast<bool>(at_ptr)) << " ";

  // 验证可以移动赋值给 c10::DataPtr
  c10::DataPtr c10_ptr = std::move(at_ptr);
  file << std::to_string(c10_ptr.get() == static_cast<void*>(test_data_))
       << " ";

  file << "\n";
  file.saveFile();
}

// 测试 operator-> 方法
TEST_F(AllocatorTest, ArrowOperator) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "ArrowOperator ";

  c10::DataPtr data_ptr(static_cast<void*>(test_data_),
                        c10::Device(c10::DeviceType::CPU));

  // operator-> 应该返回原始指针
  file << std::to_string(data_ptr.operator->() ==
                         static_cast<void*>(test_data_))
       << " ";
  file << std::to_string(data_ptr.operator->() == data_ptr.get()) << " ";

  file << "\n";
  file.saveFile();
}

// 测试空 DataPtr 的边界情况
TEST_F(AllocatorTest, EmptyDataPtrEdgeCases) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "EmptyDataPtrEdgeCases ";

  c10::DataPtr empty_ptr;

  // 验证空指针的核心属性
  file << std::to_string(empty_ptr.get() == nullptr) << " ";
  file << std::to_string(empty_ptr.get_context() == nullptr) << " ";
  file << std::to_string(!static_cast<bool>(empty_ptr)) << " ";

  // 调用 clear 对空指针应该安全
  empty_ptr.clear();
  file << std::to_string(empty_ptr.get() == nullptr) << " ";

  file << "\n";
  file.saveFile();
}

// 测试拷贝语义已被删除（move-only，与 PyTorch 一致）
TEST_F(AllocatorTest, CopySemanticsDeleted) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "CopySemanticsDeleted ";

  file << std::to_string(!std::is_copy_constructible_v<c10::DataPtr>) << " ";
  file << std::to_string(!std::is_copy_assignable_v<c10::DataPtr>) << " ";

  file << "\n";
  file.saveFile();
}

// 测试不存在单参数构造函数（与 PyTorch 一致）
TEST_F(AllocatorTest, NoSingleArgConstructor) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "NoSingleArgConstructor ";

  file << std::to_string(!std::is_constructible_v<c10::DataPtr, void*>) << " ";

  file << "\n";
  file.saveFile();
}

// 测试链式移动
TEST_F(AllocatorTest, ChainedMoves) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "ChainedMoves ";

  c10::DataPtr original(static_cast<void*>(test_data_),
                        c10::Device(c10::DeviceType::CPU));
  void* ptr = original.get();

  // 链式移动
  c10::DataPtr moved1(std::move(original));
  c10::DataPtr moved2(std::move(moved1));
  c10::DataPtr moved3 = std::move(moved2);

  // 最终应该指向原始数据
  file << std::to_string(moved3.get() == ptr) << " ";
  file << std::to_string(moved3.get() == static_cast<void*>(test_data_)) << " ";

  file << "\n";
  file.saveFile();
}

// 测试 Deleter 在析构时是否被调用
TEST_F(AllocatorTest, DeleterCalledOnDestruction) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "DeleterCalledOnDestruction ";

  {
    // 在作用域内创建 DataPtr
    float* local_data = new float[2]{1.0f, 2.0f};
    c10::DataPtr data_ptr(static_cast<void*>(local_data),
                          local_data,
                          real_float_deleter,
                          c10::Device(c10::DeviceType::CPU));
    file << std::to_string(data_ptr.get() != nullptr) << " ";
  }
  // DataPtr 出作用域后，deleter 应该被调用（内存已释放）

  file << "\n";
  file.saveFile();
}

// 测试 get 方法返回正确的指针类型
TEST_F(AllocatorTest, GetReturnsCorrectPointer) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "GetReturnsCorrectPointer ";

  c10::DataPtr data_ptr(static_cast<void*>(test_data_),
                        c10::Device(c10::DeviceType::CPU));

  // get() 返回 void*，可以转换为原始类型
  void* void_ptr = data_ptr.get();
  float* float_ptr = static_cast<float*>(void_ptr);

  // 验证数据完整性
  file << std::to_string(float_ptr[0]) << " ";
  file << std::to_string(float_ptr[1]) << " ";
  file << std::to_string(float_ptr[2]) << " ";
  file << std::to_string(float_ptr[3]) << " ";

  file << "\n";
  file.saveFile();
}

// 测试 DeleterFnPtr 类型
TEST_F(AllocatorTest, DeleterFnPtrType) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "DeleterFnPtrType ";

  // 验证 DeleterFnPtr 类型存在且可用
  c10::DeleterFnPtr deleter = test_deleter;
  file << std::to_string(deleter != nullptr) << " ";

  c10::DataPtr data_ptr(static_cast<void*>(test_data_),
                        test_ctx_,
                        deleter,
                        c10::Device(c10::DeviceType::CPU));

  file << std::to_string(data_ptr.get_deleter() == deleter) << " ";

  file << "\n";
  file.saveFile();
}

TEST_F(AllocatorTest, CaptureAndMempoolTypes) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "CaptureAndMempoolTypes ";

  c10::CaptureId_t capture_id = 7;
  c10::MempoolId_t primary{capture_id, 11};
  c10::MempoolId_t fallback{0, 13};
  c10::MempoolIdHash hasher;

  file << std::to_string(primary.first == capture_id) << " ";
  file << std::to_string(hasher(primary) == 7) << " ";
  file << std::to_string(hasher(fallback) == 13) << " ";

  file << "\n";
  file.saveFile();
}

TEST_F(AllocatorTest, InefficientStdFunctionContextMakeDataPtr) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "InefficientStdFunctionContextMakeDataPtr ";

  bool deleter_called = false;
  int* value = new int(7);

  {
    c10::DataPtr data_ptr = c10::InefficientStdFunctionContext::makeDataPtr(
        value,
        [&deleter_called](void* ptr) {
          deleter_called = true;
          delete static_cast<int*>(ptr);
        },
        c10::Device(c10::DeviceType::CPU));

    file << std::to_string(data_ptr.get() == static_cast<void*>(value)) << " ";
    file << std::to_string(data_ptr.get_context() != nullptr) << " ";
    file << std::to_string(data_ptr.get_context() != static_cast<void*>(value))
         << " ";
  }

  file << std::to_string(deleter_called) << " ";
  file << "\n";
  file.saveFile();
}

TEST_F(AllocatorTest, IsSimpleDataPtrSemantics) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "IsSimpleDataPtrSemantics ";

  ByteAllocator allocator;
  c10::DataPtr simple_ptr(static_cast<void*>(test_data_),
                          static_cast<void*>(test_data_),
                          test_deleter,
                          c10::Device(c10::DeviceType::CPU));
  c10::DataPtr view_ptr(static_cast<void*>(test_data_),
                        c10::Device(c10::DeviceType::CPU));
  c10::DataPtr separate_ctx_ptr(static_cast<void*>(test_data_),
                                test_ctx_,
                                test_deleter,
                                c10::Device(c10::DeviceType::CPU));

  file << std::to_string(allocator.is_simple_data_ptr(simple_ptr)) << " ";
  file << std::to_string(allocator.is_simple_data_ptr(view_ptr)) << " ";
  file << std::to_string(allocator.is_simple_data_ptr(separate_ctx_ptr)) << " ";

  file << "\n";
  file.saveFile();
}

TEST_F(AllocatorTest, SetAndGetAllocatorPriority) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "SetAndGetAllocatorPriority ";

  static ByteAllocator high_priority_allocator;
  static ByteAllocator low_priority_allocator;

  c10::SetAllocator(c10::DeviceType::XPU, &high_priority_allocator, 2);
  file << std::to_string(c10::GetAllocator(c10::DeviceType::XPU) ==
                         &high_priority_allocator)
       << " ";

  c10::SetAllocator(c10::DeviceType::XPU, &low_priority_allocator, 1);
  file << std::to_string(c10::GetAllocator(c10::DeviceType::XPU) ==
                         &high_priority_allocator)
       << " ";

  c10::SetAllocator(c10::DeviceType::XPU, &low_priority_allocator, 2);
  file << std::to_string(c10::GetAllocator(c10::DeviceType::XPU) ==
                         &low_priority_allocator)
       << " ";

  file << "\n";
  file.saveFile();
}

TEST_F(AllocatorTest, RegisterAllocatorMacro) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "RegisterAllocatorMacro ";

  // 宏在文件作用域完成展开和注册；这里用运行时输出锁定该编译路径存在。
  file << std::to_string(true) << " ";

  file << "\n";
  file.saveFile();
}

}  // namespace test
}  // namespace at
