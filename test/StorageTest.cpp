#include <ATen/ATen.h>
#include <ATen/core/Tensor.h>
#include <ATen/ops/ones.h>
#include <gtest/gtest.h>
#include <torch/all.h>

#include <string>
#include <vector>

#include "../src/file_manager.h"

extern paddle_api_test::ThreadSafeParam g_custom_param;

namespace at {
namespace test {

using paddle_api_test::FileManerger;
using paddle_api_test::ThreadSafeParam;

class StorageTest : public ::testing::Test {
 protected:
  void SetUp() override {
    std::vector<int64_t> shape = {2, 3, 4};
    tensor = at::ones(shape, at::kFloat);
  }

  at::Tensor tensor;
};

// 测试 storage
TEST_F(StorageTest, Storage) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();

  c10::Storage storage = tensor.storage();
  file << std::to_string(storage.data_ptr().get() != nullptr) << " ";
  file.saveFile();
}

// 测试 storage_offset
TEST_F(StorageTest, StorageOffset) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();

  int64_t offset = tensor.storage_offset();
  file << std::to_string(offset) << " ";
  file.saveFile();
}

// 测试 has_storage
TEST_F(StorageTest, HasStorage) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();

  file << std::to_string(tensor.has_storage()) << " ";
  file.saveFile();
}

// 测试 storage nbytes
TEST_F(StorageTest, StorageNbytes) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();

  c10::Storage storage = tensor.storage();
  // 2*3*4 = 24 个 float 元素，每个 4 字节
  size_t expected_nbytes = 24 * sizeof(float);
  file << std::to_string(storage.nbytes()) << " ";
  file << std::to_string(expected_nbytes) << " ";
  file << std::to_string(storage.nbytes() >= expected_nbytes) << " ";
  file.saveFile();
}

// 测试 sliced tensor 的 storage_offset
TEST_F(StorageTest, SlicedTensorStorageOffset) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();

  // 对 tensor 进行切片操作
  at::Tensor sliced = tensor.slice(0, 1, 2);  // 在第0维取索引1到2
  // 切片后的 tensor 应该共享同一个 storage
  file << std::to_string(sliced.storage().data_ptr().get() ==
                         tensor.storage().data_ptr().get())
       << " ";
  // 切片后的 offset 应该大于 0
  file << std::to_string(sliced.storage_offset()) << " ";
  file << std::to_string(sliced.storage_offset() > 0) << " ";
  file.saveFile();
}

// 测试 storage data_ptr
TEST_F(StorageTest, StorageDataPtr) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();

  c10::Storage storage = tensor.storage();
  void* storage_ptr = storage.data_ptr().get();
  void* tensor_ptr = tensor.data_ptr();
  // 对于 offset 为 0 的 tensor，两个指针应该相同
  file << std::to_string(storage_ptr == tensor_ptr) << " ";
  file.saveFile();
}

}  // namespace test
}  // namespace at
