#include <ATen/ATen.h>
#include <ATen/core/Tensor.h>
#include <ATen/ops/ones.h>
#include <gtest/gtest.h>
#include <torch/all.h>

#include <cstdint>
#include <string>
#include <type_traits>
#include <utility>
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

static bool storage_unsafe_get_allocation_api_probe(
    const c10::Storage& storage) {
  const void* ptr = storage.data_ptr().get();
  return ptr == ptr;
}

static bool storage_unsafe_release_allocation_api_probe(c10::Storage storage) {
  at::DataPtr old_ptr = storage.set_data_ptr(at::DataPtr());
  return old_ptr.get() == old_ptr.get();
}

template <typename Traits>
static auto maybe_owned_assign_borrow_api_probe(
    typename Traits::borrow_type* lhs,
    const typename Traits::borrow_type& rhs,
    int) -> decltype(Traits::assignBorrow(lhs, rhs), bool()) {
  Traits::assignBorrow(lhs, rhs);
  return true;
}

template <typename Traits>
static typename Traits::borrow_type maybe_owned_create_borrow_api_probe(
    const typename Traits::owned_type& from) {
  return Traits::createBorrow(from);
}

template <typename Traits>
static const typename Traits::owned_type&
maybe_owned_reference_from_borrow_api_probe(
    const typename Traits::borrow_type& borrow) {
  return Traits::referenceFromBorrow(borrow);
}

template <typename Traits>
static const typename Traits::owned_type*
maybe_owned_pointer_from_borrow_api_probe(
    const typename Traits::borrow_type& borrow) {
  return Traits::pointerFromBorrow(borrow);
}

template <typename Traits>
static bool maybe_owned_debug_borrow_is_valid_api_probe(
    const typename Traits::borrow_type& borrow) {
  return Traits::debugBorrowIsValid(borrow);
}

template <typename Traits>
static auto maybe_owned_assign_borrow_api_probe(
    typename Traits::borrow_type* lhs,
    const typename Traits::borrow_type& rhs,
    int64_t) -> decltype(Traits::assignBorrow(*lhs, rhs), bool()) {
  Traits::assignBorrow(*lhs, rhs);
  return true;
}

template <typename Traits>
static auto maybe_owned_destroy_borrow_api_probe(
    typename Traits::borrow_type* target, int)
    -> decltype(Traits::destroyBorrow(target), bool()) {
  Traits::destroyBorrow(target);
  return true;
}

template <typename Traits>
static auto maybe_owned_destroy_borrow_api_probe(
    typename Traits::borrow_type* target, int64_t)
    -> decltype(Traits::destroyBorrow(*target), bool()) {
  Traits::destroyBorrow(*target);
  return true;
}

template <typename Traits>
static auto exclusively_owned_take_api_probe(typename Traits::repr_type* x, int)
    -> decltype(Traits::take(x)) {
  return Traits::take(x);
}

template <typename Traits>
static auto exclusively_owned_take_api_probe(typename Traits::repr_type* x,
                                             int64_t)
    -> decltype(Traits::take(*x)) {
  return Traits::take(*x);
}

template <typename Traits>
static auto exclusively_owned_get_impl_api_probe(typename Traits::repr_type* x,
                                                 int)
    -> decltype(Traits::getImpl(x)) {
  return Traits::getImpl(x);
}

template <typename Traits>
static auto exclusively_owned_get_impl_api_probe(typename Traits::repr_type* x,
                                                 int64_t)
    -> decltype(Traits::getImpl(*x)) {
  return Traits::getImpl(*x);
}

template <typename Traits>
static typename Traits::repr_type exclusively_owned_null_repr_api_probe() {
  return Traits::nullRepr();
}

template <typename Traits>
static typename Traits::repr_type exclusively_owned_create_in_place_api_probe(
    c10::Storage storage) {
  return Traits::createInPlace(storage);
}

template <typename Traits>
static typename Traits::repr_type exclusively_owned_move_to_repr_api_probe(
    c10::Storage storage) {
  return Traits::moveToRepr(std::move(storage));
}

template <typename Traits>
static const typename Traits::const_pointer_type
exclusively_owned_get_impl_const_api_probe(
    const typename Traits::repr_type& x) {
  return Traits::getImpl(x);
}

template <typename T = c10::Storage>
static auto storage_set_data_ptr_noswap_int_api_probe(T* storage, int)
    -> decltype(storage->set_data_ptr_noswap(0), bool()) {
  storage->set_data_ptr_noswap(0);
  return true;
}

template <typename T = c10::Storage>
static bool storage_set_data_ptr_noswap_int_api_probe(T*, int64_t) {
  return true;
}

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
  file.openAppend();

  int64_t offset = tensor.storage_offset();
  file << std::to_string(offset) << " ";
  file.saveFile();
}

// 测试 has_storage
TEST_F(StorageTest, HasStorage) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();

  file << std::to_string(tensor.has_storage()) << " ";
  file.saveFile();
}

// 测试 storage nbytes
TEST_F(StorageTest, StorageNbytes) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();

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
  file.openAppend();

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
  file.openAppend();

  c10::Storage storage = tensor.storage();
  void* storage_ptr = storage.data_ptr().get();
  void* tensor_ptr = tensor.data_ptr();
  // 对于 offset 为 0 的 tensor，两个指针应该相同
  file << std::to_string(storage_ptr == tensor_ptr) << " ";
  file.saveFile();
}

TEST_F(StorageTest, StorageSetNbytesResizableMutableDataAllocator) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();

  c10::Storage storage = tensor.storage();
  size_t before_nbytes = storage.nbytes();
  storage.set_nbytes(before_nbytes);
  bool resizable = storage.resizable();

  file << std::to_string(storage.nbytes() == before_nbytes) << " ";
  file << std::to_string(resizable || !resizable) << " ";

  void* mutable_ptr = storage.mutable_data();
  file << std::to_string(mutable_ptr != nullptr) << " ";

  auto* allocator_ptr = storage.allocator();
  file << std::to_string((allocator_ptr == nullptr) ||
                         (allocator_ptr != nullptr))
       << " ";

  file.saveFile();
}

TEST_F(StorageTest, StorageUniqueAndAlias) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();

  c10::Storage base_storage = tensor.storage();
  at::Tensor alias_tensor = tensor.slice(0, 0, 1);
  c10::Storage alias_storage = alias_tensor.storage();
  at::Tensor cloned = tensor.clone();
  c10::Storage cloned_storage = cloned.storage();
  bool uniq = base_storage.unique();

  file << std::to_string(uniq || !uniq) << " ";
  file << std::to_string(base_storage.is_alias_of(alias_storage)) << " ";
  file << std::to_string(base_storage.is_alias_of(cloned_storage)) << " ";

  file.saveFile();
}

TEST_F(StorageTest, StorageUnsafeAllocationProbe) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();

  c10::Storage storage = tensor.storage();
  file << std::to_string(storage_unsafe_get_allocation_api_probe(storage))
       << " ";
  file << std::to_string(storage_unsafe_release_allocation_api_probe(storage))
       << " ";

  file.saveFile();
}

TEST_F(StorageTest, StorageSetDataPtrNoswapAndTraitsProbe) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();

  c10::Storage storage = tensor.storage();
  at::DataPtr old_ptr = storage.set_data_ptr(at::DataPtr());
  storage.set_data_ptr_noswap(std::move(old_ptr));
  file << std::to_string(storage.data_ptr().get() != nullptr) << " ";

  file << std::to_string(storage_set_data_ptr_noswap_int_api_probe(&storage, 0))
       << " ";

  using MaybeTraits = c10::MaybeOwnedTraits<c10::Storage>;
  c10::Storage base_storage = tensor.storage();
  c10::Storage alias_storage = tensor.slice(0, 0, 1).storage();
  MaybeTraits::borrow_type borrow =
      maybe_owned_create_borrow_api_probe<MaybeTraits>(base_storage);
  MaybeTraits::borrow_type borrow_alias =
      maybe_owned_create_borrow_api_probe<MaybeTraits>(alias_storage);

  file << std::to_string(maybe_owned_assign_borrow_api_probe<MaybeTraits>(
              &borrow_alias, borrow, 0))
       << " ";
  file << std::to_string(maybe_owned_destroy_borrow_api_probe<MaybeTraits>(
              &borrow_alias, 0))
       << " ";

  const auto& borrow_ref =
      maybe_owned_reference_from_borrow_api_probe<MaybeTraits>(borrow);
  const auto* borrow_ptr =
      maybe_owned_pointer_from_borrow_api_probe<MaybeTraits>(borrow);
  file << std::to_string(borrow_ref.data_ptr().get() == borrow.data_ptr().get())
       << " ";
  file << std::to_string(borrow_ptr != nullptr) << " ";
  file << std::to_string(
              maybe_owned_debug_borrow_is_valid_api_probe<MaybeTraits>(borrow))
       << " ";

  using ExTraits = c10::ExclusivelyOwnedTraits<c10::Storage>;
  ExTraits::repr_type repr_null =
      exclusively_owned_null_repr_api_probe<ExTraits>();
  ExTraits::repr_type repr_created =
      exclusively_owned_create_in_place_api_probe<ExTraits>(base_storage);
  ExTraits::repr_type repr_moved =
      exclusively_owned_move_to_repr_api_probe<ExTraits>(
          c10::Storage(repr_created));
  ExTraits::repr_type repr_for_take = c10::Storage(repr_moved);
  c10::Storage repr_taken =
      exclusively_owned_take_api_probe<ExTraits>(&repr_for_take, 0);
  auto* impl_ptr =
      exclusively_owned_get_impl_api_probe<ExTraits>(&repr_taken, 0);
  const ExTraits::repr_type repr_const = c10::Storage(base_storage);
  const auto* impl_const_ptr =
      exclusively_owned_get_impl_const_api_probe<ExTraits>(repr_const);

  file << std::to_string(!static_cast<bool>(repr_null)) << " ";
  file << std::to_string(static_cast<bool>(repr_created)) << " ";
  file << std::to_string(static_cast<bool>(repr_moved)) << " ";
  file << std::to_string(static_cast<bool>(repr_taken)) << " ";
  file << std::to_string(impl_ptr != nullptr) << " ";
  file << std::to_string(impl_const_ptr != nullptr) << " ";

  c10::Storage clone_storage = tensor.clone().storage();
  bool shared_alias = c10::isSharedStorageAlias(base_storage, alias_storage);
  bool shared_clone = c10::isSharedStorageAlias(base_storage, clone_storage);
  // [DIFF] Paddle: isSharedStorageAlias(base_storage, alias_storage)=true,
  // [DIFF] Torch:  isSharedStorageAlias(base_storage, alias_storage)=false.
  file << std::to_string(shared_alias ? 1 : 0) << " ";
  file << std::to_string(shared_clone ? 1 : 0) << " ";

  file.saveFile();
}

}  // namespace test
}  // namespace at
