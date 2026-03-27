#include <ATen/ATen.h>
#if __has_include(<ATen/core/List.h>)
#include <ATen/core/List.h>
#elif __has_include(<c10/core/List.h>)
#include <c10/core/List.h>
#endif
#include <gtest/gtest.h>

#include <string>
#include <utility>

#include "../src/file_manager.h"

extern paddle_api_test::ThreadSafeParam g_custom_param;

namespace at {
namespace test {

using paddle_api_test::FileManerger;

static void list_clear_api_probe(c10::List<int64_t>* list) { list->clear(); }

static void list_reserve_api_probe(c10::List<int64_t>* list,
                                   c10::List<int64_t>::size_type new_cap) {
  list->reserve(new_cap);
}

static c10::List<int64_t>::size_type list_capacity_api_probe(
    const c10::List<int64_t>& list) {
  return list.vec().capacity();
}

static void list_push_back_lvalue_api_probe(c10::List<int64_t>* list,
                                            const int64_t& value) {
  list->push_back(value);
}

static void list_push_back_rvalue_api_probe(c10::List<int64_t>* list,
                                            int64_t&& value) {
  list->push_back(std::move(value));
}

template <typename... Args>
static void list_emplace_back_api_probe(c10::List<int64_t>* list,
                                        Args&&... args) {
  list->emplace_back(std::forward<Args>(args)...);
}

static void list_pop_back_api_probe(c10::List<int64_t>* list) {
  list->pop_back();
}

static void list_resize_count_api_probe(c10::List<int64_t>* list,
                                        c10::List<int64_t>::size_type count) {
  list->resize(count);
}

static void list_resize_with_value_api_probe(
    c10::List<int64_t>* list,
    c10::List<int64_t>::size_type count,
    const int64_t& value) {
  list->resize(count, value);
}

class ListCompatTest : public ::testing::Test {
 protected:
  void SetUp() override {}
};

TEST_F(ListCompatTest, ListReserveAndCapacity) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();
  file << "ListReserveAndCapacity ";

  c10::List<int64_t> list({1, 2, 3});
  list_reserve_api_probe(&list, 16);
  auto cap = list_capacity_api_probe(list);
  file << std::to_string(list.size()) << " ";
  file << std::to_string(cap >= list.size()) << " ";

  file << "\n";
  file.saveFile();
}

TEST_F(ListCompatTest, ListPushBackAndEmplaceBack) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "ListPushBackAndEmplaceBack ";

  c10::List<int64_t> list;
  int64_t lv = 7;
  list_push_back_lvalue_api_probe(&list, lv);
  list_push_back_rvalue_api_probe(&list, 11);
  list_emplace_back_api_probe(&list, 23);

  file << std::to_string(list.size()) << " ";
  file << std::to_string(static_cast<int64_t>(list[2])) << " ";
  file << std::to_string(static_cast<int64_t>(list[0])) << " ";
  file << std::to_string(static_cast<int64_t>(list[1])) << " ";
  file << std::to_string(static_cast<int64_t>(list[2])) << " ";

  file << "\n";
  file.saveFile();
}

TEST_F(ListCompatTest, ListPopBackAndClear) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "ListPopBackAndClear ";

  c10::List<int64_t> list({10, 20, 30});
  list_pop_back_api_probe(&list);
  file << std::to_string(list.size()) << " ";
  list_clear_api_probe(&list);
  file << std::to_string(list.size()) << " ";
  file << std::to_string(list.empty()) << " ";

  file << "\n";
  file.saveFile();
}

TEST_F(ListCompatTest, ListResizeVariants) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "ListResizeVariants ";

  c10::List<int64_t> list({3, 4});
  list_resize_count_api_probe(&list, 5);
  file << std::to_string(list.size()) << " ";
  file << std::to_string(static_cast<int64_t>(list[4])) << " ";

  list_resize_with_value_api_probe(&list, 7, 99);
  file << std::to_string(list.size()) << " ";
  file << std::to_string(static_cast<int64_t>(list[5])) << " ";
  file << std::to_string(static_cast<int64_t>(list[6])) << " ";

  file << "\n";
  file.saveFile();
}

}  // namespace test
}  // namespace at
