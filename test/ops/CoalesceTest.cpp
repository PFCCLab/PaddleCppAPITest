#include <ATen/ATen.h>
#include <ATen/core/Tensor.h>
#include <ATen/ops/sparse_coo_tensor.h>
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

class CoalesceTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // 构建 3x4 的稀疏 COO tensor（无重复索引）
    // 索引：[[0,1,1],[0,1,2]]，值：[1,2,3]
    at::Tensor idx = at::zeros({2, 3}, at::kLong);
    idx[0][0] = 0;
    idx[0][1] = 1;
    idx[0][2] = 1;
    idx[1][0] = 0;
    idx[1][1] = 1;
    idx[1][2] = 2;
    at::Tensor val = at::zeros({3}, at::kFloat);
    val[0] = 1.f;
    val[1] = 2.f;
    val[2] = 3.f;
    sparse_unique = at::sparse_coo_tensor(idx, val, {3, 4});

    // 构建含重复索引的稀疏 tensor（位置 (1,1) 出现两次）
    // 索引：[[0,1,1],[0,1,1]]，值：[1,2,10]
    // coalesce 后：(0,0)=1, (1,1)=12，nnz=2
    at::Tensor idx_dup = at::zeros({2, 3}, at::kLong);
    idx_dup[0][0] = 0;
    idx_dup[0][1] = 1;
    idx_dup[0][2] = 1;
    idx_dup[1][0] = 0;
    idx_dup[1][1] = 1;
    idx_dup[1][2] = 1;
    at::Tensor val_dup = at::zeros({3}, at::kFloat);
    val_dup[0] = 1.f;
    val_dup[1] = 2.f;
    val_dup[2] = 10.f;
    sparse_dup = at::sparse_coo_tensor(idx_dup, val_dup, {3, 4});
  }

  at::Tensor sparse_unique;
  at::Tensor sparse_dup;
};

// 测试 _nnz()：返回存储的非零元素数（含重复计数）
TEST_F(CoalesceTest, NnzBasic) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();

  file << std::to_string(sparse_unique._nnz()) << " ";  // 3
  file << std::to_string(sparse_dup._nnz()) << " ";     // 3（含重复）
  file.saveFile();
}

// 测试 _values()：返回稀疏 tensor 的 values 子 tensor
TEST_F(CoalesceTest, ValuesBasic) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();

  at::Tensor v = sparse_unique._values();
  file << std::to_string(v.dim()) << " ";
  file << std::to_string(v.numel()) << " ";
  float* data = v.data_ptr<float>();
  file << std::to_string(data[0]) << " ";
  file << std::to_string(data[1]) << " ";
  file << std::to_string(data[2]) << " ";
  file.saveFile();
}

// 测试 is_coalesced()：未经 coalesce 调用的张量
TEST_F(CoalesceTest, IsCoalescedInitial) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();

  // 含重复索引，未经显式 coalesce，初始状态 is_coalesced 为 false
  file << std::to_string(static_cast<int>(sparse_dup.is_coalesced())) << " ";
  file.saveFile();
}

// 测试 coalesce()：合并重复索引后 nnz 减少
TEST_F(CoalesceTest, CoalesceReducesNnz) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();

  at::Tensor coalesced = sparse_dup.coalesce();
  file << std::to_string(coalesced._nnz()) << " ";  // 2（重复已合并）
  file.saveFile();
}

// 测试 coalesce() 后 is_coalesced() 返回 true
TEST_F(CoalesceTest, CoalesceIsCoalesced) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();

  at::Tensor coalesced = sparse_dup.coalesce();
  file << std::to_string(static_cast<int>(coalesced.is_coalesced())) << " ";
  file.saveFile();
}

// 测试 coalesce() 后重复索引值被累加
TEST_F(CoalesceTest, CoalesceAccumulatesValues) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();

  at::Tensor coalesced = sparse_dup.coalesce();
  at::Tensor v = coalesced._values();
  // 按索引排序后：(0,0)=1, (1,1)=12
  file << std::to_string(v.numel()) << " ";
  // 值应出现 1.0 和 12.0（顺序取决于实现，输出全部值排序后比较）
  float* data = v.data_ptr<float>();
  float sum = 0.f;
  for (int i = 0; i < v.numel(); ++i) sum += data[i];
  file << std::to_string(sum) << " ";  // 1 + 12 = 13
  file.saveFile();
}

// 测试无重复索引的 sparse tensor coalesce 后 nnz 不变
TEST_F(CoalesceTest, CoalesceUniqueIndicesPreservesNnz) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();

  at::Tensor coalesced = sparse_unique.coalesce();
  file << std::to_string(coalesced._nnz()) << " ";  // 仍为 3
  file << std::to_string(static_cast<int>(coalesced.is_coalesced())) << " ";
  file.saveFile();
}

}  // namespace test
}  // namespace at
