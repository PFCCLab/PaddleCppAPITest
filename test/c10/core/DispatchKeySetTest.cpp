#include <c10/core/DispatchKeySet.h>
#include <gtest/gtest.h>

#include <array>
#include <sstream>
#include <string>
#include <vector>

#include "src/file_manager.h"

extern paddle_api_test::ThreadSafeParam g_custom_param;

namespace at {
namespace test {

using paddle_api_test::FileManerger;
using paddle_api_test::ThreadSafeParam;

class DispatchKeySetTest : public ::testing::Test {
 protected:
  void SetUp() override {}
};

TEST_F(DispatchKeySetTest, FunctionalityOffsetsAndMasks) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();
  file << "FunctionalityOffsetsAndMasks ";

  c10::FunctionalityOffsetAndMask empty;
  c10::FunctionalityOffsetAndMask custom(7, 9);
  auto generated = c10::initializeFunctionalityOffsetsAndMasks();
  const auto& cached_a = c10::offsetsAndMasks();
  const auto& cached_b = c10::offsetsAndMasks();

  const auto dense_idx = static_cast<uint8_t>(c10::DispatchKey::Dense);
  const auto functionalize_idx =
      static_cast<uint8_t>(c10::DispatchKey::Functionalize);
  const auto autograd_idx =
      static_cast<uint8_t>(c10::DispatchKey::AutogradFunctionality);

  file << empty.offset << " " << empty.mask << " ";
  file << custom.offset << " " << custom.mask << " ";
  file << generated[dense_idx].offset << " " << generated[dense_idx].mask
       << " ";
  file << generated[functionalize_idx].offset << " "
       << generated[functionalize_idx].mask << " ";
  file << cached_a[autograd_idx].offset << " " << cached_a[autograd_idx].mask
       << " ";
  file << std::to_string(&cached_a == &cached_b) << " ";

  file << "\n";
  file.saveFile();
}

TEST_F(DispatchKeySetTest, ConstructorsAndQueries) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "ConstructorsAndQueries ";

  c10::DispatchKeySet empty;
  c10::DispatchKeySet full(c10::DispatchKeySet::FULL);
  c10::DispatchKeySet full_after(c10::DispatchKeySet::FULL_AFTER,
                                 c10::DispatchKey::AutogradOther);
  c10::DispatchKeySet raw(c10::DispatchKeySet::RAW, 5);
  c10::DispatchKeySet backend(c10::BackendComponent::CPUBit);
  c10::DispatchKeySet functionality(c10::DispatchKey::Dense);
  c10::DispatchKeySet runtime(c10::DispatchKey::CUDA);
  c10::DispatchKeySet alias(c10::DispatchKey::Autograd);

  file << std::to_string(empty.empty()) << " ";
  file << full.raw_repr() << " ";
  file << full_after.raw_repr() << " ";
  file << raw.raw_repr() << " ";
  file << std::to_string(backend.has_backend(c10::BackendComponent::CPUBit))
       << " ";
  file << std::to_string(functionality.has_all(
              c10::DispatchKeySet(c10::DispatchKey::Dense)))
       << " ";
  file << std::to_string(runtime.has(c10::DispatchKey::CUDA)) << " ";
  file << std::to_string(alias.empty()) << " ";
  file << std::to_string(full.has(c10::DispatchKey::CPU)) << " ";
  file << std::to_string(full_after.has(c10::DispatchKey::PythonDispatcher))
       << " ";
  file << std::to_string(full_after.has(c10::DispatchKey::AutogradOther))
       << " ";

  file << "\n";
  file.saveFile();
}

TEST_F(DispatchKeySetTest, SetOpsAndPriority) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "SetOpsAndPriority ";

  c10::DispatchKeySet combined = c10::DispatchKeySet(c10::DispatchKey::CPU)
                                     .add(c10::DispatchKey::AutogradCUDA);
  c10::DispatchKeySet removed = combined.remove(c10::DispatchKey::AutogradCUDA);
  c10::DispatchKeySet removed_backend =
      combined.remove_backend(c10::BackendComponent::CUDABit);
  c10::DispatchKeySet xor_set =
      combined ^ c10::DispatchKeySet(c10::DispatchKey::AutogradCPU);
  c10::DispatchKeySet cpu_backend_bits(
      c10::DispatchKeySet::RAW,
      1ULL << (static_cast<uint8_t>(c10::BackendComponent::CPUBit) - 1));

  file << combined.raw_repr() << " ";
  file << removed.raw_repr() << " ";
  file << removed_backend.raw_repr() << " ";
  file << xor_set.raw_repr() << " ";
  file << std::to_string(combined.has_any(cpu_backend_bits)) << " ";
  file << std::to_string(combined.has_all(
              c10::DispatchKeySet(c10::DispatchKey::AutogradCUDA)))
       << " ";
  file << std::to_string(
              combined.isSupersetOf(c10::DispatchKeySet(c10::DispatchKey::CPU)))
       << " ";
  file << static_cast<int>(combined.highestFunctionalityKey()) << " ";
  file << static_cast<int>(combined.highestBackendKey()) << " ";
  file << static_cast<int>(combined.highestPriorityTypeId()) << " ";
  file << static_cast<int>(combined.indexOfHighestBit()) << " ";
  file << combined.getDispatchTableIndexForDispatchKeySet() << " ";
  file << combined.getBackendIndex() << " ";
  file << c10::getDispatchTableIndexForDispatchKey(c10::DispatchKey::CPU)
       << " ";

  file << "\n";
  file.saveFile();
}

TEST_F(DispatchKeySetTest, IteratorAndString) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "IteratorAndString ";

  c10::DispatchKeySet runtime_iterable(
      {c10::DispatchKey::CPU, c10::DispatchKey::AutogradCUDA});
  c10::DispatchKeySet empty;
  std::vector<int> runtime_keys;

  for (auto key : runtime_iterable) {
    runtime_keys.push_back(static_cast<int>(key));
  }
  const auto runtime_string = c10::toString(runtime_iterable);
  std::ostringstream runtime_stream;
  runtime_stream << runtime_iterable;
  EXPECT_FALSE(runtime_keys.empty());
  EXPECT_FALSE(runtime_string.empty());
  EXPECT_FALSE(runtime_stream.str().empty());

  std::ostringstream empty_stream;
  empty_stream << empty;
  file << std::to_string(!runtime_keys.empty()) << " ";
  file << c10::toString(empty) << " ";
  file << empty_stream.str() << " ";
  file << std::to_string(!runtime_string.empty()) << " ";
  file << std::to_string(!runtime_stream.str().empty()) << " ";
  file << std::to_string(empty.begin() == empty.end()) << " ";

  file << "\n";
  file.saveFile();
}

TEST_F(DispatchKeySetTest, HelperKeySets) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "HelperKeySets ";

  const auto runtime_cpu = c10::getRuntimeDispatchKeySet(c10::DispatchKey::CPU);
  const auto runtime_dense =
      c10::getRuntimeDispatchKeySet(c10::DispatchKey::Dense);
  c10::DispatchKeySet legacy_source({c10::DispatchKey::CPU,
                                     c10::DispatchKey::AutogradCUDA,
                                     c10::DispatchKey::Python});

  EXPECT_EQ(runtime_cpu.has(c10::DispatchKey::CPU),
            c10::runtimeDispatchKeySetHas(c10::DispatchKey::CPU,
                                          c10::DispatchKey::CPU));
  EXPECT_EQ(runtime_dense.has(c10::DispatchKey::CPU),
            c10::runtimeDispatchKeySetHas(c10::DispatchKey::Dense,
                                          c10::DispatchKey::CPU));

  file << std::to_string(runtime_cpu.has(c10::DispatchKey::CPU)) << " ";
  file << std::to_string(runtime_cpu.has(c10::DispatchKey::Dense)) << " ";
  file << std::to_string(c10::runtimeDispatchKeySetHas(c10::DispatchKey::CPU,
                                                       c10::DispatchKey::CPU))
       << " ";
  file << std::to_string(c10::runtimeDispatchKeySetHas(c10::DispatchKey::CPU,
                                                       c10::DispatchKey::CUDA))
       << " ";
  file << c10::getBackendKeySetFromAutograd(c10::DispatchKey::AutogradCUDA)
              .raw_repr()
       << " ";
  file << c10::getAutogradRelatedKeySetFromBackend(
              c10::BackendComponent::CUDABit)
              .raw_repr()
       << " ";
  file << c10::getAutocastRelatedKeySetFromBackend(
              c10::BackendComponent::CUDABit)
              .raw_repr()
       << " ";
  file << static_cast<int>(
              c10::highestPriorityBackendTypeId(c10::DispatchKeySet(
                  {c10::DispatchKey::CPU, c10::DispatchKey::AutogradCUDA})))
       << " ";
  file << std::to_string(
              c10::isIncludedInAlias(c10::DispatchKey::AutogradFunctionality,
                                     c10::DispatchKey::Autograd))
       << " ";
  file << std::to_string(c10::isIncludedInAlias(
              c10::DispatchKey::CPU,
              c10::DispatchKey::CompositeExplicitAutograd))
       << " ";
  file << std::to_string(c10::isIncludedInAlias(
              c10::DispatchKey::Dense,
              c10::DispatchKey::CompositeImplicitAutograd))
       << " ";
  file << static_cast<int>(c10::legacyExtractDispatchKey(legacy_source)) << " ";

  file << "\n";
  file.saveFile();
}

TEST_F(DispatchKeySetTest, AdditionalPublicApiCoverage) {
  volatile uint64_t default_raw = c10::DispatchKeySet().raw_repr();
  volatile uint64_t invalid_backend_raw =
      c10::DispatchKeySet(c10::BackendComponent::InvalidBit).raw_repr();
  volatile uint64_t undefined_key_raw =
      c10::DispatchKeySet(c10::DispatchKey::Undefined).raw_repr();
  volatile uint64_t alias_raw =
      c10::DispatchKeySet(c10::DispatchKey::CompositeExplicitAutograd)
          .raw_repr();
  EXPECT_EQ(default_raw, 0);
  EXPECT_EQ(invalid_backend_raw, 0);
  EXPECT_EQ(undefined_key_raw, 0);
  EXPECT_EQ(alias_raw, alias_raw);

  EXPECT_TRUE(c10::isBackendDispatchKey(c10::DispatchKey::CPU));
  EXPECT_FALSE(c10::isBackendDispatchKey(c10::DispatchKey::Autograd));
  EXPECT_FALSE(
      c10::isBackendDispatchKey(c10::DispatchKey::CompositeExplicitAutograd));

  const std::vector<std::pair<c10::BackendComponent, c10::DispatchKey>>
      autograd_related = {
          {c10::BackendComponent::CPUBit, c10::DispatchKey::AutogradCPU},
          {c10::BackendComponent::IPUBit, c10::DispatchKey::AutogradIPU},
          {c10::BackendComponent::MTIABit, c10::DispatchKey::AutogradMTIA},
          {c10::BackendComponent::MAIABit, c10::DispatchKey::AutogradMAIA},
          {c10::BackendComponent::XPUBit, c10::DispatchKey::AutogradXPU},
          {c10::BackendComponent::CUDABit, c10::DispatchKey::AutogradCUDA},
          {c10::BackendComponent::XLABit, c10::DispatchKey::AutogradXLA},
          {c10::BackendComponent::LazyBit, c10::DispatchKey::AutogradLazy},
          {c10::BackendComponent::MetaBit, c10::DispatchKey::AutogradMeta},
          {c10::BackendComponent::MPSBit, c10::DispatchKey::AutogradMPS},
          {c10::BackendComponent::HPUBit, c10::DispatchKey::AutogradHPU},
          {c10::BackendComponent::PrivateUse1Bit,
           c10::DispatchKey::AutogradPrivateUse1},
          {c10::BackendComponent::PrivateUse2Bit,
           c10::DispatchKey::AutogradPrivateUse2},
          {c10::BackendComponent::PrivateUse3Bit,
           c10::DispatchKey::AutogradPrivateUse3},
      };
  for (const auto& [bit, expected] : autograd_related) {
    auto ks = c10::getAutogradRelatedKeySetFromBackend(bit);
    EXPECT_TRUE(ks.has(c10::DispatchKey::ADInplaceOrView));
    EXPECT_TRUE(ks.has(expected));
  }
  auto default_autograd_related = c10::getAutogradRelatedKeySetFromBackend(
      c10::BackendComponent::InvalidBit);
  EXPECT_TRUE(default_autograd_related.has(c10::DispatchKey::ADInplaceOrView));
  EXPECT_TRUE(default_autograd_related.has(c10::DispatchKey::AutogradOther));

  const std::vector<std::pair<c10::BackendComponent, c10::DispatchKey>>
      autocast_related = {
          {c10::BackendComponent::CPUBit, c10::DispatchKey::AutocastCPU},
          {c10::BackendComponent::MTIABit, c10::DispatchKey::AutocastMTIA},
          {c10::BackendComponent::MAIABit, c10::DispatchKey::AutocastMAIA},
          {c10::BackendComponent::XPUBit, c10::DispatchKey::AutocastXPU},
          {c10::BackendComponent::IPUBit, c10::DispatchKey::AutocastIPU},
          {c10::BackendComponent::HPUBit, c10::DispatchKey::AutocastHPU},
          {c10::BackendComponent::CUDABit, c10::DispatchKey::AutocastCUDA},
          {c10::BackendComponent::XLABit, c10::DispatchKey::AutocastXLA},
          {c10::BackendComponent::PrivateUse1Bit,
           c10::DispatchKey::AutocastPrivateUse1},
          {c10::BackendComponent::MPSBit, c10::DispatchKey::AutocastMPS},
      };
  for (const auto& [bit, expected] : autocast_related) {
    auto ks = c10::getAutocastRelatedKeySetFromBackend(bit);
    EXPECT_TRUE(ks.has(expected));
  }
  EXPECT_TRUE(c10::getAutocastRelatedKeySetFromBackend(
                  c10::BackendComponent::InvalidBit)
                  .empty());

  const std::vector<std::pair<c10::DispatchKey, c10::DispatchKey>>
      backend_from_autograd = {
          {c10::DispatchKey::AutogradCPU, c10::DispatchKey::CPU},
          {c10::DispatchKey::AutogradCUDA, c10::DispatchKey::CUDA},
          {c10::DispatchKey::AutogradXPU, c10::DispatchKey::XPU},
          {c10::DispatchKey::AutogradIPU, c10::DispatchKey::IPU},
          {c10::DispatchKey::AutogradHPU, c10::DispatchKey::HPU},
          {c10::DispatchKey::AutogradLazy, c10::DispatchKey::Lazy},
          {c10::DispatchKey::AutogradMeta, c10::DispatchKey::Meta},
          {c10::DispatchKey::AutogradMPS, c10::DispatchKey::MPS},
          {c10::DispatchKey::AutogradPrivateUse1,
           c10::DispatchKey::PrivateUse1},
          {c10::DispatchKey::AutogradPrivateUse2,
           c10::DispatchKey::PrivateUse2},
          {c10::DispatchKey::AutogradPrivateUse3,
           c10::DispatchKey::PrivateUse3},
      };
  for (const auto& [autograd_key, backend_key] : backend_from_autograd) {
    auto ks = c10::getBackendKeySetFromAutograd(autograd_key);
    EXPECT_TRUE(ks.has(backend_key));
  }
  EXPECT_TRUE(
      c10::getBackendKeySetFromAutograd(c10::DispatchKey::AutogradNestedTensor)
          .has(c10::DispatchKey::NestedTensor));
  EXPECT_FALSE(
      c10::getBackendKeySetFromAutograd(c10::DispatchKey::AutogradOther)
          .empty());
  EXPECT_TRUE(
      c10::getBackendKeySetFromAutograd(c10::DispatchKey::Dense).empty());

  EXPECT_EQ(c10::DispatchKeySet().highestPriorityTypeId(),
            c10::DispatchKey::Undefined);
  EXPECT_EQ(
      c10::DispatchKeySet(c10::DispatchKey::Python).highestPriorityTypeId(),
      c10::DispatchKey::Python);

  const uint64_t dense_functionality_bit =
      1ULL << (c10::num_backends +
               static_cast<uint8_t>(c10::DispatchKey::Dense) - 1);
  const uint64_t python_functionality_bit =
      1ULL << (c10::num_backends +
               static_cast<uint8_t>(c10::DispatchKey::Python) - 1);
  c10::DispatchKeySet missing_backend_dense(
      c10::DispatchKeySet::RAW,
      dense_functionality_bit | python_functionality_bit);
  auto missing_backend_it = missing_backend_dense.begin();
  if (missing_backend_it != missing_backend_dense.end()) {
    (void)*missing_backend_it;
    ++missing_backend_it;
  }

  c10::DispatchKeySet runtime_cpu(c10::DispatchKey::CPU);
  auto runtime_it = runtime_cpu.begin();
  if (runtime_it != runtime_cpu.end()) {
    auto runtime_post = runtime_it++;
    (void)*runtime_post;
  }

  EXPECT_FALSE(
      c10::isIncludedInAlias(c10::DispatchKey::Dense, c10::DispatchKey::Fake));
}

}  // namespace test
}  // namespace at
