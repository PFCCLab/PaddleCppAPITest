#include <c10/core/DispatchKey.h>
#include <gtest/gtest.h>

#include <functional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "src/file_manager.h"

extern paddle_api_test::ThreadSafeParam g_custom_param;

namespace at {
namespace test {

using paddle_api_test::FileManerger;
using paddle_api_test::ThreadSafeParam;

class DispatchKeyTest : public ::testing::Test {
 protected:
  void SetUp() override {}
};

TEST_F(DispatchKeyTest, AliasAndPerBackendPredicates) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();
  file << "AliasAndPerBackendPredicates ";

  file << std::to_string(c10::isAliasDispatchKey(c10::DispatchKey::Autograd))
       << " ";
  file << std::to_string(c10::isAliasDispatchKey(
              c10::DispatchKey::CompositeExplicitAutograd))
       << " ";
  file << std::to_string(c10::isAliasDispatchKey(c10::DispatchKey::CPU)) << " ";
  file << std::to_string(
              c10::isPerBackendFunctionalityKey(c10::DispatchKey::Dense))
       << " ";
  file << std::to_string(
              c10::isPerBackendFunctionalityKey(c10::DispatchKey::Sparse))
       << " ";
  file << std::to_string(c10::isPerBackendFunctionalityKey(
              c10::DispatchKey::Functionalize))
       << " ";
  file << "\n";
  file.saveFile();
}

TEST_F(DispatchKeyTest, MappingHelpers) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "MappingHelpers ";

  file << static_cast<int>(c10::toFunctionalityKey(c10::DispatchKey::Dense))
       << " ";
  file << static_cast<int>(c10::toFunctionalityKey(c10::DispatchKey::CPU))
       << " ";
  file << static_cast<int>(
              c10::toFunctionalityKey(c10::DispatchKey::QuantizedCPU))
       << " ";
  file << static_cast<int>(
              c10::toFunctionalityKey(c10::DispatchKey::SparseCsrCPU))
       << " ";
  file << static_cast<int>(
              c10::toFunctionalityKey(c10::DispatchKey::NestedTensorCPU))
       << " ";
  file << static_cast<int>(
              c10::toFunctionalityKey(c10::DispatchKey::AutogradCUDA))
       << " ";
  file << static_cast<int>(c10::toFunctionalityKey(c10::DispatchKey::Autograd))
       << " ";

  file << static_cast<int>(c10::toBackendComponent(c10::DispatchKey::CPU))
       << " ";
  file << static_cast<int>(
              c10::toBackendComponent(c10::DispatchKey::QuantizedCPU))
       << " ";
  file << static_cast<int>(c10::toBackendComponent(c10::DispatchKey::SparseCPU))
       << " ";
  file << static_cast<int>(
              c10::toBackendComponent(c10::DispatchKey::SparseCsrCPU))
       << " ";
  file << static_cast<int>(
              c10::toBackendComponent(c10::DispatchKey::NestedTensorCPU))
       << " ";
  file << static_cast<int>(
              c10::toBackendComponent(c10::DispatchKey::AutogradCUDA))
       << " ";
  file << static_cast<int>(c10::toBackendComponent(c10::DispatchKey::Autograd))
       << " ";

  file << static_cast<int>(c10::toRuntimePerBackendFunctionalityKey(
              c10::DispatchKey::Dense, c10::BackendComponent::CUDABit))
       << " ";
  file << static_cast<int>(c10::toRuntimePerBackendFunctionalityKey(
              c10::DispatchKey::Sparse, c10::BackendComponent::CPUBit))
       << " ";
  file << static_cast<int>(c10::toRuntimePerBackendFunctionalityKey(
              c10::DispatchKey::SparseCsr, c10::BackendComponent::CPUBit))
       << " ";
  file << static_cast<int>(c10::toRuntimePerBackendFunctionalityKey(
              c10::DispatchKey::Quantized, c10::BackendComponent::CPUBit))
       << " ";
  file << static_cast<int>(c10::toRuntimePerBackendFunctionalityKey(
              c10::DispatchKey::NestedTensor, c10::BackendComponent::CPUBit))
       << " ";
  file << static_cast<int>(c10::toRuntimePerBackendFunctionalityKey(
              c10::DispatchKey::AutogradFunctionality,
              c10::BackendComponent::CUDABit))
       << " ";
  file << static_cast<int>(c10::toRuntimePerBackendFunctionalityKey(
              c10::DispatchKey::Functionalize, c10::BackendComponent::CPUBit))
       << " ";

  file << "\n";
  file.saveFile();
}

TEST_F(DispatchKeyTest, AutogradAndParsing) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "AutogradAndParsing ";

  file << static_cast<int>(
              c10::getAutogradKeyFromBackend(c10::BackendComponent::CPUBit))
       << " ";
  file << static_cast<int>(
              c10::getAutogradKeyFromBackend(c10::BackendComponent::CUDABit))
       << " ";
  file << static_cast<int>(
              c10::getAutogradKeyFromBackend(c10::BackendComponent::XPUBit))
       << " ";
  file << static_cast<int>(
              c10::getAutogradKeyFromBackend(c10::BackendComponent::InvalidBit))
       << " ";

  file << static_cast<int>(c10::parseDispatchKey("Dense")) << " ";
  file << static_cast<int>(c10::parseDispatchKey("Sparse")) << " ";
  file << static_cast<int>(c10::parseDispatchKey("Autograd")) << " ";
  file << static_cast<int>(c10::parseDispatchKey("CompositeExplicitAutograd"))
       << " ";
  file << static_cast<int>(c10::parseDispatchKey("Undefined")) << " ";

  file << "\n";
  file.saveFile();
}

TEST_F(DispatchKeyTest, StringAndHash) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "StringAndHash ";

  std::ostringstream key_stream;
  key_stream << c10::DispatchKey::Dense << " " << c10::DispatchKey::Autograd;

  std::ostringstream backend_stream;
  backend_stream << c10::BackendComponent::CPUBit << " "
                 << c10::BackendComponent::CUDABit;

  file << c10::toString(c10::DispatchKey::Dense) << " ";
  file << c10::toString(c10::DispatchKey::Autograd) << " ";
  file << c10::toString(c10::BackendComponent::CUDABit) << " ";
  file << key_stream.str() << " ";
  file << backend_stream.str() << " ";
  file << std::hash<c10::DispatchKey>{}(c10::DispatchKey::Dense) << " ";
  file << std::hash<c10::DispatchKey>{}(c10::DispatchKey::Autograd) << " ";
  file << std::to_string(c10::kAutograd == c10::DispatchKey::Autograd) << " ";

  file << "\n";
  file.saveFile();
}

TEST_F(DispatchKeyTest, ExtensivePublicApiCoverage) {
  const std::vector<c10::DispatchKey> printable_keys = {
      c10::DispatchKey::Undefined,
      c10::DispatchKey::Dense,
      c10::DispatchKey::FPGA,
      c10::DispatchKey::Vulkan,
      c10::DispatchKey::Metal,
      c10::DispatchKey::Quantized,
      c10::DispatchKey::CustomRNGKeyId,
      c10::DispatchKey::MkldnnCPU,
      c10::DispatchKey::Sparse,
      c10::DispatchKey::SparseCsr,
      c10::DispatchKey::NestedTensor,
      c10::DispatchKey::BackendSelect,
      c10::DispatchKey::Python,
      c10::DispatchKey::Fake,
      c10::DispatchKey::FuncTorchDynamicLayerBackMode,
      c10::DispatchKey::Functionalize,
      c10::DispatchKey::Named,
      c10::DispatchKey::Conjugate,
      c10::DispatchKey::Negative,
      c10::DispatchKey::ZeroTensor,
      c10::DispatchKey::ADInplaceOrView,
      c10::DispatchKey::AutogradOther,
      c10::DispatchKey::AutogradFunctionality,
      c10::DispatchKey::AutogradNestedTensor,
      c10::DispatchKey::Tracer,
      c10::DispatchKey::AutocastCPU,
      c10::DispatchKey::AutocastMTIA,
      c10::DispatchKey::AutocastMAIA,
      c10::DispatchKey::AutocastXPU,
      c10::DispatchKey::AutocastIPU,
      c10::DispatchKey::AutocastHPU,
      c10::DispatchKey::AutocastXLA,
      c10::DispatchKey::AutocastMPS,
      c10::DispatchKey::AutocastCUDA,
      c10::DispatchKey::AutocastPrivateUse1,
      c10::DispatchKey::FuncTorchBatched,
      c10::DispatchKey::BatchedNestedTensor,
      c10::DispatchKey::FuncTorchVmapMode,
      c10::DispatchKey::Batched,
      c10::DispatchKey::VmapMode,
      c10::DispatchKey::FuncTorchGradWrapper,
      c10::DispatchKey::DeferredInit,
      c10::DispatchKey::PythonTLSSnapshot,
      c10::DispatchKey::FuncTorchDynamicLayerFrontMode,
      c10::DispatchKey::TESTING_ONLY_GenericWrapper,
      c10::DispatchKey::TESTING_ONLY_GenericMode,
      c10::DispatchKey::PreDispatch,
      c10::DispatchKey::PythonDispatcher,
      c10::DispatchKey::Autograd,
      c10::DispatchKey::CompositeImplicitAutograd,
      c10::DispatchKey::FuncTorchBatchedDecomposition,
      c10::DispatchKey::CompositeImplicitAutogradNestedTensor,
      c10::DispatchKey::CompositeExplicitAutograd,
      c10::DispatchKey::CompositeExplicitAutogradNonFunctional,
  };
  for (auto key : printable_keys) {
    EXPECT_FALSE(std::string(c10::toString(key)).empty());
    std::ostringstream stream;
    stream << key;
    EXPECT_FALSE(stream.str().empty());
  }

  const std::vector<c10::BackendComponent> backend_bits = {
      c10::BackendComponent::CPUBit,
      c10::BackendComponent::CUDABit,
      c10::BackendComponent::HIPBit,
      c10::BackendComponent::XLABit,
      c10::BackendComponent::MPSBit,
      c10::BackendComponent::IPUBit,
      c10::BackendComponent::XPUBit,
      c10::BackendComponent::HPUBit,
      c10::BackendComponent::VEBit,
      c10::BackendComponent::LazyBit,
      c10::BackendComponent::MTIABit,
      c10::BackendComponent::MAIABit,
      c10::BackendComponent::PrivateUse1Bit,
      c10::BackendComponent::PrivateUse2Bit,
      c10::BackendComponent::PrivateUse3Bit,
      c10::BackendComponent::MetaBit,
      c10::BackendComponent::InvalidBit,
  };
  for (auto bit : backend_bits) {
    EXPECT_FALSE(std::string(c10::toString(bit)).empty());
    std::ostringstream stream;
    stream << bit;
    EXPECT_FALSE(stream.str().empty());
  }

  const std::vector<std::pair<c10::BackendComponent, c10::DispatchKey>>
      autograd_pairs = {
          {c10::BackendComponent::CPUBit, c10::DispatchKey::AutogradCPU},
          {c10::BackendComponent::CUDABit, c10::DispatchKey::AutogradCUDA},
          {c10::BackendComponent::XPUBit, c10::DispatchKey::AutogradXPU},
          {c10::BackendComponent::IPUBit, c10::DispatchKey::AutogradIPU},
          {c10::BackendComponent::HPUBit, c10::DispatchKey::AutogradHPU},
          {c10::BackendComponent::LazyBit, c10::DispatchKey::AutogradLazy},
          {c10::BackendComponent::MetaBit, c10::DispatchKey::AutogradMeta},
          {c10::BackendComponent::MPSBit, c10::DispatchKey::AutogradMPS},
          {c10::BackendComponent::PrivateUse1Bit,
           c10::DispatchKey::AutogradPrivateUse1},
          {c10::BackendComponent::PrivateUse2Bit,
           c10::DispatchKey::AutogradPrivateUse2},
          {c10::BackendComponent::PrivateUse3Bit,
           c10::DispatchKey::AutogradPrivateUse3},
          {c10::BackendComponent::InvalidBit, c10::DispatchKey::AutogradOther},
      };
  for (const auto& [bit, expected] : autograd_pairs) {
    EXPECT_EQ(c10::getAutogradKeyFromBackend(bit), expected);
  }

  const std::vector<std::pair<c10::DispatchKey, c10::BackendComponent>>
      backend_cases = {
          {c10::DispatchKey::CPU, c10::BackendComponent::CPUBit},
          {c10::DispatchKey::QuantizedCUDA, c10::BackendComponent::CUDABit},
          {c10::DispatchKey::SparseXPU, c10::BackendComponent::XPUBit},
          {c10::DispatchKey::SparseCsrMeta, c10::BackendComponent::MetaBit},
          {c10::DispatchKey::NestedTensorPrivateUse1,
           c10::BackendComponent::PrivateUse1Bit},
          {c10::DispatchKey::AutogradMPS, c10::BackendComponent::MPSBit},
          {c10::DispatchKey::CompositeExplicitAutograd,
           c10::BackendComponent::InvalidBit},
      };
  for (const auto& [key, expected] : backend_cases) {
    EXPECT_EQ(c10::toBackendComponent(key), expected);
  }

  const std::vector<std::pair<c10::DispatchKey, c10::DispatchKey>>
      functionality_cases = {
          {c10::DispatchKey::Undefined, c10::DispatchKey::Undefined},
          {c10::DispatchKey::Dense, c10::DispatchKey::Dense},
          {c10::DispatchKey::CPU, c10::DispatchKey::Dense},
          {c10::DispatchKey::QuantizedCPU, c10::DispatchKey::Quantized},
          {c10::DispatchKey::SparseCPU, c10::DispatchKey::Sparse},
          {c10::DispatchKey::SparseCsrCPU, c10::DispatchKey::SparseCsr},
          {c10::DispatchKey::NestedTensorCPU, c10::DispatchKey::NestedTensor},
          {c10::DispatchKey::AutogradPrivateUse1,
           c10::DispatchKey::AutogradFunctionality},
          {c10::DispatchKey::CompositeExplicitAutogradNonFunctional,
           c10::DispatchKey::Undefined},
      };
  for (const auto& [input, expected] : functionality_cases) {
    EXPECT_EQ(c10::toFunctionalityKey(input), expected);
  }

  const std::vector<std::pair<const char*, c10::DispatchKey>> parse_cases = {
      {"Undefined", c10::DispatchKey::Undefined},
      {"Dense", c10::DispatchKey::Dense},
      {"FPGA", c10::DispatchKey::FPGA},
      {"Vulkan", c10::DispatchKey::Vulkan},
      {"Metal", c10::DispatchKey::Metal},
      {"Quantized", c10::DispatchKey::Quantized},
      {"Sparse", c10::DispatchKey::Sparse},
      {"SparseCsr", c10::DispatchKey::SparseCsr},
      {"NestedTensor", c10::DispatchKey::NestedTensor},
      {"BackendSelect", c10::DispatchKey::BackendSelect},
      {"Python", c10::DispatchKey::Python},
      {"Fake", c10::DispatchKey::Fake},
      {"Functionalize", c10::DispatchKey::Functionalize},
      {"Named", c10::DispatchKey::Named},
      {"Conjugate", c10::DispatchKey::Conjugate},
      {"Negative", c10::DispatchKey::Negative},
      {"ZeroTensor", c10::DispatchKey::ZeroTensor},
      {"ADInplaceOrView", c10::DispatchKey::ADInplaceOrView},
      {"AutogradOther", c10::DispatchKey::AutogradOther},
      {"AutogradFunctionality", c10::DispatchKey::AutogradFunctionality},
      {"AutogradNestedTensor", c10::DispatchKey::AutogradNestedTensor},
      {"Tracer", c10::DispatchKey::Tracer},
      {"AutocastCPU", c10::DispatchKey::AutocastCPU},
      {"AutocastCUDA", c10::DispatchKey::AutocastCUDA},
      {"Autograd", c10::DispatchKey::Autograd},
      {"CompositeImplicitAutograd",
       c10::DispatchKey::CompositeImplicitAutograd},
      {"CompositeExplicitAutograd",
       c10::DispatchKey::CompositeExplicitAutograd},
  };
  for (const auto& [name, expected] : parse_cases) {
    EXPECT_EQ(c10::parseDispatchKey(name), expected);
  }

  EXPECT_FALSE(
      std::string(c10::toString(static_cast<c10::DispatchKey>(999))).empty());
  EXPECT_FALSE(
      std::string(c10::toString(static_cast<c10::BackendComponent>(255)))
          .empty());
  try {
    (void)c10::parseDispatchKey("NotARealDispatchKey");
  } catch (...) {
  }
}

}  // namespace test
}  // namespace at
