#include <c10/core/DeviceType.h>
#include <gtest/gtest.h>

#include <functional>
#include <sstream>
#include <string>

#include "src/file_manager.h"

extern paddle_api_test::ThreadSafeParam g_custom_param;

namespace at {
namespace test {

using paddle_api_test::FileManerger;
using paddle_api_test::ThreadSafeParam;

class DeviceTypeTest : public ::testing::Test {
 protected:
  void SetUp() override {}
};

TEST_F(DeviceTypeTest, CommonEnumsAndAliases) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.createFile();
  file << "CommonEnumsAndAliases ";

  file << static_cast<int>(c10::DeviceType::CPU) << " ";
  file << static_cast<int>(c10::DeviceType::CUDA) << " ";
  file << static_cast<int>(c10::DeviceType::XPU) << " ";
  file << static_cast<int>(c10::DeviceType::IPU) << " ";
  file << static_cast<int>(c10::DeviceType::PrivateUse1) << " ";
  file << std::to_string(c10::kCPU == c10::DeviceType::CPU) << " ";
  file << std::to_string(c10::kCUDA == c10::DeviceType::CUDA) << " ";
  file << std::to_string(c10::kXPU == c10::DeviceType::XPU) << " ";
  file << std::to_string(c10::kIPU == c10::DeviceType::IPU) << " ";
  file << std::to_string(c10::kPrivateUse1 == c10::DeviceType::PrivateUse1)
       << " ";

  file << "\n";
  file.saveFile();
}

TEST_F(DeviceTypeTest, ValidityAndHash) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "ValidityAndHash ";

  file << std::to_string(c10::isValidDeviceType(c10::DeviceType::CPU)) << " ";
  file << std::to_string(c10::isValidDeviceType(c10::DeviceType::CUDA)) << " ";
  file << std::to_string(c10::isValidDeviceType(c10::DeviceType::XPU)) << " ";
  file << std::to_string(c10::isValidDeviceType(c10::DeviceType::IPU)) << " ";
  file << std::to_string(c10::isValidDeviceType(c10::DeviceType::PrivateUse1))
       << " ";
  file << std::to_string(
              c10::isValidDeviceType(static_cast<c10::DeviceType>(-1)))
       << " ";
  file << std::hash<c10::DeviceType>{}(c10::DeviceType::CPU) << " ";
  file << std::hash<c10::DeviceType>{}(c10::DeviceType::CUDA) << " ";
  file << std::hash<c10::DeviceType>{}(c10::DeviceType::PrivateUse1) << " ";

  file << "\n";
  file.saveFile();
}

TEST_F(DeviceTypeTest, StreamOutput) {
  auto file_name = g_custom_param.get();
  FileManerger file(file_name);
  file.openAppend();
  file << "StreamOutput ";

  std::ostringstream stream;
  stream << c10::DeviceType::CPU << " " << c10::DeviceType::CUDA << " "
         << c10::DeviceType::XPU << " " << c10::DeviceType::IPU << " "
         << c10::DeviceType::PrivateUse1;

  file << stream.str() << " ";
  file << "\n";
  file.saveFile();
}

}  // namespace test
}  // namespace at
