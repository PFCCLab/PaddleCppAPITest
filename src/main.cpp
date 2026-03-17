#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <mutex>

#include "gtest/gtest.h"
#if USE_PADDLE_API
#include "paddle/extension.h"
#endif

#include "../src/file_manager.h"

paddle_api_test::ThreadSafeParam g_custom_param;

std::string extract_filename(const std::string& path) {
  size_t last_slash = path.find_last_of('/');
  if (last_slash != std::string::npos) {
    return path.substr(last_slash + 1);
  }
  return path;
}

int main(int argc, char** argv) {  // NOLINT
  testing::InitGoogleTest(&argc, argv);

  auto exe_cmd = std::string(argv[0]);
  auto result_file_name = extract_filename(exe_cmd) + ".txt";
  g_custom_param.set(result_file_name);

  const std::string result_path =
      "/tmp/paddle_cpp_api_test/" + result_file_name;
  std::error_code ec;
  std::filesystem::remove(result_path, ec);

  int ret = RUN_ALL_TESTS();

  return ret;
}
