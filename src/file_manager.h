#pragma once
#include <fstream>
#include <mutex>
#include <shared_mutex>
#include <string>

namespace paddle_api_test {
class FileManerger {
 public:
  FileManerger() = default;
  explicit FileManerger(const std::string& first) : file_name_(first) {}

  void setFileName(const std::string& value);
  void createFile();
  void openAppend();
  void writeString(const std::string& str);
  FileManerger& operator<<(const std::string& str);
  void saveFile();

 private:
  mutable std::shared_mutex mutex_;
  std::string basic_path_ = "/tmp/paddle_cpp_api_test/";
  std::string file_name_ = "";
  std::ofstream file_stream_;
};

class ThreadSafeParam {
 private:
  std::string param_;
  mutable std::mutex mutex_;

 public:
  void set(const std::string& value) {
    std::lock_guard<std::mutex> lock(mutex_);
    param_ = value;
  }

  std::string get() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return param_;
  }
};
}  // namespace paddle_api_test
