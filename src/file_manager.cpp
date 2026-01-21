#include "src/file_manager.h"

#include <filesystem>
#include <iostream>

namespace paddle_api_test {

void FileManerger::createFile() {
  std::unique_lock<std::shared_mutex> lock(mutex_);

  // 确保目录存在（如果目录已存在，create_directories会返回false，这是正常情况）
  std::error_code ec;
  if (!std::filesystem::create_directories(basic_path_, ec) && ec) {
    throw std::runtime_error("Failed to create directory: " + basic_path_ +
                             ", error: " + ec.message());
  }

  // 构建完整文件路径
  std::string full_path = basic_path_ + file_name_;

  // 如果文件已存在则删除
  if (std::filesystem::exists(full_path)) {
    std::filesystem::remove(full_path);
  }

  // 创建并打开文件流
  file_stream_.open(full_path, std::ios::out | std::ios::trunc);
  if (!file_stream_.is_open()) {
    throw std::runtime_error("Failed to create file: " + full_path);
  }
}

void FileManerger::writeString(const std::string& str) {
  std::shared_lock<std::shared_mutex> lock(mutex_);
  if (file_stream_.is_open()) {
    file_stream_ << str;
  } else {
    throw std::runtime_error(
        "File stream is not open. Call createFile() first.");
  }
}

FileManerger& FileManerger::operator<<(const std::string& str) {
  writeString(str);
  return *this;
}

void FileManerger::saveFile() {
  std::unique_lock<std::shared_mutex> lock(mutex_);
  if (file_stream_.is_open()) {
    // 先刷新缓冲区，确保数据写入磁盘
    file_stream_.flush();
    // 然后关闭文件流
    file_stream_.close();
  }
}

void FileManerger::setFileName(const std::string& value) {
  std::unique_lock<std::shared_mutex> lock(mutex_);
  file_name_ = value;
}
}  // namespace paddle_api_test
