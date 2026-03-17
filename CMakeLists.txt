cmake_minimum_required(VERSION 3.18)
set(PROJECT_NAME)
project(${PROJECT_NAME} C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=1)

option(ENABLE_COVERAGE "Enable converage detection" OFF)
if(ENABLE_COVERAGE)
  message(
    STATUS "Coverage build enabled via ENABLE_COVERAGE environment variable.")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fprofile-arcs -ftest-coverage")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -lgcov")
endif()

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake ${CMAKE_MODULE_PATH})

find_package(Threads REQUIRED)

set(EXE_TARGET_NAME "all_api_tests")
if(CMAKE_COMPILER_IS_GNUCXX)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -ansi -Wno-deprecated")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -ansi -Wno-deprecated")
endif()

# ===========================================================================
# 使用本地已安装的 GoogleTest（直接从系统目录寻找，不再去 GitHub 克隆）
# ===========================================================================
find_package(GTest REQUIRED)
message(STATUS "Found System GTest: ${GTEST_INCLUDE_DIRS}")
include_directories(${GTEST_INCLUDE_DIRS})
# ===========================================================================

find_program(Python3_EXECUTABLE python3 REQUIRED)
if(EXISTS "/home/pc2160/anaconda3/envs/paddle/bin/python3")
  set(Python3_EXECUTABLE
      "/home/pc2160/anaconda3/envs/paddle/bin/python3"
      CACHE FILEPATH "Preferred Python executable" FORCE)
endif()
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

if(NOT Python3_INCLUDE_DIRS)
  message(
    FATAL_ERROR
      "Python3 header not found, please install python3-dev or python3-devel")
endif()

message(STATUS "Python Path: ${Python3_EXECUTABLE}")
set(COMMAND_TO_RUN
    "${Python3_EXECUTABLE} -c \"import paddle; print(paddle.__path__[0])\"")

include_directories(${Python3_INCLUDE_DIRS})
link_directories("${Python3_LIBRARY_DIRS}")

set(COMMON_INCLUDES ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/include
                    ${PROJECT_SOURCE_DIR}/src)

find_path(CUDA_INCLUDE_DIR cuda.h HINTS /usr/local/cuda/include /usr/include)
if(CUDA_INCLUDE_DIR)
  message(STATUS "Found CUDA headers: ${CUDA_INCLUDE_DIR}")
  list(APPEND COMMON_INCLUDES ${CUDA_INCLUDE_DIR})
else()
  message(STATUS "CUDA headers not found: CUDADataType tests may be skipped")
endif()

enable_testing()
include_directories(${COMMON_INCLUDES})
include(cmake/build.cmake)

file(GLOB TEST_SRC_FILES ${PROJECT_SOURCE_DIR}/test/*.cpp
     ${PROJECT_SOURCE_DIR}/test/ops/*.cpp)
# 排除已unmatch开头的文件
file(GLOB UNMATCH_FILES ${PROJECT_SOURCE_DIR}/test/unmatch*.cpp
     ${PROJECT_SOURCE_DIR}/test/ops/unmatch*.cpp)

list(REMOVE_ITEM TEST_SRC_FILES ${UNMATCH_FILES})

file(GLOB_RECURSE TEST_BASE_FILES ${PROJECT_SOURCE_DIR}/src/*.cpp)
set(PADDLE_TARGET_FOLDER ${CMAKE_BINARY_DIR}/paddle)

# ---------------------------------------------------------------------------
# Build Torch test case
# ---------------------------------------------------------------------------
set(TORCH_DIR
    "/home/pc2160/yangxiaochun/PaddleCppAPI/libtorch"
    CACHE PATH "Path to libtorch installation")

set(TORCH_LIB_DIR "${TORCH_DIR}/lib")
set(TORCH_LIBRARIES
    "${TORCH_LIB_DIR}/libtorch.so" "${TORCH_LIB_DIR}/libtorch_cpu.so"
    "${TORCH_LIB_DIR}/libc10.so")

foreach(_required_torch_lib ${TORCH_LIBRARIES})
  if(NOT EXISTS "${_required_torch_lib}")
    message(
      FATAL_ERROR "Required Torch library not found: ${_required_torch_lib}")
  endif()
endforeach()

set(TORCH_HAS_CUDA_LIBS OFF)
if(EXISTS "${TORCH_LIB_DIR}/libtorch_cuda.so"
   AND EXISTS "${TORCH_LIB_DIR}/libc10_cuda.so")
  list(APPEND TORCH_LIBRARIES "${TORCH_LIB_DIR}/libtorch_cuda.so"
       "${TORCH_LIB_DIR}/libc10_cuda.so")
  set(TORCH_HAS_CUDA_LIBS ON)
endif()

set(TORCH_INCLUDE_DIR "${TORCH_DIR}/include"
                      "${TORCH_DIR}/include/torch/csrc/api/include/")

set(PADDLE_SOURCE_DIR "/home/pc2160/yangxiaochun/PaddleCppAPI/Paddle")
set(PADDLE_BUILD_DIR "${PADDLE_SOURCE_DIR}/build")
if(NOT EXISTS "${PADDLE_BUILD_DIR}")
  message(FATAL_ERROR "Paddle build 目录不存在，请确认路径是否正确: ${PADDLE_BUILD_DIR}")
endif()

set(PADDLE_HAS_CUDA_LIBS OFF)
if(EXISTS "${PADDLE_BUILD_DIR}/paddle/phi/libphi_gpu.so")
  set(PADDLE_HAS_CUDA_LIBS ON)
endif()

set(TORCH_CUDA_TEST_FILE "${PROJECT_SOURCE_DIR}/test/TorchCudaTest.cpp")
if(TORCH_HAS_CUDA_LIBS AND PADDLE_HAS_CUDA_LIBS)
  message(
    STATUS "Enable TorchCudaTest: Torch/Paddle CUDA runtime libs detected")
else()
  list(REMOVE_ITEM TEST_SRC_FILES "${TORCH_CUDA_TEST_FILE}")
  message(
    WARNING
      "Disable TorchCudaTest.cpp because CUDA runtime is incomplete for compatibility run. "
      "Torch CUDA libs found: ${TORCH_HAS_CUDA_LIBS}, Paddle CUDA libs found: ${PADDLE_HAS_CUDA_LIBS}"
  )
endif()

set(TORCH_TARGET_FOLDER ${CMAKE_BINARY_DIR}/torch)
set(BIN_PREFIX "torch_")
create_paddle_tests(
  "${BIN_PREFIX}" "${TEST_SRC_FILES}" "${TORCH_TARGET_FOLDER}"
  "${TORCH_LIBRARIES}" "${TORCH_INCLUDE_DIR}" 0)

# ---------------------------------------------------------------------------
# Build Paddle test case (使用本地编译的源码及 .so)
# ---------------------------------------------------------------------------

message(STATUS "Using Paddle source headers from: ${PADDLE_SOURCE_DIR}")
message(STATUS "Using Paddle compiled libs from: ${PADDLE_BUILD_DIR}")

# 1. 设置包含目录 (Include Directories) - 全面指向源码和 build 生成的第三方头文件
set(PADDLE_INCLUDE_DIR "${PADDLE_SOURCE_DIR}/")
set(PADDLE_INCLUDE_DIR
    ${PADDLE_INCLUDE_DIR}
    "${PADDLE_SOURCE_DIR}/paddle/phi/api/include"
    "${PADDLE_SOURCE_DIR}/paddle/phi/api/include/compat"
    "${PADDLE_SOURCE_DIR}/paddle/phi/api/include/compat/torch/csrc/api/include"
    # 添加 build 目录下生成的头文件及第三方依赖(如 pybind, glog 等)
    "${PADDLE_BUILD_DIR}/")

# ======= 自动获取 Python 环境中的 pybind11 头文件路径 =======
execute_process(
  COMMAND ${Python3_EXECUTABLE} -c
          "import pybind11; print(pybind11.get_include())"
  OUTPUT_VARIABLE PYBIND11_INC_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE pybind_res)

if(NOT pybind_res EQUAL 0 OR NOT EXISTS "${PYBIND11_INC_DIR}")
  if(EXISTS
     "/home/pc2160/anaconda3/envs/paddle/lib/python3.12/site-packages/pybind11/include"
  )
    set(PYBIND11_INC_DIR
        "/home/pc2160/anaconda3/envs/paddle/lib/python3.12/site-packages/pybind11/include"
    )
  endif()

  find_path(
    PYBIND11_INC_DIR
    NAMES pybind11/pybind11.h
    HINTS
      $ENV{CONDA_PREFIX}/include
      /home/pc2160/anaconda3/envs/paddle/lib/python3.12/site-packages/pybind11/include
      /usr/local/include
      /usr/include)
endif()

if(PYBIND11_INC_DIR AND EXISTS "${PYBIND11_INC_DIR}")
  message(STATUS "Found pybind11 headers at: ${PYBIND11_INC_DIR}")
  list(APPEND PADDLE_INCLUDE_DIR "${PYBIND11_INC_DIR}")
else()
  message(
    WARNING
      "pybind11 headers not found; continue without explicit pybind11 include path."
  )
endif()
# ==============================================================

# =========================================================================
# 1. 智能全局扫描并指定动态库 (彻底解决路径变动问题)
# =========================================================================
set(PADDLE_LIBRARIES "")

# ⚠️ 严格按照从高层到底层的依赖顺序排列库名称（在 GCC 中，顺序错了也会报错）
set(TARGET_LIB_NAMES "libpaddle.so" "libphi_gpu.so" # 如果你之前编译的是 CPU 版，它会安全跳过
                     "libphi.so" "libphi_core.so" "libcommon.so")

foreach(lib_name ${TARGET_LIB_NAMES})
  # 在 build/paddle 目录下进行全盘递归搜索
  file(GLOB_RECURSE found_libs "${PADDLE_BUILD_DIR}/paddle/${lib_name}")

  if(found_libs)
    # 如果找到（甚至找到多个），提取第一个准确的绝对路径
    list(GET found_libs 0 selected_lib)
    list(APPEND PADDLE_LIBRARIES "${selected_lib}")
    message(STATUS "✅ Found Paddle lib: ${selected_lib}")
  else()
    message(STATUS "⚠️ Skipped Paddle lib (Not found): ${lib_name}")
  endif()
endforeach()

# 兜底检查
if("${PADDLE_LIBRARIES}" STREQUAL "")
  message(FATAL_ERROR "❌ 没有在 ${PADDLE_BUILD_DIR} 中找到任何 Paddle 动态库，请检查是否编译成功！")
endif()
# =========================================================================
set(BIN_PREFIX "paddle_")

# 1. 生成测试目标
create_paddle_tests(
  "${BIN_PREFIX}" "${TEST_SRC_FILES}" "${PADDLE_TARGET_FOLDER}"
  "${PADDLE_LIBRARIES}" "${PADDLE_INCLUDE_DIR}" 1)
