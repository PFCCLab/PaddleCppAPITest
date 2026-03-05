cmake_minimum_required(VERSION 3.18)
set(PROJECT_NAME)
project(${PROJECT_NAME} C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Use linker group for torch builds to resolve circular dependencies
set(CMAKE_EXE_LINKER_FLAGS "-Wl,--start-group -Wl,--end-group")

add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=1)

option(ENABLE_COVERAGE "Enable converage detection" OFF)
if(ENABLE_COVERAGE)
  message(
    STATUS "Coverage build enabled via ENABLE_COVERAGE environment variable.")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fprofile-arcs -ftest-coverage")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -lgcov")
endif()

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake ${CMAKE_MODULE_PATH})
get_filename_component(THIRD_ROOT "${PROJECT_BINARY_DIR}/3rd_party" ABSOLUTE)

include(external)

find_package(Threads REQUIRED)

set(EXE_TARGET_NAME "all_api_tests")
if(CMAKE_COMPILER_IS_GNUCXX)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -ansi -Wno-deprecated")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -ansi -Wno-deprecated")
endif()

if(WIN32)
  set(_OPT_CMAKE_ARGS
      "-Dgtest_force_shared_crt=ON;-DCMAKE_SH=CMAKE_SH-NOTFOUND")
else()
  set(_OPT_CMAKE_ARGS "")
endif()

externalproject(
  "https://github.com/google/googletest.git" "main" ${THIRD_ROOT} CMAKE_ARGS
  "-DCMAKE_INSTALL_PREFIX=${THIRD_ROOT};${_OPT_CMAKE_ARGS}")

include_directories("${THIRD_ROOT}/include")
link_directories("${THIRD_ROOT}/lib")
link_directories("${THIRD_ROOT}/lib64")

# Set PYTHONPATH to find paddle module
set(PADDLE_PYTHON_PATH "/home/aistudio/Paddle")
set(ENV{PYTHONPATH} "${PADDLE_PYTHON_PATH}:$ENV{PYTHONPATH}")

find_program(Python3_EXECUTABLE python3 REQUIRED)
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

enable_testing()
include_directories(${COMMON_INCLUDES})
include(cmake/build.cmake)

# Only test specified files for now
set(TEST_SRC_FILES ${PROJECT_SOURCE_DIR}/test/ops/IndexTest.cpp)

file(GLOB_RECURSE TEST_BASE_FILES ${PROJECT_SOURCE_DIR}/src/*.cpp)
set(PADDLE_TARGET_FOLDER ${CMAKE_BINARY_DIR}/paddle)

# ---------------------------------------------------------------------------
# Build Torch test case
# ---------------------------------------------------------------------------
set(TORCH_DIR "/home/aistudio/libtorch")

# Add library paths
link_directories("${TORCH_DIR}/lib")
link_directories("${THIRD_ROOT}/lib")

# Use gtest from torch lib directory
set(TORCH_GTEST "${TORCH_DIR}/lib/libgtest.a")
set(TORCH_GTEST_MAIN "${TORCH_DIR}/lib/libgtest_main.a")

# Get all torch libraries
file(GLOB_RECURSE TORCH_ALL_SO "${TORCH_DIR}/lib/*.so")

# Include c10 directly in the list
set(TORCH_LIBRARIES ${TORCH_ALL_SO})

set(TORCH_INCLUDE_DIR
    "${TORCH_DIR}/include" "${TORCH_DIR}/include/torch/csrc/api/include/"
    "${TORCH_DIR}/include/ATen")

message(STATUS "Torch libraries: ${TORCH_LIBRARIES}")
message(STATUS "Torch include: ${TORCH_INCLUDE_DIR}")
message(STATUS "Torch gtest: ${TORCH_GTEST}, ${TORCH_GTEST_MAIN}")

set(TORCH_TARGET_FOLDER ${CMAKE_BINARY_DIR}/torch)
set(BIN_PREFIX "torch_")
set(C10_LIB "${TORCH_DIR}/lib/libc10.so")
message(STATUS "C10 library: ${C10_LIB}")

create_paddle_tests(
  "${BIN_PREFIX}" "${TEST_SRC_FILES}" "${TORCH_TARGET_FOLDER}"
  "${TORCH_LIBRARIES}" "${TORCH_INCLUDE_DIR}" 0)

# ---------------------------------------------------------------------------
# Build Paddle test case (使用 external-libraries + 源码头文件)
# ---------------------------------------------------------------------------
# 库来自 pip 安装的 paddle，头文件来自源码
set(PADDLE_DIR
    "/home/aistudio/external-libraries/lib/python3.10/site-packages/paddle")
set(PADDLE_SOURCE_DIR "/home/aistudio/Paddle")
message(STATUS "Using paddle libs from: ${PADDLE_DIR}")
message(STATUS "Using paddle headers from: ${PADDLE_SOURCE_DIR}")

set(PADDLE_INCLUDE_DIR "${PADDLE_SOURCE_DIR}/")
message(STATUS "PADDLE_INCLUDE_DIR: ${PADDLE_INCLUDE_DIR}")
set(PADDLE_INCLUDE_DIR
    ${PADDLE_INCLUDE_DIR}
    "${PADDLE_SOURCE_DIR}/paddle/phi/api/include"
    "${PADDLE_SOURCE_DIR}/paddle/phi/api/include/compat/"
    "${PADDLE_SOURCE_DIR}/paddle/phi/api/include/compat/torch/csrc/api/include/"
    "/home/aistudio/external-libraries/lib/python3.10/site-packages/pybind11/include"
)

# Find CUDA and set up GPU libraries
option(ENABLE_GPU "Enable GPU support" OFF)

if(ENABLE_GPU)
  find_package(CUDA QUIET)
  if(CUDA_FOUND)
    message(STATUS "CUDA found: ${CUDA_VERSION}")
    set(CUDA_LIBRARIES ${CUDA_CUDA_LIBRARIES} ${CUDA_CUDART_LIBRARY})
    link_directories(${CUDA_LIBRARY_DIRS})
  else()
    message(STATUS "CUDA not found, trying default paths")
    set(CUDA_LIBRARIES "/usr/local/cuda/lib64/libcudart.so")
    link_directories("/usr/local/cuda/lib64")
  endif()

  set(PADDLE_GPU_LIB "${PADDLE_DIR}/libs/libphi_gpu.so")
endif()

# 使用 pip 安装的库
set(PADDLE_LIBRARIES
    "${PADDLE_DIR}/base/libpaddle.so" "${PADDLE_DIR}/libs/libcommon.so"
    "${PADDLE_DIR}/libs/libphi.so" "${PADDLE_DIR}/libs/libphi_core.so"
    ${PADDLE_GPU_LIB} ${CUDA_LIBRARIES})
link_directories("${PADDLE_DIR}/base")
link_directories("${PADDLE_DIR}/libs")

set(BIN_PREFIX "paddle_")

create_paddle_tests(
  "${BIN_PREFIX}" "${TEST_SRC_FILES}" "${PADDLE_TARGET_FOLDER}"
  "${PADDLE_LIBRARIES}" "${PADDLE_INCLUDE_DIR}" 1)
