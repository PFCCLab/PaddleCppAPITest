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

externalproject("https://github.com/google/googletest.git" "main" ${THIRD_ROOT}
                CMAKE_ARGS "${_OPT_CMAKE_ARGS}")

include_directories("${THIRD_ROOT}/include")
link_directories("${THIRD_ROOT}/lib")
link_directories("${THIRD_ROOT}/lib64")

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

file(GLOB TEST_SRC_FILES ${PROJECT_SOURCE_DIR}/test/*.cpp
     ${PROJECT_SOURCE_DIR}/test/ops/*.cpp)
# 排除已unmatch开头的文件
file(GLOB UNMATCH_FILES ${PROJECT_SOURCE_DIR}/test/unmatch*.cpp
     ${PROJECT_SOURCE_DIR}/test/ops/unmatch*.cpp)

list(REMOVE_ITEM TEST_SRC_FILES ${UNMATCH_FILES})

file(GLOB_RECURSE TEST_BASE_FILES ${PROJECT_SOURCE_DIR}/src/*.cpp)
set(PADDLE_TARGET_FOLDER ${CMAKE_BINARY_DIR}/paddle)

# ---------------------------------------------------------------------------
# CUDA Toolkit (needed for CUDA-specific test headers in the Torch build)
# ---------------------------------------------------------------------------
find_package(CUDAToolkit QUIET)
if(CUDAToolkit_FOUND)
  message(STATUS "Found CUDA Toolkit: ${CUDAToolkit_INCLUDE_DIRS}")
  set(CUDA_INCLUDE_DIRS "${CUDAToolkit_INCLUDE_DIRS}")
elseif(EXISTS "/usr/local/cuda/include")
  set(CUDA_INCLUDE_DIRS "/usr/local/cuda/include")
  message(STATUS "Using default CUDA include dir: ${CUDA_INCLUDE_DIRS}")
else()
  message(WARNING "CUDA headers not found; CUDA tests may not compile.")
  set(CUDA_INCLUDE_DIRS "")
endif()

# ---------------------------------------------------------------------------
# Build Torch test case
# ---------------------------------------------------------------------------
set(TORCH_DIR
    "/usr/lib/libtorch/"
    CACHE PATH "Path to libtorch installation")

set(TORCH_LIBRARIES "")
file(GLOB_RECURSE TORCH_LIBRARIES "${TORCH_DIR}/lib/*.so"
     "${TORCH_DIR}/lib/*.a")

find_package(CUDAToolkit QUIET)
set(TORCH_INCLUDE_DIR "${TORCH_DIR}/include"
                      "${TORCH_DIR}/include/torch/csrc/api/include/")
if(CUDAToolkit_FOUND)
  list(APPEND TORCH_INCLUDE_DIR "${CUDAToolkit_INCLUDE_DIRS}")
endif()

set(TORCH_TARGET_FOLDER ${CMAKE_BINARY_DIR}/torch)
set(BIN_PREFIX "torch_")
create_paddle_tests(
  "${BIN_PREFIX}" "${TEST_SRC_FILES}" "${TORCH_TARGET_FOLDER}"
  "${TORCH_LIBRARIES}" "${TORCH_INCLUDE_DIR}" 0)

# ---------------------------------------------------------------------------
# Build Paddle test case
# ---------------------------------------------------------------------------
execute_process(
  COMMAND ${Python3_EXECUTABLE} -c "import paddle; print(paddle.__path__[0])"
  OUTPUT_VARIABLE PADDLE_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE result)

if(result EQUAL 0)
  message(STATUS "Found package at: ${PADDLE_DIR}")
else()
  message(FATAL_ERROR "Package not found!")
endif()

set(PADDLE_INCLUDE_DIR "${PADDLE_DIR}/")
message(STATUS "PADDLE_INCLUDE_DIR: ${PADDLE_INCLUDE_DIR}")
set(PADDLE_INCLUDE_DIR
    ${PADDLE_INCLUDE_DIR}
    "${PADDLE_DIR}/include"
    "${PADDLE_DIR}/include/third_party"
    "${PADDLE_DIR}/include/paddle/phi/api/include/compat/"
    "${PADDLE_DIR}/include/paddle/phi/api/include/compat/torch/csrc/api/include/"
)
if(CUDAToolkit_FOUND)
  list(APPEND PADDLE_INCLUDE_DIR "${CUDAToolkit_INCLUDE_DIRS}")
  list(APPEND PADDLE_INCLUDE_DIR "${CUDAToolkit_INCLUDE_DIRS}/cccl")
endif()

set(PADDLE_LIBRARIES
    "${PADDLE_DIR}/base/libpaddle.so" "${PADDLE_DIR}/libs/libcommon.so"
    "${PADDLE_DIR}/libs/libphi.so" "${PADDLE_DIR}/libs/libphi_core.so"
    "${PADDLE_DIR}/libs/libphi_gpu.so")
if(CUDAToolkit_FOUND)
  list(APPEND PADDLE_LIBRARIES CUDA::cudart)
endif()
link_directories("${PADDLE_DIR}/base")
link_directories("${PADDLE_DIR}/libs")

set(BIN_PREFIX "paddle_")

# Detect CUDA toolkit to enable PADDLE_WITH_CUDA for Paddle tests that need
# CUDA-specific Paddle compat headers (e.g. CUDAGeneratorImpl).
find_package(CUDAToolkit QUIET)
if(CUDAToolkit_FOUND)
  message(STATUS "CUDAToolkit found: ${CUDAToolkit_INCLUDE_DIRS}")
  # CUDA 13+ ships Thrust inside cccl/; add it so <thrust/...> resolves.
  set(_paddle_cuda_incs "${CUDAToolkit_INCLUDE_DIRS}"
                        "${CUDAToolkit_INCLUDE_DIRS}/cccl")
  set(_paddle_cuda_defs "PADDLE_WITH_CUDA")
  # CUDA runtime is needed by Paddle GPU compat headers.
  list(APPEND PADDLE_LIBRARIES CUDA::cudart)
else()
  set(_paddle_cuda_incs "")
  set(_paddle_cuda_defs "")
endif()

create_paddle_tests(
  "${BIN_PREFIX}"
  "${TEST_SRC_FILES}"
  "${PADDLE_TARGET_FOLDER}"
  "${PADDLE_LIBRARIES}"
  "${PADDLE_INCLUDE_DIR}"
  1
  EXTRA_DEFS
  ${_paddle_cuda_defs}
  EXTRA_INCS
  ${_paddle_cuda_incs})
