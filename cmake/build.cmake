function(
  create_paddle_tests
  BIN_PREFIX
  TEST_SRC_FILES
  TARGET_FOLDER
  DEPS_LIBRARIES
  INCLUDE_DIR
  USE_PADDLE_API)

  foreach(_test_file ${TEST_SRC_FILES})
    get_filename_component(_file_name ${_test_file} NAME_WE)
    set(_test_name ${BIN_PREFIX}${_file_name})
    add_executable(${_test_name} ${_test_file} ${TEST_BASE_FILES}
                                 ${PROJECT_SOURCE_DIR}/src/file_manager.cpp)
    add_dependencies(${_test_name} "googletest.git")
    target_link_libraries(
      ${_test_name} gtest gtest_main ${CMAKE_THREAD_LIBS_INIT}
      ${DEPS_LIBRARIES} ${Python3_LIBRARIES})
    target_include_directories(${_test_name} PRIVATE ${Python3_INCLUDE_DIRS})
    target_include_directories(${_test_name} PRIVATE ${INCLUDE_DIR}
                                                     ${PROJECT_SOURCE_DIR}/src)
    target_include_directories(${_test_name} PRIVATE ${PROJECT_SOURCE_DIR}/src)
    message(STATUS "include dir: ${INCLUDE_DIR}")
    target_compile_definitions(${_test_name}
                               PRIVATE USE_PADDLE_API=${USE_PADDLE_API})
    if(${USE_PADDLE_API})
      # Paddle's CUDA compat headers (CUDAContextLight.h, CUDAFunctions.h)
      # require PADDLE_WITH_CUDA to be defined so that GPU type aliases
      # (gpuStream_t, cudaDeviceProp) are resolved via cuda_runtime.h.
      target_compile_definitions(${_test_name} PRIVATE PADDLE_WITH_CUDA)
      # Link libcudart for CUDA runtime symbols used by the Paddle CUDA compat
      # layer.
      if(TARGET CUDA::cudart)
        target_link_libraries(${_test_name} CUDA::cudart)
      elseif(EXISTS "/usr/local/cuda/lib64/libcudart.so")
        target_link_libraries(${_test_name}
                              "/usr/local/cuda/lib64/libcudart.so")
      endif()
    endif()
    message(STATUS "USE_PADDLE_API: ${USE_PADDLE_API}")
    add_test(NAME ${_test_name} COMMAND ${_test_name})
    set_tests_properties(${_test_name} PROPERTIES TIMEOUT 5)
    set_target_properties(${_test_name} PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                                   "${TARGET_FOLDER}")
  endforeach()
endfunction()
