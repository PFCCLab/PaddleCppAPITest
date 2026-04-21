# Use ccache if found ccache program

if(NOT WIN32)
  find_program(CCACHE_PATH ccache)
  if(CCACHE_PATH)
    execute_process(COMMAND ccache -V OUTPUT_VARIABLE ccache_output)
    execute_process(COMMAND ccache -v -s cache directory
                    OUTPUT_VARIABLE cache_directory)
    execute_process(COMMAND ccache -p OUTPUT_VARIABLE ccache_config)
    string(REGEX MATCH "[0-9]+.[0-9]+" ccache_version ${ccache_output})
    string(REGEX MATCH "cache_dir = ([^\n\r]+)" _cache_dir_match
                 "${ccache_config}")
    set(CCACHE_DIR_PATH "${CMAKE_MATCH_1}")

    if(CCACHE_DIR_PATH)
      set(_ccache_write_probe "${CCACHE_DIR_PATH}/.cmake_ccache_write_probe")
      execute_process(
        COMMAND ${CMAKE_COMMAND} -E touch "${_ccache_write_probe}"
        RESULT_VARIABLE ccache_write_result
        ERROR_QUIET)
      if(ccache_write_result EQUAL 0)
        execute_process(COMMAND ${CMAKE_COMMAND} -E rm -f
                                "${_ccache_write_probe}")
        message(
          STATUS "ccache is founded, use ccache to speed up compile on Unix.")
        # show statistics summary of ccache
        message("ccache version\t\t\t    " ${ccache_version} "\n"
                ${cache_directory})
        set(CMAKE_C_COMPILER_LAUNCHER ${CCACHE_PATH})
        set(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE_PATH})
        set(CMAKE_CUDA_COMPILER_LAUNCHER ${CCACHE_PATH})
      else()
        message(
          STATUS
            "ccache cache directory is not writable; skip ccache for this build."
        )
      endif()
    else()
      message(STATUS "Failed to detect ccache cache directory; skip ccache.")
    endif()
  endif()
elseif("${CMAKE_GENERATOR}" STREQUAL "Ninja")
  # Only Ninja Generator can support sccache now
  find_program(SCCACHE_PATH sccache)
  if(SCCACHE_PATH)
    execute_process(COMMAND sccache -V OUTPUT_VARIABLE sccache_version)
    message(
      STATUS
        "sccache is founded, use [${SCCACHE_PATH}] to speed up compile on Windows."
    )
    set(CMAKE_C_COMPILER_LAUNCHER ${SCCACHE_PATH})
    set(CMAKE_CXX_COMPILER_LAUNCHER ${SCCACHE_PATH})
  endif()
endif()
