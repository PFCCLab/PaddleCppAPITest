include(ExternalProject)

function(ExternalProject repourl tag destination)

  message(STATUS "Get external project from: ${repourl} : ${tag}")

  string(REGEX MATCH "([^\\/]+)[.]git$" _name ${repourl})
  message(STATUS "_name = ${_name}")

  set(options)
  set(oneValueArgs)
  set(multiValueArgs CMAKE_ARGS)
  cmake_parse_arguments(ExternalProject "${options}" "${oneValueArgs}"
                        "${multiValueArgs}" ${ARGN})

  set(cmake_cli_args -DCMAKE_INSTALL_PREFIX=${destination}
                     -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE})
  if(CMAKE_C_COMPILER_LAUNCHER)
    list(APPEND cmake_cli_args
         -DCMAKE_C_COMPILER_LAUNCHER=${CMAKE_C_COMPILER_LAUNCHER})
  endif()
  if(CMAKE_CXX_COMPILER_LAUNCHER)
    list(APPEND cmake_cli_args
         -DCMAKE_CXX_COMPILER_LAUNCHER=${CMAKE_CXX_COMPILER_LAUNCHER})
  endif()
  if(CMAKE_CUDA_COMPILER_LAUNCHER)
    list(APPEND cmake_cli_args
         -DCMAKE_CUDA_COMPILER_LAUNCHER=${CMAKE_CUDA_COMPILER_LAUNCHER})
  endif()
  if(CMAKE_TOOLCHAIN_FILE)
    get_filename_component(_ft_path ${CMAKE_TOOLCHAIN_FILE} ABSOLUTE)
    get_filename_component(_cm_rt_opath ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
                           ABSOLUTE)
    set(cmake_cli_args ${cmake_cli_args} -DCMAKE_TOOLCHAIN_FILE=${_ft_path}
                       -DCMAKE_RUNTIME_OUTPUT_DIRECTORY=${_cm_rt_opath})
  endif()

  foreach(cmake_key ${ExternalProject_CMAKE_ARGS})
    list(APPEND cmake_cli_args ${cmake_key})
  endforeach()

  message(STATUS "ARGS for ExternalProject_Add(${name}): ${cmake_cli_args}")
  message(STATUS "CMAKE_CXX_FLAGS = ${CMAKE_CXX_FLAGS}")

  ExternalProject_Add(
    ${_name}
    GIT_REPOSITORY ${repourl}
    GIT_TAG ${tag}
    UPDATE_COMMAND ""
    CMAKE_ARGS -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
               -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER} ${cmake_cli_args}
               -DCMAKE_CXX_STANDARD=17
    PREFIX "${destination}"
    INSTALL_DIR "${destination}"
    INSTALL_COMMAND "${CMAKE_COMMAND}" --install "<BINARY_DIR>" --prefix
                    "${destination}"
    BUILD_BYPRODUCTS
      "${destination}/lib/libgtest.a" "${destination}/lib/libgtest_main.a"
      "${destination}/lib/libgmock.a" "${destination}/lib64/libgtest.a"
      "${destination}/lib64/libgtest_main.a" "${destination}/lib64/libgmock.a")
endfunction()
