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
  if(CMAKE_TOOLCHAIN_FILE)
    get_filename_component(_ft_path ${CMAKE_TOOLCHAIN_FILE} ABSOLUTE)
    get_filename_component(_cm_rt_opath ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
                           ABSOLUTE)
    set(cmake_cli_args ${cmake_cli_args} -DCMAKE_TOOLCHAIN_FILE=${_ft_path}
                       -DCMAKE_RUNTIME_OUTPUT_DIRECTORY=${_cm_rt_opath})
  endif()

  foreach(cmake_key ${ExtProjectGit_CMAKE_ARGS})
    set(cmake_cli_args ${cmake_key} ${cmake_cli_args})
  endforeach()

  message(STATUS "ARGS for ExternalProject_Add(${name}): ${cmake_cli_args}")
  message(STATUS "CMAKE_CXX_FLAGS = ${CMAKE_CXX_FLAGS}")

  set(_local_source_dir "${PROJECT_SOURCE_DIR}/build/3rd_party/src/${_name}")
  if(EXISTS "${_local_source_dir}/CMakeLists.txt")
    message(STATUS "Reuse local external source: ${_local_source_dir}")
    ExternalProject_Add(
      ${_name}
      SOURCE_DIR "${_local_source_dir}"
      CMAKE_ARGS -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
                 -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER} ${cmake_cli_args}
                 -DCMAKE_CXX_STANDARD=17
      PREFIX "${destination}"
      INSTALL_DIR "${destination}"
      DOWNLOAD_COMMAND ""
      UPDATE_COMMAND ""
      INSTALL_COMMAND "${CMAKE_COMMAND}" --install "<BINARY_DIR>" --prefix
                      "${destination}")
  else()
    ExternalProject_Add(
      ${_name}
      GIT_REPOSITORY ${repourl}
      GIT_TAG ${tag}
      CMAKE_ARGS -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
                 -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER} ${cmake_cli_args}
                 -DCMAKE_CXX_STANDARD=17
      PREFIX "${destination}"
      INSTALL_DIR "${destination}"
      UPDATE_COMMAND ""
      INSTALL_COMMAND "${CMAKE_COMMAND}" --install "<BINARY_DIR>" --prefix
                      "${destination}")
  endif()
endfunction()
