cmake_minimum_required(VERSION 3.14)
project(llama_common CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)

# llama.cpp source tree checked out at the SAME tag as the vendored binaries
# (LLAMA_CPP_VERSION in the Rakefile). Only headers are used.
set(LLAMA_CPP_DIR "" CACHE PATH "Path to llama.cpp source at the pinned tag")
# Directory containing the vendored shared libraries (lib/rllama/<platform>),
# including libllama-common shipped with llama.cpp releases.
set(RLLAMA_LIB_DIR "" CACHE PATH "Path to vendored llama shared libraries")

if(NOT LLAMA_CPP_DIR OR NOT RLLAMA_LIB_DIR)
  message(FATAL_ERROR "Set -DLLAMA_CPP_DIR and -DRLLAMA_LIB_DIR")
endif()

add_library(llama_common SHARED llama_common.cpp)
target_compile_definitions(llama_common PRIVATE RLC_BUILD)
target_include_directories(llama_common PRIVATE
  ${LLAMA_CPP_DIR}/common
  ${LLAMA_CPP_DIR}/include
  ${LLAMA_CPP_DIR}/ggml/include
  ${LLAMA_CPP_DIR}/vendor
)

target_link_directories(llama_common PRIVATE ${RLLAMA_LIB_DIR})
target_link_libraries(llama_common PRIVATE llama-common ggml-base)

if(APPLE)
  set_target_properties(llama_common PROPERTIES
    INSTALL_RPATH "@loader_path"
    BUILD_WITH_INSTALL_RPATH ON
  )
elseif(UNIX)
  set_target_properties(llama_common PROPERTIES
    INSTALL_RPATH "$ORIGIN"
    BUILD_WITH_INSTALL_RPATH ON
  )
endif()

install(TARGETS llama_common LIBRARY DESTINATION ${RLLAMA_LIB_DIR})
