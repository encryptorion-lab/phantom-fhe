cmake_minimum_required(VERSION 3.20)

project(Phantom LANGUAGES CXX CUDA VERSION 1.3 DESCRIPTION "CUDA-Accelerated Homomorphic Encryption Library")

# [option] PHANTOM_BENCH (default: OFF)
option(PHANTOM_BENCH "Profiling mode" OFF)
message(STATUS "Profiling mode: ${PHANTOM_BENCH}")
if (PHANTOM_BENCH)
    add_compile_definitions(PHANTOM_BENCH)
endif ()

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

# output the compile option in file compile_commands.json
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# use C++20
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# CUDA separable compilation
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# -fPIC
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin)

# include "include/" directory is enough
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# library
add_subdirectory(src)

# example
add_subdirectory(example)

# benchmark
add_subdirectory(benchmark)

# test
add_subdirectory(test)

# python binding
add_subdirectory(python)

# config for installation
install(TARGETS Phantom EXPORT PhantomConfig)
install(EXPORT PhantomConfig NAMESPACE phantom:: DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/phantom)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/phantom)
