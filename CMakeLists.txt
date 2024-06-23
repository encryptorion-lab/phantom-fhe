cmake_minimum_required(VERSION 3.20)

project(Phantom LANGUAGES CXX CUDA VERSION 2.0 DESCRIPTION "CUDA-Accelerated Homomorphic Encryption Library")

# [option] PHANTOM_USE_CUDA_PTX (default: ON)
option(PHANTOM_USE_CUDA_PTX "Use CUDA PTX Assembly" ON)
message(STATUS "Use CUDA PTX Assembly: ${PHANTOM_USE_CUDA_PTX}")
if (PHANTOM_USE_CUDA_PTX)
    add_compile_definitions(PHANTOM_USE_CUDA_PTX)
endif ()

if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif ()

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

# library
add_subdirectory(src)

# example
# [option] PHANTOM_ENABLE_EXAMPLE (default: ON)
option(PHANTOM_ENABLE_EXAMPLE "Enable examples" ON)
message(STATUS "Enable examples: ${PHANTOM_ENABLE_EXAMPLE}")
if (PHANTOM_ENABLE_EXAMPLE)
    add_subdirectory(example)
endif ()

# benchmark
# [option] PHANTOM_ENABLE_BENCH (default: ON)
option(PHANTOM_ENABLE_BENCH "Enable benchmarks" ON)
message(STATUS "Enable benchmarks: ${PHANTOM_ENABLE_BENCH}")
if (PHANTOM_ENABLE_BENCH)
    add_subdirectory(benchmark)
endif ()

# test
# [option] PHANTOM_ENABLE_TEST (default: ON)
option(PHANTOM_ENABLE_TEST "Enable tests" ON)
message(STATUS "Enable tests: ${PHANTOM_ENABLE_TEST}")
if (PHANTOM_ENABLE_TEST)
    add_subdirectory(test)
endif ()

# python binding
# [option] PHANTOM_ENABLE_PYTHON_BINDING (default: ON)
option(PHANTOM_ENABLE_PYTHON_BINDING "Enable python binding" ON)
message(STATUS "Enable python binding: ${PHANTOM_ENABLE_PYTHON_BINDING}")
if (PHANTOM_ENABLE_PYTHON_BINDING)
    add_subdirectory(python)
endif ()

# config for installation
install(TARGETS Phantom EXPORT PhantomConfig)
install(EXPORT PhantomConfig NAMESPACE phantom:: DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/phantom)
install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/phantom)
