add_subdirectory(pybind11)
pybind11_add_module(pyPhantom src/binding.cu)
target_link_libraries(pyPhantom PRIVATE Phantom)
target_include_directories(pyPhantom PUBLIC ${CMAKE_SOURCE_DIR}/include)
