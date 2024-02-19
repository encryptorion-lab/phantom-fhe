add_subdirectory(pybind11)
pybind11_add_module(pyPhantom src/binding.cu)
target_link_libraries(pyPhantom PRIVATE Phantom)
