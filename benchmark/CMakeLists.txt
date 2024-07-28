add_subdirectory(nvbench)

# example usage of nvbench
add_executable(example_bench example_bench.cu)
target_link_libraries(example_bench nvbench::main)

add_executable(ntt_bench ntt_bench.cu)
target_link_libraries(ntt_bench PRIVATE nvbench::main Phantom)
target_include_directories(ntt_bench PUBLIC ${CMAKE_SOURCE_DIR}/include)

add_executable(keyswitch_bench keyswitch_bench.cu)
target_link_libraries(keyswitch_bench PRIVATE nvbench::main Phantom)
target_include_directories(keyswitch_bench PUBLIC ${CMAKE_SOURCE_DIR}/include)

add_executable(bfv_bench bfv_bench.cu)
target_link_libraries(bfv_bench PRIVATE Phantom)
target_include_directories(bfv_bench PUBLIC ${CMAKE_SOURCE_DIR}/include)

add_executable(bgv_bench bgv_bench.cu)
target_link_libraries(bgv_bench PRIVATE Phantom)
target_include_directories(bgv_bench PUBLIC ${CMAKE_SOURCE_DIR}/include)

add_executable(ckks_bench ckks_bench.cu)
target_link_libraries(ckks_bench PRIVATE Phantom)
target_include_directories(ckks_bench PUBLIC ${CMAKE_SOURCE_DIR}/include)

add_executable(matmul_bench matmul_bench.cu)
target_link_libraries(matmul_bench PRIVATE nvbench::main Phantom)
target_include_directories(matmul_bench PUBLIC ${CMAKE_SOURCE_DIR}/include)
