add_subdirectory(nvbench)

# example usage of nvbench
add_executable(example_bench example_bench.cu)
target_link_libraries(example_bench nvbench::main)

add_executable(nwt_bench nwt_bench.cu)
target_link_libraries(nwt_bench PRIVATE nvbench::main Phantom)

add_executable(keyswitch_bench keyswitch_bench.cu)
target_link_libraries(keyswitch_bench PRIVATE nvbench::main Phantom)

add_executable(bfv_bench bfv_bench.cu)
add_executable(bgv_bench bgv_bench.cu)
add_executable(ckks_bench ckks_bench.cu)

target_link_libraries(bfv_bench PRIVATE Phantom)
target_link_libraries(bgv_bench PRIVATE Phantom)
target_link_libraries(ckks_bench PRIVATE Phantom)

add_executable(matmul_bench matmul_bench.cu)
target_link_libraries(matmul_bench PRIVATE nvbench::main Phantom)
