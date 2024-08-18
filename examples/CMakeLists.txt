# test SEAL examples
add_executable(example_context
        1_bfv.cu
        2_bgv.cu
        3_ckks.cu
        4_kernel_fusing.cu
        example.cu
)

target_link_libraries(example_context PRIVATE Phantom)
target_include_directories(example_context PUBLIC ${CMAKE_SOURCE_DIR}/include)
