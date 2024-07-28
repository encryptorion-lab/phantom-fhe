add_executable(ntt_test ntt_test.cu)
target_link_libraries(ntt_test Phantom)
target_include_directories(ntt_test PUBLIC ${CMAKE_SOURCE_DIR}/include)

# find CUDAToolkit to enable cuFFT
find_package(CUDAToolkit REQUIRED)

add_executable(fft_test fft_test.cu)
target_link_libraries(fft_test PUBLIC Phantom CUDA::cufft)
target_include_directories(fft_test PUBLIC ${CMAKE_SOURCE_DIR}/include)
