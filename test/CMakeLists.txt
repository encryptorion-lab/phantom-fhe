add_executable(nwt_test nwt_test.cu)
target_link_libraries(nwt_test Phantom)
target_include_directories(nwt_test PUBLIC ${CMAKE_SOURCE_DIR}/include)
