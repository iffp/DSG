cmake_minimum_required(VERSION 3.16)
project(RangeFilteringTests LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(ENABLE_AVX512 "Enable AVX-512 instructions for the SIMD benchmark" OFF)

add_executable(test_range_benchmark
    tests/test_range_benchmark.cpp
)

add_executable(test_range_benchmark_simd
    tests/test_range_benchmark_simd.cpp
)
target_compile_options(test_range_benchmark_simd PRIVATE -mavx2 -mfma)
if(ENABLE_AVX512)
    target_compile_options(test_range_benchmark_simd PRIVATE -mavx512f)
endif()

add_executable(test_l2_benchmark
    tests/test_l2_benchmark.cpp
)
