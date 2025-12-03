cmake_minimum_required(VERSION 3.16)
project(DynamicSegmentGraph LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenMP REQUIRED)

add_subdirectory(src/utils)

add_library(dsg_core STATIC src/dsg.cc)
target_include_directories(dsg_core PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/include/utils
    ${PROJECT_SOURCE_DIR}/src
)
target_link_libraries(dsg_core PRIVATE UTIL OpenMP::OpenMP_CXX)
target_compile_options(dsg_core PRIVATE -Wall -march=native -O3 ${OpenMP_CXX_FLAGS})

# -----------------------------------------------------------------------------
# Release vs Debug Configuration
# -----------------------------------------------------------------------------
# By default, dsg_core builds with high optimization (-O3) and native SIMD support (-march=native).
#
# To enable Debug mode (with symbols and no optimization), enable the option below:
#   cmake -B build -DDSG_BUILD_INDEX_DEBUG=ON ...
#
# To enable Benchmark executables (e.g. test_hnsw_range), use:
#   cmake -B build -DBUILD_RANGE_BENCHMARKS=ON ...
#
# -----------------------------------------------------------------------------
option(DSG_BUILD_INDEX_DEBUG "Build a debug-instrumented build_index target" OFF)
if(DSG_BUILD_INDEX_DEBUG)
    target_compile_options(dsg_core PRIVATE -O0 -g)
    target_compile_definitions(dsg_core PRIVATE DSG_DEBUG_BUILD)
endif()

add_subdirectory(apps)

option(BUILD_RANGE_BENCHMARKS "Build range benchmark executables under tests/" OFF)
option(ENABLE_AVX512 "Enable AVX-512 instructions for the SIMD benchmark" OFF)

if(BUILD_RANGE_BENCHMARKS)
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

    add_executable(test_hnsw_range
        tests/test_hnsw_range.cpp
    )
    target_include_directories(test_hnsw_range PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/include/utils
    )
    target_link_libraries(test_hnsw_range PRIVATE UTIL)
    target_compile_options(test_hnsw_range PRIVATE -march=native -O3)
endif()
