set(CMAKE_CXX_STANDARD 17)

function(add_dsg_app TARGET_NAME SOURCE_FILE)
    add_executable(${TARGET_NAME} ${SOURCE_FILE})
    target_include_directories(${TARGET_NAME} PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/include/utils
        ${PROJECT_SOURCE_DIR}/src
    )
    target_link_libraries(${TARGET_NAME} PRIVATE dsg_core UTIL OpenMP::OpenMP_CXX)
    target_compile_options(${TARGET_NAME} PRIVATE -Wall -march=native -O3 ${OpenMP_CXX_FLAGS})
    target_link_libraries(${TARGET_NAME} PRIVATE ${OpenMP_CXX_FLAGS})
endfunction()

add_dsg_app(generate_groundtruth generate_groundtruth.cc)
add_dsg_app(build_index build_index.cc)
add_dsg_app(query_index query_index.cc)

if(DSG_BUILD_INDEX_DEBUG)
    add_dsg_app(build_index_debug build_index.cc)
    target_compile_definitions(build_index_debug PRIVATE DSG_DEBUG_BUILD)
    target_compile_options(build_index_debug PRIVATE -O0 -g)
endif()
