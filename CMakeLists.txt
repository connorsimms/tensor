cmake_minimum_required(VERSION 4.2)
project(tensor VERSION 0.0.1 LANGUAGES CXX)

enable_testing()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(tensor_lib INTERFACE)
target_include_directories(tensor_lib INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_options(tensor_lib INTERFACE -Wall -Wextra -Wpedantic)

if(SKBUILD)

    find_package(Python 3.13 REQUIRED COMPONENTS Interpreter Development.Module)
    find_package(nanobind CONFIG REQUIRED)

    nanobind_add_module(tensor bindings/python/tensor.cpp)
    target_link_libraries(tensor PRIVATE tensor_lib)
    install(TARGETS tensor DESTINATION .)

else()
    add_subdirectory(tests)
    add_subdirectory(benchmarks)

    find_program(CLANG_TIDY_EXE NAMES "clang-tidy")
    if(CLANG_TIDY_EXE)
        set(TIDY_CHECKS "-*,google-*,performance-*,bugprone-*,-readability-magic-numbers,-misc-include-cleaner")
        
        set_target_properties(
            tensor_test PROPERTIES 
            CXX_CLANG_TIDY "${CLANG_TIDY_EXE};-checks=${TIDY_CHECKS}"
        )
    endif()

endif()
