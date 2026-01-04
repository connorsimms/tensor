cmake_minimum_required(VERSION 4.2)
project(tensor VERSION 0.0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)
FetchContent_Declare(
    pybind11
    URL https://github.com/pybind/pybind11/archive/f5fbe867d2d26e4a0a9177a51f6e568868ad3dc8.zip
    SYSTEM
)
FetchContent_MakeAvailable(pybind11)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

pybind11_add_module(tensor
    src/main.cpp
)

target_include_directories(tensor PRIVATE src)
