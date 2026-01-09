include (FetchContent)

FetchContent_Declare(
    googlebenchmark
    URL https://github.com/google/benchmark/archive/eddb0241389718a23a42db6af5f0164b6e0139af.zip
    SYSTEM
)

FetchContent_MakeAvailable(googlebenchmark)

add_executable(
    tensor_benchmark
    tensor_benchmark.cpp
)

target_link_libraries(tensor_benchmark PRIVATE
    tensor_lib
    benchmark::benchmark
    benchmark::benchmark_main
)

target_compile_options(tensor_benchmark PRIVATE -O3 -march=native)


