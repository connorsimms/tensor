include(FetchContent)

FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/52eb8108c5bdec04579160ae17225d66034bd723.zip
  SYSTEM
)

FetchContent_MakeAvailable(googletest)

add_executable(
    tensor_test 
    tensor_test.cpp
)

target_link_libraries(tensor_test PRIVATE
  tensor_lib
  GTest::gtest
  GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(tensor_test)
