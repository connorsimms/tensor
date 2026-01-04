#pragma once

#include <vector>

template <typename T>
class Tensor
{
public:
    Tensor () : data_{} {}

    Tensor (std::size_t size) : data_(size) {}

    std::string dummy() { return std::string("Hello World\n"); }

private:
    std::vector<T> data_;
};
