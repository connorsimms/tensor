#pragma once

#include <cstdint>
#include <functional>
#include <numeric>
#include <utility>
#include <vector>

template <typename T>
class Tensor
{
public:
    Tensor (const std::vector<std::uint32_t>& shape) 
    : data_(std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<std::uint32_t>()))
    , shape_{shape}
    , stride_(shape.size())
    {
        std::transform(
            shape_.rbegin(), 
            shape_.rend(), 
            stride_.rbegin(), 
            [&, n = 1](const std::uint32_t dim) mutable {
                auto next = n;
                n *= dim;
                return next;
            }
        );
    }

    std::vector<T> getData() const { return data_; }
    std::vector<std::uint32_t> getShape() const { return shape_; }
    std::vector<std::uint32_t> getStride() const { return stride_; }

    std::string getInfo()
    {
        std::string info {};
        for (auto d: shape_) info += std::to_string(d) + " ";
        info += "\n";
        for (auto d: stride_) info += std::to_string(d) + " "; 
        info += "\n";
        return info;
    }

private:
    std::vector<T> data_;
    std::vector<std::uint32_t> shape_;
    std::vector<std::uint32_t> stride_;
};
