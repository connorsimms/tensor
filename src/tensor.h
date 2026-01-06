#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <functional>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

template <typename T> class Tensor
{
public:
  Tensor(const std::vector<std::uint32_t> &shape)
      : data_(std::accumulate(shape.begin(), shape.end(), 1,
                              std::multiplies<std::uint32_t>())),
        shape_{shape}, stride_(shape.size())
  {
    std::transform(shape_.rbegin(), shape_.rend(), stride_.rbegin(),
                   [&, n = 1](const std::uint32_t dim) mutable
                   {
                     auto next = n;
                     n *= dim;
                     return next;
                   });
  }

  const std::vector<T> &getData() const { return data_; }
  const std::vector<std::uint32_t> &getShape() const { return shape_; }
  const std::vector<std::uint32_t> &getStride() const { return stride_; }

  T &at(const std::vector<std::uint32_t> &indices)
  {
    if (indices.size() != shape_.size())
    {
      throw std::invalid_argument(
          "Number of arguments does not match dimension");
    }

    if (!std::equal(indices.begin(), indices.end(), shape_.begin(), [](auto idx, auto bound) {
                return idx < bound;
        }))
    {
        throw std::invalid_argument( "Index arguments are out of bounds");
    }

    std::size_t pos =
        std::inner_product(stride_.begin(), stride_.end(), indices.begin(), 0u);

    return data_[pos];
  }

  template <class... Args> T &operator[](Args... args)
  {
    return at({static_cast<std::uint32_t>(args)...});
  }

private:
  std::vector<T> data_;
  std::vector<std::uint32_t> shape_;
  std::vector<std::uint32_t> stride_;
};
