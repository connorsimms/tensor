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

  Tensor(Tensor const &other)
      : data_{other.data_}, shape_{other.shape_}, stride_{other.stride_}
  {
  }

  Tensor(Tensor const *other)
      : data_{other->data_}, shape_{other->shape_}, stride_{other->stride_}
  {
  }

  const std::vector<T> &getData() const { return data_; }
  const std::vector<std::uint32_t> &getShape() const { return shape_; }
  const std::vector<std::uint32_t> &getStride() const { return stride_; }

  void fill(T val) { std::fill(data_.begin(), data_.end(), val); }

  Tensor clone() const { return Tensor(this); }

  T &at(const std::vector<std::uint32_t> &indices)
  {
    if (indices.size() != shape_.size())
    {
      throw std::invalid_argument(
          "Number of arguments does not match dimension");
    }

    if (!std::equal(indices.begin(), indices.end(), shape_.begin(),
                    [](auto idx, auto bound) { return idx < bound; }))
    {
      throw std::invalid_argument("Index arguments are out of bounds");
    }

    std::size_t pos =
        std::inner_product(stride_.begin(), stride_.end(), indices.begin(), 0u);

    return data_[pos];
  }

  template <class... Args> T &operator()(Args... args)
  {
    if (sizeof...(args) != shape_.size())
    {
      throw std::invalid_argument(
          "Number of arguments does not match dimension");
    }

    std::array<std::uint32_t, sizeof...(args)> indices = {
        static_cast<std::uint32_t>(args)...};

    if (!std::equal(indices.begin(), indices.end(), shape_.begin(),
                    [](auto idx, auto bound) { return idx < bound; }))
    {
      throw std::invalid_argument("Index arguments are out of bounds");
    }

    std::size_t pos =
        std::inner_product(stride_.begin(), stride_.end(), indices.begin(), 0u);

    return data_[pos];
  }

  Tensor operator+(Tensor const &other) const
  {
    if (shape_ != other.shape_)
    {
      throw std::invalid_argument("Tensors are of different shape");
    }

    Tensor result = Tensor(shape_);

    std::transform(data_.begin(), data_.end(), other.data_.begin(),
                   result.data_.begin(),
                   [](T const &a, T const &b) { return a + b; });

    return result;
  }

  Tensor operator*(Tensor const &other) const
  {
    if (shape_ != other.shape_)
    {
      throw std::invalid_argument("Tensors are of different shape");
    }

    Tensor result = Tensor(shape_);

    std::transform(data_.begin(), data_.end(), other.data_.begin(),
                   result.data_.begin(),
                   [](T const &a, T const &b) { return a * b; });

    return result;
  }

  Tensor matmul(Tensor const &other) const
  {
    if (shape_.size() != 2 || other.shape_.size() != 2)
    {
      throw std::invalid_argument("MatMul not defined for non-2D tensors");
    }

    std::uint32_t M = shape_[0];
    std::uint32_t K = shape_[1];

    if (other.shape_[0] != K)
    {
      throw std::invalid_argument(
          "Dimensions incompatible: inner dimensions must match");
    }

    std::uint32_t N = other.shape_[1];

    Tensor result({M, N});

    for (std::size_t i{}; i < M; ++i)
    {
      for (std::size_t j{}; j < N; ++j)
      {
        auto idx = i * result.stride_[0] + j * result.stride_[1];
        for (std::size_t k{}; k < K; ++k)
        {
          auto a = this->data_[i * this->stride_[0] + k * this->stride_[1]];
          auto b = other.data_[k * other.stride_[0] + j * other.stride_[1]];
          result.data_[idx] += a * b;
        }
      }
    }

    return result;
  }

  void transpose_(std::size_t dimA, std::size_t dimB)
  {
    std::swap(shape_[dimA], shape_[dimB]);

    std::swap(stride_[dimA], stride_[dimB]);
  }

  static Tensor transpose(Tensor const &other, std::size_t dimA,
                          std::size_t dimB)
  {
    Tensor result = Tensor(other);

    result.transpose_(dimA, dimB);

    return result;
  }

  static Tensor relu(Tensor const &other)
  {
    Tensor result = Tensor(other);

    std::transform(result.data_.begin(), result.data_.end(),
                   result.data_.begin(),
                   [](T val) { return std::max(static_cast<T>(0), val); });
    return result;
  }

  static Tensor relu_backward(const Tensor &input, const Tensor &grad_out)
  {
    Tensor result = Tensor(input.shape_);

    std::transform(grad_out.data_.begin(), grad_out.data_.end(),
                   input.data_.begin(), result.data_.begin(),
                   [](const auto g, const auto i) { return (i > 0 ? g : 0); });

    return result;
  }

private:
  std::vector<T> data_;
  std::vector<std::uint32_t> shape_;
  std::vector<std::uint32_t> stride_;
};
