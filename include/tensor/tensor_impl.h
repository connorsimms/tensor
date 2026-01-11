#pragma once

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <iomanip>
#include <functional>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <vector>

template <typename T>
struct TensorImpl
{
    std::vector<std::uint32_t> shape_;
    std::vector<std::uint32_t> stride_;
    std::shared_ptr<std::vector<T>> data_;

    bool requires_grad_;
    std::vector<std::shared_ptr<TensorImpl>> parents_;
    std::shared_ptr<TensorImpl> grad_;
    std::function<void()> backward_;

    template <std::same_as<std::uint32_t>... Args>
    TensorImpl(Args... args)
    : shape_{ args... }
    , stride_(shape_.size())
    , data_(std::make_shared<std::vector<T>>(std::accumulate(shape_.begin(), shape_.end(), 1, std::multiplies<std::uint32_t>())))
    , requires_grad_{ false }
    , parents_{}
    , grad_{ nullptr }
    , backward_{}
    {
        std::transform(shape_.rbegin(), shape_.rend(), stride_.rbegin(),
                       [&, n = 1](const std::uint32_t dim) mutable
                       {
                         auto next = n;
                         n *= dim;
                         return next;
                       });
    }

    TensorImpl(std::vector<std::uint32_t> const &shape)
    : shape_{ shape }
    , stride_(shape_.size())
    , data_(std::make_shared<std::vector<T>>(std::accumulate(shape_.begin(), shape_.end(), 1, std::multiplies<std::uint32_t>())))
    , requires_grad_{ false }
    , parents_{}
    , grad_{ nullptr }
    , backward_{}
    {
        std::transform(shape_.rbegin(), shape_.rend(), stride_.rbegin(),
                       [&, n = 1](const std::uint32_t dim) mutable
                       {
                         auto next = n;
                         n *= dim;
                         return next;
                       });
    }

    // deep copy everything except data & grad
    // may create view version
    TensorImpl(TensorImpl const &other)
    : shape_{ other.shape_ }
    , stride_{ other.stride_ }
    , data_{ other.data_ }
    , requires_grad_{ other.requires_grad_ }
    , parents_{ other.parents_ }
    , grad_{ other.grad_ }
    , backward_{ other.backward_ }
    {
    }

    void fill(T const& val)
    {
        std::fill(data_->begin(), data_->end(), val);
    }

    TensorImpl operator-() const
    {
        TensorImpl result = TensorImpl(this->shape_);

        std::transform(this->data_->begin(), this->data_->end(),
                       result.data_->begin(), [](T const& a) { return -a; });

        return result;
    }

    TensorImpl operator+(TensorImpl const& other) const
    {
        if (this->shape_ != other.shape_)
        {
            throw std::invalid_argument("Tensors are of different shape");
        }

        TensorImpl result = TensorImpl(this->shape_);

        std::transform(this->data_->begin(), this->data_->end(), 
                       other.data_->begin(), result.data_->begin(),
                       [](T const &a, T const &b) { return a + b; });

        return result;
    }

    TensorImpl operator-(TensorImpl const& other) const
    {
        if (this->shape_ != other.shape_)
        {
            throw std::invalid_argument("Tensors are of different shape");
        }

        TensorImpl result = TensorImpl(this->shape_);

        std::transform(this->data_->begin(), this->data_->end(), 
                       other.data_->begin(), result.data_->begin(),
                       [](T const &a, T const &b) { return a - b; });

        return result;
    }

    TensorImpl operator*(T const& val) const
    {
        TensorImpl result = TensorImpl(this->shape_);

        std::transform(this->data_->begin(), this->data_->end(),
                       result.data_->begin(), [val](T const& a) { return a * val; });

        return result;
    }

    void operator+=(TensorImpl const &other)
    {
        if (this->shape_ != other.shape_)
        {
            throw std::invalid_argument("TensorImpl are of different shape");
        }

        std::transform(this->data_->begin(), this->data_->end(), other.data_->begin(),
                       this->data_->begin(), [](T const& a, T const& b)
                       { return a + b; });
    }
    
    void operator-=(TensorImpl const &other)
    {
        if (this->shape_ != other.shape_)
        {
            throw std::invalid_argument("TensorImpl are of different shape");
        }

        std::transform(this->data_->begin(), this->data_->end(), other.data_->begin(),
                       this->data_->begin(), [](T const& a, T const& b)
                       { return a - b; });
    }

    void operator*=(T const& val) const
    {
        std::transform(this->data_->begin(), this->data_->end(),
                       this->data_->begin(), [val](T const& a) { return a * val; });
    }

    template <std::same_as<std::uint32_t>... Args> 
    T& operator[](Args... args)
    {
        if (sizeof...(args) != shape_.size()) 
        {
            throw std::invalid_argument("Number of arguments mismatch dimension");
        }

        std::array<std::uint32_t, sizeof...(args)> indices = { args... };

        if (!std::equal(indices.begin(), indices.end(), shape_.begin(),
                       [](T const& idx, T const& bound) { return idx < bound; }))
        {
            throw std::invalid_argument("Indices are out of bounds");
        }

        auto pos = std::inner_product(indices.begin(), indices.end(), stride_.begin(), 0u);

        return (*data_)[pos];
    }

    TensorImpl transpose(std::uint32_t dimA, std::uint32_t dimB) const
    {
        if (dimA >= shape_.size() || dimB >= shape_.size())
        {
            throw std::invalid_argument("Dimensions are out of bounds");
        }

        TensorImpl result = TensorImpl(this->shape_); 

        std::copy(this->data_->begin(), this->data_->end(), result.data_->begin());
        
        std::swap(result.stride_[dimA], result.stride_[dimB]);
        std::swap(result.shape_[dimA], result.shape_[dimB]);

        return result;
    }

    TensorImpl matmul(TensorImpl const& other) const
    {
        if (shape_.size() != 2 || other.shape_.size() != 2)
        {
            throw std::invalid_argument("MatMul not defined for non-2D tensors");
        }

        std::uint32_t K = shape_[1];

        if (K != other.shape_[0])
        {
            throw std::invalid_argument("Inner dimensions must match");
        }

        std::uint32_t M = shape_[0];
        std::uint32_t N = other.shape_[1];

        TensorImpl result(M, N);

        for (std::size_t i{}; i < M; ++i)
        {
            for (std::size_t j{}; j < N; ++j)
            {
                auto idx = i * result.stride_[0] + j * result.stride_[1];
                for (std::size_t k{}; k < K; ++k)
                {
                    auto a = (*this->data_)[i * this->stride_[0] + k * this->stride_[1]];
                    auto b = (*other.data_)[k * other.stride_[0] + j * other.stride_[1]];
                    (*result.data_)[idx] += a * b;
                }
            }
        }
        
        return result;
    }

    template <typename U>
    friend std::ostream& operator<<(std::ostream& out, TensorImpl<U> const& impl);
};

template <typename T>
std::ostream& operator<<(std::ostream& out, TensorImpl<T> const &impl)
{
    out << "Shape: ";
    for (auto s: impl.shape_) { std::cout << s << ", "; }
    out << '\n';

    out << "Stride: ";
    for (auto s: impl.stride_) { std::cout << s << ", "; }
    out << '\n';

    out << "Data: ";
    for (auto x: *impl.data_) { std::cout << x << ", "; }
    out << '\n';

    out << "Grad?: " << std::boolalpha << impl.requires_grad_ << '\n';

    if (impl.requires_grad_)
    {
        if (impl.grad_) { out << "Grad:\n " << *impl.grad_; }
    }

    return out;
}
