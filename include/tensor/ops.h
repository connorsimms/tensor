#pragma once

#include "tensor.h"

// 
// Subscribing to pytorch semantics:
// "When iterating over the dimension sizes, 
// starting at the trailing dimension, the dimension 
// sizes must either be equal, one of them is 1, 
// or one of them does not exist."
//
// TODO: clean up logic - combine dne and shape == 1 cases
//
template <typename T>
std::tuple<std::vector<std::uint32_t>, std::vector<std::uint32_t>, std::vector<std::uint32_t>>
broadcast_shapes(TensorImpl<T> const& lhs, TensorImpl<T> const& rhs)
{
    std::vector<std::uint32_t> shape_out;
    std::vector<std::uint32_t> lhs_stride;
    std::vector<std::uint32_t> rhs_stride;

    std::size_t dim = std::max(lhs.shape_.size(), rhs.shape_.size());
    shape_out.resize(dim);
    lhs_stride.resize(dim);
    rhs_stride.resize(dim);

    int i = lhs.shape_.size() - 1;
    int j = rhs.shape_.size() - 1;
    int k = dim - 1;

    while (k >= 0)
    {
        // lhs dim does not exist
        if (i < 0 && j >= 0) 
        { 
            lhs_stride[k] = 0;
            rhs_stride[k] = rhs.stride_[j];
            shape_out[k--] = rhs.shape_[j--]; 
            continue;
        }

        // rhs dim does not exist
        if (i >= 0 && j < 0) 
        { 
            lhs_stride[k] = lhs.stride_[i];
            rhs_stride[k] = 0;
            shape_out[k--] = lhs.shape_[i--]; 
            continue;
        }

        if (lhs.shape_[i] == 1)
        {
            lhs_stride[k] = 0;
            shape_out[k--] = std::max(lhs.shape_[i--], rhs.shape_[j--]);
            continue;
        }

        if (rhs.shape_[j] == 1)
        {
            rhs_stride[k] = 0;
            shape_out[k--] = std::max(lhs.shape_[i--], rhs.shape_[j--]);
            continue;
        }

        if (lhs.shape_[i] == rhs.shape_[j])
        {
            lhs_stride[k] = lhs.stride_[i];
            rhs_stride[k] = rhs.stride_[j];
            shape_out[k--] = lhs.shape_[i--]; --j;
            continue;
        }
        else
        {
            throw std::invalid_argument("Broadcast not compatible");
        }
    }

    return std::tie(shape_out, lhs_stride, rhs_stride);
}

template <typename T> Tensor<T> add(Tensor<T> const &lhs, Tensor<T> const &rhs)
{
  Tensor<T> result(*lhs.impl() + *rhs.impl());

  if (lhs.impl()->requires_grad_ || rhs.impl()->requires_grad_)
  {
    result.impl()->requires_grad_ = true;
    result.impl()->parents_ = {lhs.impl(), rhs.impl()};

    result.impl()->backward_ = [result, lhs, rhs]()
    {
      if (!result.impl()->grad_)
        return;

      if (lhs.impl()->requires_grad_)
      {
        if (lhs.impl()->grad_)
        {
          *(lhs.impl()->grad_) += *(result.impl()->grad_);
        }
        else
        {
          lhs.impl()->grad_ =
              std::make_shared<TensorImpl<T>>(*(result.impl()->grad_));
        }
      }

      if (rhs.impl()->requires_grad_)
      {
        if (rhs.impl()->grad_)
        {
          *(rhs.impl()->grad_) += *(result.impl()->grad_);
        }
        else
        {
          rhs.impl()->grad_ =
              std::make_shared<TensorImpl<T>>(*(result.impl()->grad_));
        }
      }
    };
  }
  return result;
}

template <typename T> Tensor<T> sub(Tensor<T> const &lhs, Tensor<T> const &rhs)
{
  Tensor<T> result(*lhs.impl() - *rhs.impl());

  if (lhs.impl()->requires_grad_ || rhs.impl()->requires_grad_)
  {
    result.impl()->requires_grad_ = true;
    result.impl()->parents_ = {lhs.impl(), rhs.impl()};

    result.impl()->backward_ = [result, lhs, rhs]()
    {
      if (!result.impl()->grad_)
        return;

      if (lhs.impl()->requires_grad_)
      {
        if (lhs.impl()->grad_)
        {
          *(lhs.impl()->grad_) += *(result.impl()->grad_);
        }
        else
        {
          lhs.impl()->grad_ =
              std::make_shared<TensorImpl<T>>(*(result.impl()->grad_));
        }
      }

      if (rhs.impl()->requires_grad_)
      {
        if (rhs.impl()->grad_)
        {
          *(rhs.impl()->grad_) -= *(result.impl()->grad_);
        }
        else
        {
          rhs.impl()->grad_ =
              std::make_shared<TensorImpl<T>>(-(*(result.impl()->grad_)));
        }
      }
    };
  }
  return result;
}

template <typename T>
Tensor<T> transpose(Tensor<T> const &inp, std::size_t dimA, std::size_t dimB)
{
  Tensor<T> result(inp.impl()->transpose(dimA, dimB));

  if (inp.impl()->requires_grad_)
  {
    result.impl()->requires_grad_ = true;
    result.impl()->parents_ = {inp.impl()};

    result.impl()->backward_ = [result, inp, dimA, dimB]()
    {
      if (inp.impl()->grad_)
      {
        *(inp.impl()->grad_) += result.impl()->grad_->transpose(dimA, dimB);
      }
      else
      {
        inp.impl()->grad_ = std::make_shared<TensorImpl<T>>(
            result.impl()->grad_->transpose(dimA, dimB));
      }
    };
  }
  return result;
}

template <typename T>
Tensor<T> matmul(Tensor<T> const &lhs, Tensor<T> const &rhs)
{
  Tensor<T> result(lhs.impl()->matmul(*rhs.impl()));

  if (lhs.impl()->requires_grad_ || rhs.impl()->requires_grad_)
  {
    result.impl()->requires_grad_ = true;
    result.impl()->parents_ = {lhs.impl(), rhs.impl()};

    result.impl()->backward_ = [result, lhs, rhs]()
    {
      if (!result.impl()->grad_)
        return;

      if (lhs.impl()->requires_grad_)
      {
        if (lhs.impl()->grad_)
        {
          *(lhs.impl()->grad_) +=
              result.impl()->grad_->matmul(rhs.impl()->transpose(0, 1));
        }
        else
        {
          lhs.impl()->grad_ = std::make_shared<TensorImpl<T>>(
              result.impl()->grad_->matmul(rhs.impl()->transpose(0, 1)));
        }
      }

      if (rhs.impl()->requires_grad_)
      {
        if (rhs.impl()->grad_)
        {
          *(rhs.impl()->grad_) +=
              lhs.impl()->transpose(0, 1).matmul(*result.impl()->grad_);
        }
        else
        {
          rhs.impl()->grad_ = std::make_shared<TensorImpl<T>>(
              lhs.impl()->transpose(0, 1).matmul(*result.impl()->grad_));
        }
      }
    };
  }
  return result;
}

template <typename T> Tensor<T> relu(Tensor<T> const &input)
{
  auto inp = input.impl();

  Tensor<T> result(inp->relu());

  auto res = result.impl();

  if (inp->requires_grad_)
  {
    res->requires_grad_ = true;
    res->parents_ = {inp};
    res->backward_ = [=]()
    {
      if (!res->grad_)
        return;

      if (!inp->grad_)
      {
        inp->grad_ = std::make_shared<TensorImpl<T>>(inp->shape_);
      }

      for (std::size_t i{}; i < inp->grad_->data_->size(); ++i)
      {
        (*inp->grad_->data_)[i] +=
            (((*inp->data_)[i] > 0) ? (*res->grad_->data_)[i]
                                    : static_cast<T>(0));
      }
    };
  }

  return result;
}
