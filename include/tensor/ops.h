#pragma once

#include "tensor.h"

template <typename T>
Tensor<T> add(Tensor<T> const& lhs, Tensor<T> const &rhs) 
{
    Tensor<T> result(*lhs.impl() + *rhs.impl());

    if (lhs.impl()->requires_grad_ || rhs.impl()->requires_grad_)
    {
        result.impl()->requires_grad_ = true;
        result.impl()->parents_ = {lhs.impl(), rhs.impl()};

        result.impl()->backward_ = [result, lhs, rhs]() 
        {
            if (!result.impl()->grad_) return;

            if (lhs.impl()->requires_grad_)
            {
                if (lhs.impl()->grad_) 
                { 
                    *(lhs.impl()->grad_) += *(result.impl()->grad_); 
                }
                else
                {
                    lhs.impl()->grad_ = std::make_shared<TensorImpl<T>>(*(result.impl()->grad_)); 
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
                    rhs.impl()->grad_ = std::make_shared<TensorImpl<T>>(*(result.impl()->grad_)); 
                }
            }
        };
    }
    return result;
}

template <typename T>
Tensor<T> sub(Tensor<T> const &lhs, Tensor<T> const &rhs) 
{
    Tensor<T> result(*lhs.impl() - *rhs.impl());

    if (lhs.impl()->requires_grad_ || rhs.impl()->requires_grad_)
    {
        result.impl()->requires_grad_ = true;
        result.impl()->parents_ = {lhs.impl(), rhs.impl()};

        result.impl()->backward_ = [result, lhs, rhs]() 
        {
            if (!result.impl()->grad_) return;

            if (lhs.impl()->requires_grad_)
            {
                if (lhs.impl()->grad_) 
                { 
                    *(lhs.impl()->grad_) += *(result.impl()->grad_); 
                }
                else
                {
                    lhs.impl()->grad_ = std::make_shared<TensorImpl<T>>(*(result.impl()->grad_)); 
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
                    rhs.impl()->grad_ = std::make_shared<TensorImpl<T>>(-(*(result.impl()->grad_))); 
                }
            }
        };
    }
    return result;
}

template <typename T>
Tensor<T> transpose(Tensor<T> const& inp, std::size_t dimA, std::size_t dimB) 
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
                *(inp.impl()->grad_) += result.impl()->grad_->transpose(dimA,dimB);
            }
            else
            {
                inp.impl()->grad_ = std::make_shared<TensorImpl<T>>(result.impl()->grad_->transpose(dimA,dimB));
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
            if (!result.impl()->grad_) return;

            if (lhs.impl()->requires_grad_)
            {
                if (lhs.impl()->grad_) 
                { 
                    *(lhs.impl()->grad_) += result.impl()->grad_->matmul(rhs.impl()->transpose(0,1));
                }
                else
                {
                    lhs.impl()->grad_ = std::make_shared<TensorImpl<T>>(result.impl()->grad_->matmul(rhs.impl()->transpose(0, 1)));
                }
            }

            if (rhs.impl()->requires_grad_)
            {
                if (rhs.impl()->grad_)
                {
                    *(rhs.impl()->grad_) += lhs.impl()->transpose(0,1).matmul(*result.impl()->grad_);
                }
                else
                {
                    rhs.impl()->grad_ = std::make_shared<TensorImpl<T>>(lhs.impl()->transpose(0,1).matmul(*result.impl()->grad_));
                }
            }
        };
    }
    return result;
}

template <typename T>
Tensor<T> relu(Tensor<T> const& input)
{
    auto inp = input.impl();

    Tensor<T> result(inp->relu());

    if (inp->requires_grad_)
    {
        result.impl()->requires_grad_ = true;
        result.impl()->parents_ = {inp};
        result.impl()->backward_ = [=]()
        {
            if (!result.impl()->grad_) return;

            if (!inp->grad_)
            {
                inp->grad_ = std::make_shared<TensorImpl<T>>(inp->shape_);
            }

            std::transform(inp->grad_->data_->begin(), inp->grad_->data_->end(), 
                           inp->data_->begin(), result.impl()->grad_->data_->begin(), 
                           [](T const& a, T const& b)
                           {
                                return (a > 0) ? b : 0;
                           });
        };
    }

    return result;
}
