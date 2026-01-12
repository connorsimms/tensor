#pragma once

#include "tensor.h"

template <typename T>
class SGD
{
public:
    SGD(std::vector<Tensor<T>> const& params, float const& lr)
    : params_{ params }
    , learning_rate_{ lr }
    { }

    void step()
    {
        for (auto param: params_)
        {
            auto p = param.impl();

            if (!p->grad_) continue; 

             std::transform(p->data_->begin(), p->data_->end(),
                            p->grad_->data_->begin(), p->data_->begin(),
                            [this](T const& d, T const& g) 
                            {
                                return d - (g * static_cast<T>(this->learning_rate_));
                            });
        }
    }

    void reset_grad()
    {
        for (auto param: params_)
        {
            auto p = param.impl();
            p->grad_ = nullptr;
        }
    }

private:
    std::vector<Tensor<T>> params_;
    float learning_rate_;
};
