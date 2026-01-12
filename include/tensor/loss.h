#include "tensor.h"
#include "ops.h"

template <typename T>
struct MSELoss
{
    Tensor<T> operator() (Tensor<T> const& lhs, Tensor<T> const& rhs)
    {
        // (pred - targ) ^ 2 -> sum / num elements
        // grad i,j = 2 * (pred - targ) / num elem

        Tensor<T> result(1); // this is technically 1d but currently no way to make 0d

        auto pred = lhs.impl();
        auto targ = rhs.impl();
        auto loss = result.impl();
        T N = static_cast<T>(pred->data_->size());

        auto error = std::inner_product(pred->data_->begin(), pred->data_->end(),
                                        targ->data_->begin(), static_cast<T>(0),
                                        std::plus<T>(),
                                        [](T const& p, T const& t) 
                                        { return (p - t) * (p - t); }); 

        (*loss->data_)[0] = error / N;

        if (pred->requires_grad_ || targ->requires_grad_)
        {
            loss->requires_grad_ = true;
            loss->parents_ = {pred, targ};

            loss->backward_ = [pred, targ, loss, N]()
            {
                if (pred->requires_grad_)
                {
                    if (!pred->grad_)
                    {
                        pred->grad_ = std::make_shared<TensorImpl<T>>(pred->shape_);
                    }
                    auto pg = pred->grad_->data_->begin();
                    auto pd = pred->data_->begin();
                    auto td = targ->data_->begin();

                    for (; pg != pred->grad_->data_->end(); ++pg, ++pd, ++td)
                    {
                        *pg += (*pd - *td) * static_cast<T>(2) / N;
                    }
                }

                if (targ->requires_grad_)
                {
                    if (!targ->grad_)
                    {
                        targ->grad_ = std::make_shared<TensorImpl<T>>(targ->shape_);
                    }

                    auto tg = targ->grad_->data_->begin();
                    auto pd = pred->data_->begin();
                    auto td = targ->data_->begin();

                    for (; tg != targ->grad_->data_->end(); ++tg, ++pd, ++td)
                    {
                        *tg += (*td - *pd) * static_cast<T>(2) / N;
                    }
                }
            };
        }

        return result;
    }
};
