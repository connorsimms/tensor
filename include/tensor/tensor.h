#pragma once

#include "tensor_impl.h"

template <typename T>
class Tensor
{
public:
    template <std::same_as<std::uint32_t>...  Args>
    Tensor(Args... args) 
    : impl_{ std::make_shared<TensorImpl<T>>(args...) } 
    { }

    Tensor(std::vector<std::uint32_t> const &shape) 
    : impl_{ std::make_shared<TensorImpl<T>>(shape) } { }

    Tensor(Tensor const &other) : impl_{ other.impl_ } { }

    Tensor(TensorImpl<T> impl) 
    : impl_{ std::make_shared<TensorImpl<T>>(impl) } { }

    Tensor operator=(Tensor const &other) 
    { 
        impl_ = other.impl_; 
        return *this;
    }

    std::shared_ptr<TensorImpl<T>> impl() { return impl_; }

    void fill(T const& val) { impl_->fill(val); }

    template <std::same_as<std::uint32_t>... Args>
    T& operator[](Args... args)
    {
        return (*impl_)[args...];
    }

    Tensor operator+(Tensor const &other) const
    {
        Tensor result(*this->impl_ + *other.impl_);

        if (this->impl_->requires_grad_ || other.impl_->requires_grad_)
        {
            result.impl_->requires_grad_ = true;
            result.impl_->parents_ = {this->impl_, other.impl_};

            result.impl_->backward_ = [result, *this, other]() 
            {
                if (!result.impl_->grad_) return;


                if (this->impl_->requires_grad_)
                {
                    if (this->impl_->grad_) 
                    { 
                        *(this->impl_->grad_) += *(result.impl_->grad_); 
                    }
                    else
                    {
                        this->impl_->grad_ = std::make_shared<TensorImpl<T>>(*(result.impl_->grad_)); 
                    }
                }

                if (other.impl_->requires_grad_)
                {
                    if (other.impl_->grad_)
                    {
                        *(other.impl_->grad_) += *(result.impl_->grad_); 
                    }
                    else
                    {
                        other.impl_->grad_ = std::make_shared<TensorImpl<T>>(*(result.impl_->grad_)); 
                    }
                }
            };
        }
        return result;
    }

    Tensor operator-(Tensor const &other) const
    {
        Tensor result(*this->impl_ - *other.impl_);

        if (this->impl_->requires_grad_ || other.impl_->requires_grad_)
        {
            result.impl_->requires_grad_ = true;
            result.impl_->parents_ = {this->impl_, other.impl_};

            result.impl_->backward_ = [result, *this, other]() 
            {
                if (!result.impl_->grad_) return;


                if (this->impl_->requires_grad_)
                {
                    if (this->impl_->grad_) 
                    { 
                        *(this->impl_->grad_) += *(result.impl_->grad_); 
                    }
                    else
                    {
                        this->impl_->grad_ = std::make_shared<TensorImpl<T>>(*(result.impl_->grad_)); 
                    }
                }

                if (other.impl_->requires_grad_)
                {
                    if (other.impl_->grad_)
                    {
                        *(other.impl_->grad_) -= *(result.impl_->grad_); 
                    }
                    else
                    {
                        other.impl_->grad_ = std::make_shared<TensorImpl<T>>(-(*(result.impl_->grad_))); 
                    }
                }
            };
        }
        return result;
    }

    Tensor transpose(std::size_t dimA, std::size_t dimB)
    {
        Tensor result(this->impl_->transpose(dimA, dimB)); 

        if (this->impl_->requires_grad_)
        {
            result.impl_->requires_grad_ = true;
            result.impl_->parents_ = {this->impl_};

            result.impl_->backward_ = [result, *this, dimA, dimB]() 
            {
                if (this->impl_->grad_)
                {
                    *(this->impl_->grad_) += result.impl_->grad_->transpose(dimA,dimB);
                }
                else
                {
                    this->impl_->grad_ = std::make_shared<TensorImpl<T>>(result.impl_->grad_->transpose(dimA,dimB));
                }
            };
        }
        return result;
    }

    Tensor matmul(Tensor const &other) const
    {
        Tensor result(this->impl_->matmul(*other.impl_)); 

        if (this->impl_->requires_grad_ || other.impl_->requires_grad_)
        {
            result.impl_->requires_grad_ = true;
            result.impl_->parents_ = {this->impl_, other.impl_};

            result.impl_->backward_ = [result, *this, other]() 
            {
                if (!result.impl_->grad_) return;


                if (this->impl_->requires_grad_)
                {
                    if (this->impl_->grad_) 
                    { 
                        *(this->impl_->grad_) += result.impl_->grad_->matmul(other.impl_->transpose(0,1));
                    }
                    else
                    {
                        this->impl_->grad_ = std::make_shared<TensorImpl<T>>(result.impl_->grad_->matmul(other.impl_->transpose(0, 1)));
                    }
                }

                if (other.impl_->requires_grad_)
                {
                    if (other.impl_->grad_)
                    {
                        *(other.impl_->grad_) += this->impl_->transpose(0,1).matmul(*result.impl_->grad_);
                    }
                    else
                    {
                        other.impl_->grad_ = std::make_shared<TensorImpl<T>>(this->impl_->transpose(0,1).matmul(*result.impl_->grad_));
                    }
                }
            };
        }
        return result;
    }

    static void topoSort(std::shared_ptr<TensorImpl<T>> v, 
                  std::vector<std::shared_ptr<TensorImpl<T>>>& visited, 
                  std::vector<std::shared_ptr<TensorImpl<T>>>& topo)
    {
        if (std::find(visited.begin(), visited.end(), v) != visited.end()) return;

        visited.push_back(v);

        for (auto &parent: v->parents_) { topoSort(parent, visited, topo); }

        topo.push_back(v);
    };

    void backward() 
    { 
        if (!impl_->grad_)
        {
            impl_->grad_ = std::make_shared<TensorImpl<T>>(impl_->shape_);
            impl_->grad_->fill(static_cast<T>(1));
        }

        std::vector<std::shared_ptr<TensorImpl<T>>> visited;
        std::vector<std::shared_ptr<TensorImpl<T>>> topo;

        topoSort(impl_, visited, topo);

        for (auto it = topo.rbegin(); it != topo.rend(); ++it)
        {
            if((*it)->backward_) (*it)->backward_(); 
        }
    }

    template <typename U>
    friend std::ostream& operator<<(std::ostream& out, Tensor<U> const &t);

private:
    std::shared_ptr<TensorImpl<T>> impl_;
};

template <typename T>
std::ostream& operator<<(std::ostream& out, Tensor<T> const &t)
{
    out << *t.impl_;

    return out;
}
