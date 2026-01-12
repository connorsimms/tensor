#pragma once

#include "tensor_impl.h"

template <typename T> class Tensor;
template <typename T> std::ostream& operator<<(std::ostream& out, Tensor<T> const &t);

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

    std::shared_ptr<TensorImpl<T>> impl() const { return impl_; }

    void fill(T const& val) { impl_->fill(val); }

    template <std::same_as<std::uint32_t>... Args>
    T& operator[](Args... args)
    {
        return (*impl_)[args...];
    }

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

    static void topoSort(std::shared_ptr<TensorImpl<T>> v, 
                  std::vector<std::shared_ptr<TensorImpl<T>>>& visited, 
                  std::vector<std::shared_ptr<TensorImpl<T>>>& topo)
    {
        if (std::find(visited.begin(), visited.end(), v) != visited.end()) return;

        visited.push_back(v);

        for (auto &parent: v->parents_) { topoSort(parent, visited, topo); }

        topo.push_back(v);
    };


    friend std::ostream& operator<< <>(std::ostream& out, Tensor<T> const &t);

private:
    std::shared_ptr<TensorImpl<T>> impl_;
};

template <typename T>
std::ostream& operator<<(std::ostream& out, Tensor<T> const &t)
{
    out << *t.impl_;

    return out;
}
