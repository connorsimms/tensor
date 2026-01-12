#pragma once

#include "tensor.h"

template <typename T>
class SGD
{
public:
private:
    std::vector<Tensor<T>> params_;
    float learning_rate_;
}
