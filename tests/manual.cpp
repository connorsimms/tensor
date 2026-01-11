#include "tensor/tensor.h"

int main()
{
    Tensor<float> X(2u, 2u);
    X[0u, 0u] = 1.0; X[0u, 1u] = 2.0;
    X[1u, 0u] = 3.0; X[1u, 1u] = 4.0;
    X.impl()->requires_grad_ = true;

    std::cout << std::fixed << std::setprecision(3);

    std::cout << X;

    Tensor<float> W(2u, 2u);
    W[0u, 0u] = 0.1f; W[0u, 1u] = 0.2f;
    W[1u, 0u] = 0.3f; W[1u, 1u] = 0.4f;
    W.impl()->requires_grad_ = true;

    std::cout << W;

    Tensor<float> B(2u, 2u);
    B.fill(1.0f);
    B.impl()->requires_grad_ = true;

    std::cout << B;

    auto mm = X.matmul(W);
    auto ma = mm + B;

    std::cout << ma;

    ma.backward();

    std::cout << X;
    std::cout << W;
    std::cout << B;

    return 0;
}
