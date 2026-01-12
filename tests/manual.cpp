#include "tensor/tensor.h"
#include "tensor/ops.h"
#include "tensor/loss.h"
#include "tensor/sgd.h"

int main() {
    Tensor<float> x({4, 2});

    x[0u,0u] = 0.0f; x[0u,1u] = 1.0f;
    x[1u,0u] = 0.0f; x[1u,1u] = 1.0f;
    x[2u,0u] = 1.0f; x[2u,1u] = 0.0f;
    x[3u,0u] = 1.0f; x[3u,1u] = 1.0f;

    Tensor<float> y({4, 1});
    y[0u,0u] = 0.0f; 
    y[1u,0u] = 1.0f; 
    y[2u,0u] = 1.0f;
    y[3u,0u] = 0.0f; 

    Tensor<float> w1({2, 4}); 
    (*w1.impl()->data_) = {0.1f, -0.2f, 0.3f, 0.5f, -0.5f, 0.2f, 0.1f, -0.1f};
    w1.impl()->requires_grad_ = true;

    Tensor<float> b1({4, 4});
    b1.fill(0.0f);
    b1.impl()->requires_grad_ = true;

    Tensor<float> w2({4, 1});
    w2.fill(0.0f);
    (*w2.impl()->data_) = {0.2f, -0.4f, 0.1f, -0.5f};
    w2.impl()->requires_grad_ = true;

    Tensor<float> b2({4, 1});
    b2.fill(0.0f);
    b2.impl()->requires_grad_ = true;

    SGD<float> optim({w1, b1, w2, b2}, 0.1f); // Learning Rate = 0.1
    MSELoss<float> criterion;

    for (int epoch = 0; epoch < 1000; ++epoch) {
        optim.reset_grad();

        auto h1_linear = add(matmul(x, w1), b1); //4x4
        
        Tensor<float> h1 = relu(h1_linear); // 4x4
        
        auto pred = add(matmul(h1, w2), b2); 

        auto loss = criterion(pred, y);

        loss.backward();

        optim.step();

        if (epoch % 100 == 0) {
            std::cout << "Epoch " << epoch << " | Loss: " << (*loss.impl()->data_)[0] << "\n";
        }
    }

    std::cout << "\n--- Final Predictions ---\n";
    auto final_pred = add(matmul(add(matmul(x, w1), b1), w2), b2); 
    for(size_t i=0; i<4; ++i) {
        std::cout << "Input " << i << ": " << (*final_pred.impl()->data_)[i] << " (Target: " << (*y.impl()->data_)[i] << ")\n";
    }

    return 0;
}
