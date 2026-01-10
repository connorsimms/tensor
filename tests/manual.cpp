#include "tensor/tensor.h"

#include <iomanip>

int main()
{
    Tensor<float> t(2u, 3u, 4u);

    auto i = t.impl();

    std::cout << "Addr: " << i << '\n';
    
    std::cout << "Shape: ";
    for (auto s: i->shape_) { std::cout << s << ", "; }
    std::cout << '\n';

    std::cout << "Stride: ";
    for (auto s: i->stride_) { std::cout << s << ", "; }
    std::cout << '\n';

    std::cout << "Data: ";
    for (auto x: *i->data_) { std::cout << x << ", "; }
    std::cout << '\n';

    std::cout << "Grad?: " << std::boolalpha << i->requires_grad_ << '\n';

    return 0;
}
