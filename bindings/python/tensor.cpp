#include "tensor/tensor.h"

#include <nanobind/nanobind.h>
#include <nanobind/operators.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/bind_vector.h>

namespace nb = nanobind;

NB_MODULE(tensor, m)
{
  nb::class_<Tensor<float>>(m, "FloatTensor")

      .def(nb::init<std::vector<std::uint32_t>>())

      .def(nb::self + nb::self)

      .def("__matmul__", &Tensor<float>::matmul)

      .def("transpose", &Tensor<float>::transpose)

      ;
}
