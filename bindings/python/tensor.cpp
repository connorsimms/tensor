#include "tensor/tensor.h"
#include "tensor/ops.h"

#include <nanobind/nanobind.h>
#include <nanobind/operators.h>
#include <nanobind/stl/bind_vector.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;

NB_MODULE(tensor, m)
{
  nb::class_<Tensor<float>>(m, "FloatTensor")

      .def(nb::init<std::vector<std::uint32_t>>())

      ;
}
