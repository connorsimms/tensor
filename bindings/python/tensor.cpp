#include "tensor/tensor.h"

#include <nanobind/nanobind.h>
#include <nanobind/operators.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/bind_vector.h>

namespace nb = nanobind;

NB_MODULE(tensor, m)
{
  m.doc() = "Tensor";

  nb::class_<Tensor<float>>(m, "FloatTensor")
      .def(nb::init<std::vector<std::uint32_t>>())

      .def("getData", &Tensor<float>::getData,
           nb::rv_policy::automatic)

      .def("getShape", &Tensor<float>::getShape,
           nb::rv_policy::automatic)

      .def("getStride", &Tensor<float>::getStride,
           nb::rv_policy::automatic)

      .def("__getitem__",
           [](Tensor<float> &self, const std::vector<std::uint32_t> &indices)
           { return self.at(indices); })

      .def("__setitem__",
           [](Tensor<float> &self, const std::vector<std::uint32_t> &indices,
              float v) { self.at(indices) = v; })

      .def("fill", &Tensor<float>::fill)

      .def("clone", &Tensor<float>::clone)

      .def(nb::self + nb::self, nb::rv_policy::automatic)

      .def(nb::self += nb::self)

      .def(nb::self * nb::self, nb::rv_policy::automatic)

      .def("__matmul__", &Tensor<float>::matmul, nb::rv_policy::automatic)

      .def("transpose_", nb::overload_cast<std::size_t, std::size_t>(
                             &Tensor<float>::transpose_))

      .def_static(
          "transpose",
          nb::overload_cast<const Tensor<float> &, std::size_t, std::size_t>(
              &Tensor<float>::transpose))

      .def_static("relu", &Tensor<float>::relu)

      .def_static("relu_backward", &Tensor<float>::relu_backward)

      ;
}
