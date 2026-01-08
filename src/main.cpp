#include "tensor.h"
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(tensor, m)
{
  m.doc() = "Tensor";

  py::class_<Tensor<float>>(m, "FloatTensor")
      .def(py::init<std::vector<std::uint32_t>>())

      .def("getData", &Tensor<float>::getData,
           py::return_value_policy::reference)

      .def("getShape", &Tensor<float>::getShape,
           py::return_value_policy::reference)

      .def("getStride", &Tensor<float>::getStride,
           py::return_value_policy::reference)

      .def("fill", &Tensor<float>::fill)

      .def("clone", &Tensor<float>::clone)

      .def("__getitem__",
           [](Tensor<float> &self, const std::vector<std::uint32_t> &indices)
           { return self.at(indices); })

      .def("__setitem__",
           [](Tensor<float> &self, const std::vector<std::uint32_t> &indices,
              float v) { self.at(indices) = v; })

      .def(py::self + py::self)

      .def(py::self * py::self)

      .def("__matmul__", &Tensor<float>::matmul)

      .def("transpose_", py::overload_cast<std::size_t, std::size_t>(
                             &Tensor<float>::transpose_))

      .def_static(
          "transpose",
          py::overload_cast<const Tensor<float> &, std::size_t, std::size_t>(
              &Tensor<float>::transpose))

      .def_static("relu", &Tensor<float>::relu)

      .def_static("relu_backward", &Tensor<float>::relu)

      ;
}
