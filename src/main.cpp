#include "tensor.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>

namespace py = pybind11;

PYBIND11_MODULE(tensor, m)
{
  m.doc() = "Tensor";

  py::class_<Tensor<float>>(m, "FloatTensor")
      .def(py::init<std::vector<std::uint32_t>>())
      .def("getData", &Tensor<float>::getData)
      .def("getShape", &Tensor<float>::getShape)
      .def("getStride", &Tensor<float>::getStride, py::return_value_policy::reference)

        .def("clone", &Tensor<float>::clone)

        .def(py::self + py::self)

      .def("__getitem__",
           [](Tensor<float> &self, const std::vector<std::uint32_t> &indices)
           { return self.at(indices); })

      .def("__setitem__",
           [](Tensor<float> &self, const std::vector<std::uint32_t> &indices,
              float v) { self.at(indices) = v; });
}
