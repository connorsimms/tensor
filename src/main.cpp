#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "tensor.h"

namespace py = pybind11;

PYBIND11_MODULE(tensor, m)
{
    m.doc() = "tensor test run";

    py::class_<Tensor<float>>(m, "FloatTensor")
        .def(py::init<>())
        .def(py::init<std::size_t>())
        .def("dummy", &Tensor<float>::dummy);
}
