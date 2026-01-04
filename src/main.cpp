#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "tensor.h"

namespace py = pybind11;

PYBIND11_MODULE(tensor, m)
{
    m.doc() = "Tensor";

    py::class_<Tensor<float>>(m, "FloatTensor")
        .def(py::init<std::vector<std::uint32_t>>())
        .def("getData", &Tensor<float>::getData)
        .def("getShape", &Tensor<float>::getShape)
        .def("getStride", &Tensor<float>::getStride)
        .def("getInfo", &Tensor<float>::getInfo)
    ;
}
