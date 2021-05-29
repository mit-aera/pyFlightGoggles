#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "pybind11/eigen.h"

namespace py = pybind11;

void bind_FlightGogglesClient(py::module &m);

PYBIND11_MODULE(flightgoggles_client, m) {
    m.doc() = "Python package of FlightGogglesClient";
    bind_FlightGogglesClient(m);
}