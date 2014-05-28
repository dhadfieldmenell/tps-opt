#include "numpy_utils.hpp"
#include <boost/python.hpp>
#include <iostream>
#include "cdist.cuh"
namespace py = boost::python;

py::object sqdistGPU(py::object x) {
  x = np_mod.attr("array")(x, "float32");
  int xdim = py::extract<int>(x.attr("shape")[0]);
  float* xdata = getPointer<float>(x);
  float* zdata = sqdistWrapper(xdata, xdata, xdim, xdim);
  return toNdarray2<float>(zdata, xdim, xdim);
}

int pyMean(py::object x) {
  x = np_mod.attr("array")(x, "float32");
  int xdim = py::extract<int>(x.attr("shape")[0]);
  float* xdata = getPointer<float>(x);
  printf("dimension is %i\n", xdim);
  int mean = 0;
  for(int i = 0; i < xdim; ++i) {
    mean += xdata[i];
    printf("idx %i, mean value is %d\n", i, mean);
  }
  mean = mean/(float) xdim;
  return mean;
}

BOOST_PYTHON_MODULE(test_boost) {
    np_mod = py::import("numpy");
    py::def("cpp_mean", &pyMean, (py::arg("x")));
    py::def("cpp_sqdist", &sqdistGPU, (py::arg("x")));
}
