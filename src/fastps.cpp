#include "numpy_utils.hpp"
#include <boost/python.hpp>
#include <iostream>
#include "tps.cuh"

namespace py = boost::python;

py::object pyProbNM(py::object x, py::object y, py::object xw, py::object yw, 
		    py::list xdims, py::list ydims, int N, int stride,
		    float outlier_prior, float r){
  x = np_mod.attr("array")(x, "float32");
  float* x_data = getPointer<float>(x);
  xw = np_mod.attr("array")(xw, "float32");
  float* xw_data = getPointer<float>(xw);
  int* xdim_data = getListPointer<int>(xdims);
  y = np_mod.attr("array")(y, "float32");
  float* y_data = getPointer<float>(y);
  yw = np_mod.attr("array")(yw, "float32");
  float* yw_data = getPointer<float>(yw);
  int* ydim_data = getListPointer<int>(ydims);
  float* corr = initProbNMWrapper(x_data, y_data, xw_data, yw_data, N, stride, xdim_data, ydim_data, outlier_prior, r);
  return toNdarray1<float>(corr, N * stride * stride);
}

py::object pyNormProbNM(py::object x, py::object y, py::object xw, py::object yw, 
			py::list xdims, py::list ydims, int N, int stride,
			float outlier_prior, float r, float outlier_frac, int norm_iters){
  x = np_mod.attr("array")(x, "float32");
  float* x_data = getPointer<float>(x);
  xw = np_mod.attr("array")(xw, "float32");
  float* xw_data = getPointer<float>(xw);
  int* xdim_data = getListPointer<int>(xdims);
  y = np_mod.attr("array")(y, "float32");
  float* y_data = getPointer<float>(y);
  yw = np_mod.attr("array")(yw, "float32");
  float* yw_data = getPointer<float>(yw);
  int* ydim_data = getListPointer<int>(ydims);
  printf("in pyNormProbNM\n");
  float* corr = initAndNormProbNMWrapper(x_data, y_data, xw_data, yw_data, N, stride, 
				  xdim_data, ydim_data, outlier_prior, r, outlier_frac, norm_iters);
  return toNdarray1<float>(corr, N * stride * stride);
}

py::object pyGetTargPts(py::object x, py::object y, py::object xw, py::object yw, 
			py::list xdims, py::list ydims, int N, int stride,
			float outlier_prior, float r, float outlier_frac, int norm_iters,
			float outlier_cutoff) {
  x = np_mod.attr("array")(x, "float32");
  float* x_data = getPointer<float>(x);
  xw = np_mod.attr("array")(xw, "float32");
  float* xw_data = getPointer<float>(xw);
  int* xdim_data = getListPointer<int>(xdims);
  y = np_mod.attr("array")(y, "float32");
  float* y_data = getPointer<float>(y);
  yw = np_mod.attr("array")(yw, "float32");
  float* yw_data = getPointer<float>(yw);
  int* ydim_data = getListPointer<int>(ydims);
  printf("in pyGetTargPts\n");
  float* xt_data = new float[N * stride * DATA_DIM]; 
  float* yt_data = new float[N * stride * DATA_DIM];
  float* corr_data = new float[N * stride * stride];
  getTargPtsWrapper(x_data, y_data, xw_data, yw_data, N, stride,
		    xdim_data, ydim_data, outlier_prior, r, outlier_frac, 
		    norm_iters, outlier_cutoff, xt_data, yt_data, corr_data);
  py::object py_xt = toNdarray1<float>(xt_data, N * stride * DATA_DIM); 
  py::object py_yt = toNdarray1<float>(yt_data, N * stride * DATA_DIM); 
  py::object py_corr = toNdarray1<float>(corr_data, N * stride * stride); 
  return py::make_tuple(py_xt, py_yt, py_corr);
}

void pyPrintArr(py::object x, int N){
  long int p = py::extract<long int>(x.attr("ptr"));
  printf("x ptr is %li\n", p);
  gpuPrintArr( (float*) p, N);
}


BOOST_PYTHON_MODULE(fastps) {
  np_mod = py::import("numpy");
  py::def("prob_nm", &pyProbNM, (py::arg("x"), py::arg("y"), py::arg("xw"), py::arg("yw"), 
				 py::arg("xdims"), py::arg("ydims"), py::arg("N"), py::arg("stride"),
				 py::arg("outlier_prior"), py::arg("r")));
  py::def("prob_norm_nm", &pyNormProbNM, (py::arg("x"), py::arg("y"), py::arg("xw"), py::arg("yw"), 
				      py::arg("xdims"), py::arg("ydims"), py::arg("N"), 
				      py::arg("stride"), py::arg("outlier_prior"), py::arg("r"),
				      py::arg("outlier_frac"), py::arg("norm_iters")));
  py::def("get_targ_pts", &pyGetTargPts, (py::arg("x"), py::arg("y"), py::arg("xw"), py::arg("yw"), 
					  py::arg("xdims"), py::arg("ydims"), py::arg("N"), 
					  py::arg("stride"), py::arg("outlier_prior"), py::arg("r"),
					  py::arg("outlier_frac"), py::arg("norm_iters"),
					  py::arg("outlier_cutoff")));
  py::def("gpu_print_arr", &pyPrintArr, (py::arg("x"), py::arg("N")));
}
