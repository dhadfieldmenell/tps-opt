#include "numpy_utils.hpp"
#include <boost/python.hpp>
#include <iostream>
#include "tps.cuh"

namespace py = boost::python;

void pyFloatPrintArr(py::object x, int N){
  float* p = getGPUPointer<float>(x);
  printf("x ptr is %li\n", (long int) p);
  gpuPrintArr(p, N);
}

void pyIntPrintArr(py::object x, int N){
  int* p = getGPUPointer<int>(x);
  printf("x ptr is %li\n", (long int) p);
  gpuPrintArr(p, N);
}


void pyInitProbNM(py::object x, py::object y, py::object xw, py::object yw, 
		  py::object xdims, py::object ydims, int N, int stride,
		  float outlier_prior, float r, py::object corr){
  /*
   * Initilialized correspondence matrix returned in corr
   */
  float* x_ptr   = getGPUPointer<float>(x);
  float* xw_ptr  = getGPUPointer<float>(xw);
  int* xdims_ptr = getGPUPointer<int>(xdims);

  float* y_ptr   = getGPUPointer<float>(y);
  float* yw_ptr  = getGPUPointer<float>(yw);
  int* ydims_ptr = getGPUPointer<int>(ydims);

  float* corr_ptr = getGPUPointer<float>(corr);

  initProbNM(x_ptr, y_ptr, xw_ptr, yw_ptr, N, stride, xdims_ptr, ydims_ptr, 
	     outlier_prior, r, corr_ptr);
}

void pyNormProbNM(py::object corr, py::object xdims, py::object ydims,
			int N, int stride, float outlier_frac, int norm_iters){

  float* corr_ptr = getGPUPointer<float>(corr);
  int* xdims_ptr  = getGPUPointer<int>(xdims);
  int* ydims_ptr  = getGPUPointer<int>(ydims);

  normProbNM(corr_ptr, xdims_ptr, ydims_ptr, N, stride, outlier_frac, norm_iters);
}

void pyGetTargPts(py::object x, py::object y, py::object xw, py::object yw, 
		  py::object corr, py::object xdims, py::object ydims, float cutoff,
		  int N, int stride, py::object xt, py::object yt){
  /*
   * target vectors returned in xt and yt
   */

  float* x_ptr   = getGPUPointer<float>(x);
  float* xw_ptr  = getGPUPointer<float>(xw);
  int* xdims_ptr = getGPUPointer<int>(xdims);

  float* y_ptr   = getGPUPointer<float>(y);
  float* yw_ptr  = getGPUPointer<float>(yw);
  int* ydims_ptr = getGPUPointer<int>(ydims);

  float* corr_ptr = getGPUPointer<float>(corr);

  float* xt_ptr = getGPUPointer<float>(xt);
  float* yt_ptr = getGPUPointer<float>(yt);
  
  getTargPts(x_ptr, y_ptr, xw_ptr, yw_ptr, corr_ptr, xdims_ptr, ydims_ptr, 
	     cutoff, stride, N, xt_ptr, yt_ptr);
}



BOOST_PYTHON_MODULE(fastps) {
  py::def("float_gpu_print_arr", &pyFloatPrintArr, (py::arg("x"), py::arg("N")));
  py::def("int_gpu_print_arr", &pyIntPrintArr, (py::arg("x"), py::arg("N")));

  py::def("init_prob_nm", &pyInitProbNM, (py::arg("x"), py::arg("y"), py::arg("xw"), py::arg("yw"), 
					  py::arg("xdims"), py::arg("ydims"), py::arg("N"), 
					  py::arg("stride"), py::arg("outlier_prior"), 
					  py::arg("r"), py::arg("corr")));

  py::def("norm_prob_nm", &pyNormProbNM, (py::arg("corr"), py::arg("xdims"), py::arg("ydims"), 
					  py::arg("N"), py::arg("stride"), py::arg("outlier_frac"),
					  py::arg("norm_iters")));

  py::def("get_targ_pts", &pyGetTargPts, (py::arg("x"), py::arg("y"), py::arg("xw"), py::arg("yw"), 
					  py::arg("corr"), py::arg("xdims"), py::arg("ydims"), 
					  py::arg("outlier_cutoff"), py::arg("N"), 
					  py::arg("stride"), py::arg("xt"), py::arg("yt")));
}
