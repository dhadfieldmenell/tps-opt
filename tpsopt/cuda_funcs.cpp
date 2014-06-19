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

void pyFillMat(py::object dest, py::object val, py::object dims, int N){
  float** dest_ptr = getGPUPointer<float*>(dest);
  float** val_ptr  = getGPUPointer<float*>(val);
  int* dims_ptr    = getGPUPointer<int>(dims);

  fillMat(dest_ptr, val_ptr, dims_ptr, N);
}

void pyInitProbNM(py::object x, py::object y, py::object xw, py::object yw, 
		  py::object xdims, py::object ydims, int N,
		  float outlier_prior, float r, 
		  py::object corr_cm, py::object corr_rm){
  /*
   * Initilialized correspondence matrix returned in corr
   */
  float** x_ptr  = getGPUPointer<float*>(x);
  float** xw_ptr = getGPUPointer<float*>(xw);
  int* xdims_ptr = getGPUPointer<int>(xdims);

  float** y_ptr  = getGPUPointer<float*>(y);
  float** yw_ptr = getGPUPointer<float*>(yw);
  int* ydims_ptr = getGPUPointer<int>(ydims);

  float** corr_ptr_cm = getGPUPointer<float*>(corr_cm);
  float** corr_ptr_rm = getGPUPointer<float*>(corr_rm);

  initProbNM(x_ptr, y_ptr, xw_ptr, yw_ptr, N, xdims_ptr, ydims_ptr, 
	     outlier_prior, r, corr_ptr_cm, corr_ptr_rm);
}

void pyNormProbNM(py::object corr_cm, py::object corr_rm, py::object xdims, 
		  py::object ydims, int N, float outlier_frac, int norm_iters){

  float** corr_ptr_cm = getGPUPointer<float*>(corr_cm);
  float** corr_ptr_rm = getGPUPointer<float*>(corr_rm);

  int* xdims_ptr  = getGPUPointer<int>(xdims);
  int* ydims_ptr  = getGPUPointer<int>(ydims);

  normProbNM(corr_ptr_cm, corr_ptr_rm, xdims_ptr, ydims_ptr, 
	     N, outlier_frac, norm_iters);
}

void pyGetTargPts(py::object x, py::object y, py::object xw, py::object yw, 
		  py::object corr_cm, py::object corr_rm ,
		  py::object xdims, py::object ydims, float cutoff,
		  int N, py::object xt, py::object yt){
  /*
   * target vectors returned in xt and yt
   */

  float** x_ptr  = getGPUPointer<float*>(x);
  float** xw_ptr = getGPUPointer<float*>(xw);
  int* xdims_ptr = getGPUPointer<int>(xdims);

  float** y_ptr  = getGPUPointer<float*>(y);
  float** yw_ptr = getGPUPointer<float*>(yw);
  int* ydims_ptr = getGPUPointer<int>(ydims);

  float** corr_ptr_cm = getGPUPointer<float*>(corr_cm);
  float** corr_ptr_rm = getGPUPointer<float*>(corr_rm);

  float** xt_ptr = getGPUPointer<float*>(xt);
  float** yt_ptr = getGPUPointer<float*>(yt);

  getTargPts(x_ptr, y_ptr, xw_ptr, yw_ptr, corr_ptr_cm, corr_ptr_rm, 
	     xdims_ptr, ydims_ptr, cutoff, N, xt_ptr, yt_ptr);
}

void pyCheckCudaErr(){
  checkCudaErr();
}

BOOST_PYTHON_MODULE(cuda_funcs) {
  py::def("float_gpu_print_arr", &pyFloatPrintArr, (py::arg("x"), py::arg("N")));
  py::def("int_gpu_print_arr", &pyIntPrintArr, (py::arg("x"), py::arg("N")));

  py::def("fill_mat", &pyFillMat, (py::arg("dest"), py::arg("vals"), py::arg("dims"), py::arg("N")));

  py::def("init_prob_nm", &pyInitProbNM, (py::arg("x"), py::arg("y"), py::arg("xw"), py::arg("yw"), 
					  py::arg("xdims"), py::arg("ydims"), py::arg("N"), 
					  py::arg("outlier_prior"), py::arg("r"), 
					  py::arg("corr_cm"), py::arg("corr_rm")));

  py::def("norm_prob_nm", &pyNormProbNM, (py::arg("corr_cm"), py::arg("corr_rm"),
					  py::arg("xdims"), py::arg("ydims"), 
					  py::arg("N"), py::arg("outlier_frac"), 
					  py::arg("norm_iters")));

  py::def("get_targ_pts", &pyGetTargPts, (py::arg("x"), py::arg("y"), py::arg("xw"), py::arg("yw"), 
					  py::arg("corr_cm"), py::arg("corr_rm"),
					  py::arg("xdims"), py::arg("ydims"), 
					  py::arg("outlier_cutoff"), py::arg("N"), 
					  py::arg("xt"), py::arg("yt")));
  py::def("check_cuda_err", &pyCheckCudaErr);
}
