#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <math.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "tps.cuh"

__global__ void _gpuPrintArr(float* x, int N){
  int ix = blockIdx.x * blockDim.x + threadIdx.x;
  if (ix < N){
    printf("GPU Print:\t arr[%i] = %f\n", ix, x[ix]);
  }
}

void gpuPrintArr(float* x, int N){
  int n_threads = 10;
  int n_blocks = N / 10;
  if (n_blocks * n_threads < N){
    n_blocks += 1;
  }
  _gpuPrintArr<<<n_blocks, n_threads>>>(x, N);
}

__const__ int max_dim = 150;

TPSContext::TPSContext(float* _x, float* _y, int* _xdims, int* _ydims,
		       float* _P, float* _q, int _N, int _stride, 
		       float _outlier_prior, float _r){
  if (_stride > max_dim){
    fprintf(stderr, "stride exceeds maximum dimensions\n");
    exit(1);
  }
  if (cublasCreate(&cublasHandle) != CUBLAS_STATUS_SUCCESS){
    fprintf(stdout, "CUBLAS initialization failed!\n");
    cudaDeviceReset();
    exit(1);
  }
  x = _x; y = _y; xdims = _xdims; ydims = _ydims; P = _P; q = _q;
  N = _N; stride = _stride; outlier_prior = _outlier_prior; r = _r;
  //initialize the other pointers
  xw = xt = yw = yt = 0;
  corr = kx = ky = 0;

  //nothing is copied to the gpu yet
  xdims_gpu = ydims_gpu = 0;
  x_gpu = y_gpu = 0;
  xw_gpu = yw_gpu = xt_gpu = yt_gpu = corr_gpu = 0;
  P_gpu = q_gpu =  kx_gpu = ky_gpu = 0;
  xdims_set = ydims_set = x_set = y_set = 0;
  xw_set = yw_set = xt_set = yt_set = corr_set = 0;
  P_set = q_set = kx_set = ky_set = 0;
  gpuAllocate();
}

int TPSContext::dataInd(int n, int i, int j){
  if ( (n > N) || (i >= xdims[n]) || (j >= DATA_DIM))
    freeAndExit();
  return n * stride * DATA_DIM  + i * DATA_DIM + j;
}

int TPSContext::corrInd(int n, int i, int j){
  if ( (n > N) || (i >= xdims[n] + 1) || (j >= ydims[n] + 1))
    freeAndExit();
  return n * stride * stride + i * (ydims[n] + 1) + j;
}

int TPSContext::kInd(int n, int i, int j){
  if ( (n > N) || (i >= xdims[n]) || (j >= ydims[n]))
    freeAndExit();
  return n * stride * stride + i * ydims[n] + j;
}

void TPSContext::freeData(){
  freeDataGPU();
  freeDataHost();
}

void TPSContext::freeDataGPU(){
  if (xdims_set) cudaFree(xdims_gpu);
  if (x_set) cudaFree(x_gpu);
  if (xw_set) cudaFree(xw_gpu);
  if (xt_set) cudaFree(xt_gpu);
  if (ydims_set) cudaFree(ydims_gpu);
  if (y_set) cudaFree(y_gpu);
  if (yw_set) cudaFree(yw_gpu);
  if (yt_set) cudaFree(yt_gpu);
  if (corr_set) cudaFree(corr_gpu);
  if (P_set) cudaFree(P_gpu);
  if (q_set) cudaFree(q_gpu);
  cublasDestroy(cublasHandle);
}

void TPSContext::freeDataHost(){
  // free any temporary memory allocated
  // NOTE that xw should be freed here, but isn't for testing purposes
  // if (xdims) delete[] xdims;
  // if (x) delete[] x;
  // if (xw) delete[] xw;
  if (xt) delete[] xt;
  // if (ydims) delete[] ydims;
  // if (y) delete[] y;
  // if (yw) delete[] yw;
  if (yt) delete[] yt;
  if (corr) delete[] corr;
  if (P) delete[] P;
  if (q) delete[] q;  

}

void TPSContext::gpuAllocate(){
  //for now just allocating space for current functionality
  cudaError_t err_xdims = cudaMalloc((void **) &xdims_gpu, dimSize());
  cudaError_t err_x = cudaMalloc((void **) &x_gpu, dataSize());
  cudaError_t err_xw = cudaMalloc((void **) &xw_gpu, dataSize());
  cudaError_t err_xt = cudaMalloc((void **) &xt_gpu, dataSize());
  cudaError_t err_ydims = cudaMalloc((void **) &ydims_gpu, dimSize());
  cudaError_t err_y = cudaMalloc((void **) &y_gpu, dataSize());
  cudaError_t err_yw = cudaMalloc((void **) &yw_gpu, dataSize());
  cudaError_t err_yt = cudaMalloc((void **) &yt_gpu, dataSize());
  cudaError_t err_corr = cudaMalloc((void **) &corr_gpu, corrSize());
  if ( (err_xdims != cudaSuccess) ||
       (err_x != cudaSuccess) ||
       (err_xt != cudaSuccess) ||
       (err_xw != cudaSuccess) ||
       (err_ydims != cudaSuccess) ||
       (err_y != cudaSuccess) ||
       (err_yt != cudaSuccess) ||
       (err_yw != cudaSuccess) ||
       (err_corr != cudaSuccess)){
    fprintf(stderr, "!!!!!!!!!!!Error Allocating GPU Memory!!!!!!!!!!!\n");
    freeData();
    exit(1);
  }
  xdims_set = ydims_set = 1;
  x_set = xw_set = xt_set = y_set = yw_set = yt_set = corr_set = 1;
}

void TPSContext::sendToGPU(){
  cudaError_t err_xdims = cudaMemcpy(xdims_gpu, xdims, dimSize(), cudaMemcpyHostToDevice);
  cudaError_t err_x = cudaMemcpy(x_gpu, x, dataSize(), cudaMemcpyHostToDevice);
  cudaError_t err_xw = cudaMemcpy(xw_gpu, xw, dataSize(), cudaMemcpyHostToDevice);
  cudaError_t err_ydims = cudaMemcpy(ydims_gpu, ydims, dimSize(), cudaMemcpyHostToDevice);
  cudaError_t err_y = cudaMemcpy(y_gpu, y, dataSize(), cudaMemcpyHostToDevice);
  cudaError_t err_yw = cudaMemcpy(yw_gpu, yw, dataSize(), cudaMemcpyHostToDevice);
  if ( (err_xdims != cudaSuccess) ||
       (err_x != cudaSuccess) ||
       (err_xw != cudaSuccess) ||
       (err_ydims != cudaSuccess) ||
       (err_y != cudaSuccess) ||
       (err_yw != cudaSuccess)) {
    fprintf(stderr, "!!!!!!!!!!!Error Transferring to  GPU Memory!!!!!!!!!!!\n");
    freeData();
    exit(1);
  }    
}

void TPSContext::getCorr(float* arr) {
  cudaError_t err_corr = cudaMemcpy(arr, corr_gpu, corrSize(), cudaMemcpyDeviceToHost);
  if (err_corr != cudaSuccess){
    fprintf(stderr, "!!!!!!!!!!!Error Retreiving Corr Matrix!!!!!!!!!!!\n");
    freeData();
    exit(1);
  }
}
void TPSContext::getCorr() {
  if (!corr){
    corr = new float[N * stride * stride];
  }
  getCorr(corr);
}

void TPSContext::getXT(float* arr){
cudaError_t err_corr = cudaMemcpy(arr, xt_gpu, dataSize(), cudaMemcpyDeviceToHost);
  if (err_corr != cudaSuccess){
    fprintf(stderr, "!!!!!!!!!!!Error Retreiving Corr Matrix!!!!!!!!!!!\n");
    freeData();
    exit(1);
  }
}

void TPSContext::getYT(float* arr){
cudaError_t err_corr = cudaMemcpy(arr, yt_gpu, dataSize(), cudaMemcpyDeviceToHost);
  if (err_corr != cudaSuccess){
    fprintf(stderr, "!!!!!!!!!!!Error Retreiving Corr Matrix!!!!!!!!!!!\n");
    freeData();
    exit(1);
  }
}

__device__ float dataNormDev(int x_ix, int y_ix, float* xdata, float* ydata){
  float outval = 0;
  float diff;
  for(int i = 0; i < DATA_DIM; ++i){
    diff = xdata[x_ix + i] - ydata[y_ix + i];
    outval += diff * diff;
  }
  return sqrt(outval);
}

float dataNorm(int x_ix, int y_ix, float* xdata, float* ydata){
  float outval = 0;
  float diff;
  for(int i = 0; i < DATA_DIM; ++i){
    diff = xdata[x_ix + i] - ydata[y_ix + i];
    outval += diff * diff;
  }
  return sqrt(outval);
}

__global__ void initProbNM(float* x, float* y, float* x_warped, float* y_warped, 
			   int N, int stride, int* xdims, int* ydims, 
			   float p, float r, float* z) {

  int stride_sq = stride * stride;
  int ix = blockIdx.x * blockDim.x + threadIdx.x;
  // printf("test printing ix %i\n", ix);
  if (ix > N * stride_sq) return;
  
  //get the block index and the dimensions
  int block_ix = ix / stride_sq;

  int xdim = xdims[block_ix];
  int ydim = ydims[block_ix];
  int local_ix = ix - block_ix * stride_sq;

  if (local_ix > (xdim+1) * (ydim+1)) return; //return if locally out of bounds

  //output written to z[ix]
  int l_x_ix = local_ix / (ydim + 1); //row we want for X data
  int l_y_ix = local_ix - l_x_ix * (ydim + 1); // row we want for Y data
  // global index into data
  int g_x_ix = DATA_DIM * (l_x_ix + block_ix * stride); 
  int g_y_ix = DATA_DIM * (l_y_ix + block_ix * stride);

  if (l_x_ix < xdim && l_y_ix < ydim) {// do the standard distance conversion
    float dist = dataNormDev(g_x_ix, g_y_ix, x_warped, y);
    dist += dataNormDev(g_x_ix, g_y_ix, x, y_warped);//forward and backward
    z[ix] = exp( -1 * dist / (float) (2 * r));
    return;
  }
  //handle the corner specially
  if(l_x_ix ==  xdim && l_y_ix == ydim ){
    z[ix] = p * sqrt((float)(xdim * ydim));
    return;
  }

  if ((l_x_ix == xdim) ^ (l_y_ix == ydim)){
    //initialize outlier prob
    z[ix] = p;
    return;
  }
}

void initProbNMWrapper(TPSContext* handle, bool fetchCorr) {
  /*
    x, y, x_warped, and y_warped are 3 dimensional arrays of floats
    x and x_warped have same dimensions

    x[i, :, :] is xdims[i] by DATA_DIM , 0 < i < N 
   */
  const int n_threads = 50;
  int N = handle->N; int stride = handle->stride;
  int n_blocks = (N * stride * stride)/n_threads;
  if (n_threads * n_blocks < N * stride * stride)
    n_blocks += 1;
  // printf("calling initProbNM with %i blocks and %i threads\n", n_blocks, n_threads);
  initProbNM<<<n_blocks, n_threads>>>(handle->x_gpu, handle->y_gpu, handle->xw_gpu, handle->yw_gpu,
					handle->N, handle->stride, handle->xdims_gpu, handle->ydims_gpu,
					handle->outlier_prior, handle->r, handle->corr_gpu);

  if (fetchCorr) {handle->getCorr();}
}

float* initProbNMWrapper(float* x, float* y, float* xw, float* yw,
			 int N, int stride, int* xdims, int* ydims,
			 float outlier_prior, float r){
  if (stride > max_dim){
    fprintf(stderr, "Matrix Size Exceeds Max Dimensions\n");
    exit(1);
  }
  TPSContext* handle = new TPSContext(x, y, xdims, ydims, new float[1], 
				      new float[1], N, stride, outlier_prior, r);
  handle->gpuAllocate();
  handle->xw = xw; handle->yw = yw;
  handle->sendToGPU();
  initProbNMWrapper(handle, false);
  cudaError_t err_launch = cudaDeviceSynchronize();
  if (err_launch != cudaSuccess){
    fprintf(stderr, "!!!!!!!!!!!Error Launching Kernel!!!!!!!!!!!\n");
    delete handle;
    exit(1);
  }
  float* z = new float[N * stride * stride];
  handle->getCorr(z);
  delete handle;
  return z;
}

void initProbNMWrapper(TPSContext* handle) {
  bool fetchCorr = false;
  initProbNMWrapper(handle, fetchCorr);
}

			   
float prob_nm_val(float* x, float* y, float* x_warped, float* y_warped,
		  int i, int j, int xdim, int ydim, float p, float r){
  if(i == xdim && j == ydim)
    return p * sqrt(xdim * ydim);
  if(i == xdim ^ j == ydim)
    return p;
  float dist = dataNorm(i * DATA_DIM, j * DATA_DIM, x_warped, y);
  dist += dataNorm(i * DATA_DIM, j * DATA_DIM, x, y_warped);  
  return exp(-1 * dist * dist / (float) (2 * r));
}

__device__ int rMInd(int offset, int i, int j, int n_cols){
  /*
   * returns an index into an array with n_cols colums
   * at row i, column j (A[i, j]) stored in Row-Major Format
   */
  return offset + i * n_cols + j;
}

__device__ int cMInd(int offset, int i, int j, int n_rows){
  /*
   * returns an index into an array with n_rows rows
   * at row i, column j (A[i, j]) stored in Column-Major Format
   */
  return offset + i + n_rows * j;
}

__global__ void normProbNM(float* prob_nm, int* xdims_all, int* ydims_all,
			   int _stride, int _N, float _outlierfrac, int _norm_iters){
  /*  row - column normalizes prob_nm
   *  Launch with 1 block per matrix, store xdims, ydims, stride, N in constant memory
   *  Thread.idx governs which row/column to normalize
   *  Assumed to have more than 7 threads
   *  ---Might be able to run without synchronization
   *  1. Copies prob_nm, ydims, xdims, into shared memory
   *  2. Sums rows
   *  3. Norm rows
   *  4. Sum Columns
   *  5. Norm Colums -- repeat
   *  
   */
  //set up shared variables to be read once
  __shared__ int xdim, ydim, offset, norm_iters;
  __shared__ float outlierfrac;
  __shared__ float col_coeffs[max_dim], row_coeffs[max_dim];
  int ix; float r_sum, c_sum;
  if (threadIdx.x == 0) xdim = xdims_all[blockIdx.x] + 1;
  if (threadIdx.x == 1) ydim = ydims_all[blockIdx.x] + 1;
  if (threadIdx.x == 2) outlierfrac = _outlierfrac;
  if (threadIdx.x == 3) offset = blockIdx.x * _stride * _stride;
  if (threadIdx.x == 4) norm_iters = _norm_iters;

  __syncthreads();
  if (threadIdx.x < xdim) row_coeffs[threadIdx.x] = 1;
  if (threadIdx.x < ydim) col_coeffs[threadIdx.x] = 1;
  __syncthreads();

  //do normalization
  for(int ctr = 0; ctr < norm_iters; ++ctr){
    r_sum = 0;
    c_sum = 0;
    if (threadIdx.x < xdim){
      //sum rows and divide
      for (int i = 0; i < ydim; ++i) {
  	r_sum = r_sum + prob_nm[rMInd(offset, threadIdx.x, i, ydim)] * col_coeffs[i];
      }
      if (threadIdx.x == xdim - 1) {
  	row_coeffs[threadIdx.x] = ((ydim-1) * outlierfrac) / r_sum;
      } else {
  	row_coeffs[threadIdx.x] = 1 / r_sum;
      } 
    }
    __syncthreads();
    if (threadIdx.x < ydim){
      //sum cols and divide
      for (int i = 0; i < xdim; ++i) {
  	c_sum = c_sum + prob_nm[rMInd(offset, i, threadIdx.x, ydim)] * row_coeffs[i];
      }
      if (threadIdx.x == ydim - 1) {
  	col_coeffs[threadIdx.x] = ((xdim-1) * outlierfrac) / c_sum;
      } else {
  	col_coeffs[threadIdx.x] = 1 / c_sum;
      } 
    }
    __syncthreads();
  }
  //copy results back
  if (threadIdx.x < xdim){
    for(int i = 0; i < ydim; ++i){
      ix = rMInd(offset, threadIdx.x, i, ydim);
      prob_nm[ix] = prob_nm[ix] * row_coeffs[threadIdx.x] * col_coeffs[i];
    }
  }
}


void normProbNMWrapper(TPSContext* handle, float outlier_frac, int norm_iters){
  int n_blocks = handle->N;
  int n_threads = handle->stride+1;
  printf("Launching Normalization Kernel with %i blocks and %i threads\n", n_blocks, n_threads);
  normProbNM<<<n_blocks, n_threads>>>(handle->corr_gpu, handle->xdims_gpu, handle->ydims_gpu,
				      handle->stride, handle->N, outlier_frac, norm_iters);
}

float* initAndNormProbNMWrapper(float* x, float* y, float* xw, float* yw,
				int N, int stride, int* xdims, int* ydims,
				float outlier_prior, float r, 
				float outlier_frac, int norm_iters){
  printf("in init and norm prob nm\n");
  if (stride > max_dim){
    fprintf(stderr, "Matrix Size Exceeds Max Dimensions\n");
    exit(1);
  }
  TPSContext* handle = new TPSContext(x, y, xdims, ydims, new float[1], 
				      new float[1], N, stride, outlier_prior, r);
  handle->gpuAllocate();
  handle->xw = xw; handle->yw = yw;
  handle->sendToGPU();
  initProbNMWrapper(handle, false);
  normProbNMWrapper(handle, outlier_frac, norm_iters);
  cudaError_t err_launch = cudaDeviceSynchronize();
  if (err_launch != cudaSuccess){
    fprintf(stderr, "!!!!!!!!!!!Error Launching Kernel!!!!!!!!!!!\n");
    printf("CUDA error: %s\n", cudaGetErrorString(err_launch));
    delete handle;
    exit(1);
  }
  float* z = new float[N * stride * stride];
  handle->getCorr(z);
  delete handle;
  return z;
}

__global__ void getTargPts(float* x, float* y, float* xw, float*yw,
			   float* prob_nm, float* xt, float* yt, 
			   int* xdims, int* ydims, float cutoff,
			   int stride, int N){
    /*  row - column normalizes prob_nm
   *  Launch with 1 block per item
   *  Thread.idx governs which row/column we are dealing with
   *  Assumed to have more than 4 threads
   *  
   *  1. Copies prob_nm, ydims, xdims, into shared memory
   *  2. Norm rows, detect outliers, 
   *  3. Norm rows
   *  4. Sum Columns
   *  5. Norm Colums -- repeat
   *  NOTE: Target Values Stored in Column Major Order!!
   */
  __shared__ int xdim, ydim, nm_offset, d_offset, nm_stride;
  int tix = threadIdx.x; int bix = blockIdx.x;
  int ind; float targ;
  if (threadIdx.x == 0) xdim = xdims[bix];
  if (threadIdx.x == 1) ydim = ydims[bix];
  if (threadIdx.x == 2) nm_stride = ydims[bix] + 1;
  if (threadIdx.x == 3) nm_offset = bix * stride * stride;
  if (threadIdx.x == 4) d_offset = bix * stride * DATA_DIM;
  __syncthreads();
  if (tix < xdim){
    float r_sum = 0; 
    for(int i = 0; i < ydim; ++i){
      r_sum = r_sum + prob_nm[rMInd(nm_offset, tix, i, nm_stride)];
    }
    //if the point is an outlier map it to its current warp
    if (r_sum < cutoff){      
      for(int i = 0; i < DATA_DIM; ++i){
	xt[cMInd(d_offset, tix, i, stride)] = xw[rMInd(d_offset, tix, i, DATA_DIM)];
      }
    } else {
      for(int i = 0; i < DATA_DIM; ++i){
	targ = 0;
	for(int j = 0; j < ydim; ++j){
	  targ = targ + prob_nm[rMInd(nm_offset, tix, j, nm_stride)] 
	    * y[rMInd(d_offset, j, i, DATA_DIM)] / r_sum;
	}
	xt[cMInd(d_offset, tix, i, stride)] = targ;
      }
    }
  } else if (tix < stride){
    for(int i = 0; i < DATA_DIM; ++i){
      ind = cMInd(d_offset, tix, i, stride);
      xt[ind] = 0;
    }
  }
  if (tix < ydim){
    float c_sum = 0; 
    for(int i = 0; i < xdim; ++i){
      c_sum = c_sum + prob_nm[rMInd(nm_offset, i, tix, nm_stride)];
    }
    if (c_sum < cutoff){
      for(int i = 0; i < DATA_DIM; ++i){
	yt[cMInd(d_offset, tix, i, stride)] = yw[rMInd(d_offset, tix, i, DATA_DIM)];
      }
    } else {
      for(int i = 0; i < DATA_DIM; ++i){
	targ = 0;
	for(int j = 0; j < xdim; ++j){
	  targ = targ + prob_nm[rMInd(nm_offset, j, tix, nm_stride)] 
	    * x[rMInd(d_offset, j, i, DATA_DIM)] / c_sum;
	}
	yt[cMInd(d_offset, tix, i, stride)] = targ;
      }
    }
  } else if (tix < stride){
    for(int i = 0; i < DATA_DIM; ++i){
      ind = cMInd(d_offset, tix, i, stride);
      yt[ind] = 0;
    }
  }
}

void getTargPtsWrapper(TPSContext* handle, float cutoff){
  int n_blocks = handle->N;
  int n_threads = handle->stride+1;
  printf("Launching Get Tart Pts Kernel with %i blocks and %i threads\n", n_blocks, n_threads);
  getTargPts<<<n_blocks, n_threads>>>(handle->x_gpu, handle->y_gpu, handle->xw_gpu, handle->yw_gpu,
				      handle->corr_gpu, handle->xt_gpu, handle->yt_gpu,
				      handle->xdims_gpu, handle->ydims_gpu,
				      cutoff, handle->stride, handle->N);  
}

void getTargPtsWrapper(float* x, float* y, float* xw, float* yw,
		       int N, int stride, int* xdims, int* ydims,
		       float outlier_prior, float r, 
		       float outlier_frac, int norm_iters, float outlier_cutoff,
		       float* xt, float* yt, float* corr){
  /* computes correspondences, then uses them to find target points
   * assumes xt yt and corr are appropriately allocated by the caller
   */
  printf("in get targ pts wrapper\n");
  if (stride > max_dim){
    fprintf(stderr, "Matrix Size Exceeds Max Dimensions\n");
    exit(1);
  }
  TPSContext* handle = new TPSContext(x, y, xdims, ydims, new float[1], 
				      new float[1], N, stride, outlier_prior, r);
  handle->gpuAllocate();
  handle->xw = xw; handle->yw = yw;
  handle->sendToGPU();
  initProbNMWrapper(handle, false);
  normProbNMWrapper(handle, outlier_frac, norm_iters);
  getTargPtsWrapper(handle, outlier_cutoff);
  cudaError_t err_launch = cudaDeviceSynchronize();
  if (err_launch != cudaSuccess){
    fprintf(stderr, "!!!!!!!!!!!Error Launching Kernel!!!!!!!!!!!\n");
    printf("CUDA error: %s\n", cudaGetErrorString(err_launch));
    delete handle;
    exit(1);
  }
  handle->getXT(xt);
  handle->getYT(yt);
  handle->getCorr(corr);
  delete handle;
}

// __global__ void toColumMajor(float* in, int m, int n, int stride, float* out){
//   /*
//    * x is a 3D matrix of N x m x n, stored in row-major format, where stride
//    * denotes the offset between successive entries
//    * converted in place to be column major for the last part
//    * expected to be called with one block per component
//    * and at least m threads
//    */
  
// }

// void toColumnMajorWrapper(float* x, int m, int n, int stride, int N){
//   int n_blocks = N;
//   int n_threads = m;
//   toColumnMajor<<<n_blocks, n_threads>>>(x, m, n, stride);
// }


void print_data(float* prob_nm, float* x, float* y, float* x_warped, float* y_warped,// float* retvals,
		int xdim, int ydim, float p, float r, int offset){
  printf("z values:");
  for(int i = 0; i < xdim + 1; ++i){
    printf("\n[ ");
    for(int j = 0; j < ydim + 1; ++j){
      printf("%.2f ", prob_nm[i * (ydim + 1) + j]);
    }
    printf("]");
  }
  printf("\n correct values:");
  for(int i = 0; i < xdim + 1; ++i){
    printf("\n[ ");
    for(int j = 0; j < ydim + 1; ++j){
      printf("%.2f ", prob_nm_val(x, y, x_warped, y_warped,
				 i, j, xdim, ydim, p, r));
    }
    printf("]");
  }
  printf("\n data offset is %i", offset);
  printf("\n x values:");
  for(int i = 0; i < xdim; ++i){
    printf("\n[ ");
    for(int k = 0; k < DATA_DIM; ++k){
      printf("%.2f ", x[i * DATA_DIM + k]);
    }
    printf("]");
  }
  printf("\n y values:\n");
  for(int i = 0; i < ydim; ++i){
    printf("\n[ ");
    for(int k = 0; k < DATA_DIM; ++k){
      printf("%.2f ", y[i * DATA_DIM + k]);
    }
    printf("]");
  }
  printf("\n\n");
}

int main(void)
{
  printf("In Main\n");
  int stride = 100;
  int N = 10;
  int* xdims = new int[N];
  int* ydims = new int[N];
  for(int i = 0; i < N; ++i){
    xdims[i] = i+5;
    ydims[i] = i + 7;
  }
  
  float* x = new float[N * stride * DATA_DIM];   float* y = new float[N * stride * DATA_DIM];
  float* x_warped = new float[N * stride * DATA_DIM];   float* y_warped = new float[N * stride * DATA_DIM];
  int idx; int offset; 
  for(int i = 0; i < N; ++i){
    offset = i * stride * DATA_DIM;
    for(int j = 0; j < stride; ++j){
      idx = j * DATA_DIM + offset;
      if(j < xdims[i]){
	for(int k = 0; k < DATA_DIM; ++k){
	  x[idx + k] = (float) idx + k;
	  x_warped[idx + k] = (float) idx + k;
	}
      }
      if(j < ydims[i]){
	for(int k = 0; k < DATA_DIM; ++k){
	  y[idx + k] = (float) idx + k;
	  y_warped[idx + k] = (float) idx + k;
	}
      }
    }
  }

  float p = .2; float r = 1;
  float* z = initProbNMWrapper(x, y, x_warped, y_warped, N, stride, xdims, ydims, p, r);

  //check solution
  float zval; float actualval; bool success; int z_offset;
  float* c_x; float* c_y; float* c_x_w; float* c_y_w;
  for(int n = 0; n < N; ++n){
    z_offset = n * stride * stride;
    offset = n * stride * DATA_DIM;
    success = 1;
    c_x = &x[offset]; c_y = &y[offset]; 
    c_x_w = &x_warped[offset]; c_y_w = &y_warped[offset];
    for(int i = 0; i < xdims[n] + 1; ++i){
      for(int j = 0; j < ydims[n] + 1; ++j){
	zval = z[z_offset + i*(ydims[n] + 1) + j];
	actualval = prob_nm_val(c_x, c_y, c_x_w, c_y_w, i, j, xdims[n], ydims[n], p, r);
	success &= (abs(actualval - zval) < .00001);
      }
    }
    if(!success){
      printf("!!!!!!data for problem %i failed!!!!!!!!!!!", n);
      float* corr_block = &z[z_offset];
      print_data(corr_block, c_x, c_y, c_x_w, c_y_w, xdims[n], ydims[n], p, r, offset);
    }
  }
  if(success){
    printf("basic test succeeded!\n");
  }
  delete[] x; delete[] y; delete[] x_warped; delete[] y_warped;
  delete[] xdims; delete[] ydims;
  return 0;
}
