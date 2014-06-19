#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <math.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "tps.cuh"

/**************************************************************
 ****************Device Utility Functions**********************
 **************************************************************/

__device__ int rMInd(int offset, int i, int j, int n_cols){
  /*
   * returns an index into an array with n_cols colums
   * at row i, column j (A[i, j]) stored in Row-Major Format
   */
  return offset + i * n_cols + j;
}
__device__ int rMInd(int i, int j, int n_cols){
  /*
   * returns an index into an array with n_cols colums
   * at row i, column j (A[i, j]) stored in Row-Major Format
   */
  return i * n_cols + j;
}

__device__ int cMInd(int offset, int i, int j, int n_rows){
  /*
   * returns an index into an array with n_rows rows
   * at row i, column j (A[i, j]) stored in Column-Major Format
   */
  return offset + i + n_rows * j;
}

__device__ int cMInd(int i, int j, int n_rows){
  /*
   * returns an index into an array with n_rows rows
   * at row i, column j (A[i, j]) stored in Column-Major Format
   */
  return i + n_rows * j;
}

/************************************************************
 *********************GPU Kernels****************************
 ************************************************************/
__global__ void _gpuFloatPrintArr(float* x, int N){
  int ix = blockIdx.x * blockDim.x + threadIdx.x;
  if (ix < N){
    printf("GPU Print:\t arr[%i] = %f\n", ix, x[ix]);
  }
}

__global__ void _gpuIntPrintArr(int* x, int N){
  int ix = blockIdx.x * blockDim.x + threadIdx.x;
  if (ix < N){
    printf("GPU Print:\t arr[%i] = %i\n", ix, x[ix]);
  }
}

__global__ void _fillMat(float* dest_ptr[], float* val_ptr[], int* dims){
  /*
   * Fills the matrices pointed to in dest with the colum values in vals
   * called with 1 block per item and at least dims[bix] threads
   */
  __shared__ int dim;
  __shared__ float s_val[DATA_DIM], *dest;
  int bix = blockIdx.x; int tix = threadIdx.x;
  if (tix == 0) {
    dim = dims[bix];
    dest = dest_ptr[bix];
    float* val = val_ptr[bix];
    for(int i = 0; i < DATA_DIM; ++i){
      s_val[i] = val[i];
    }
  }
  __syncthreads();
  
  if (tix < dim){
    for (int i = 0; i < DATA_DIM; ++i){
      dest[rMInd(tix, i, DATA_DIM)] = s_val[i];
    }
  }    
}

__global__ void _initProbNM(float* x_ptr[], float* y_ptr[], float* xw_ptr[], float* yw_ptr[], 
			    int* xdims, int* ydims, float p, float r, int N, 
			    float* corr_ptr_cm[], float* corr_ptr_rm[]) {
  /* Batch Initialize the correspondence matrix for use in TPS-RPM
   * Called with 1 Block per item in the batch and MAX_DIM threads
   * assumes data is padded with 0's beyond bounds
   */
  __shared__ float s_x[MAX_DIM * DATA_DIM], s_y[MAX_DIM * DATA_DIM];
  __shared__ float s_xw[MAX_DIM * DATA_DIM], s_yw[MAX_DIM * DATA_DIM];
  __shared__ int xdim, ydim, m_dim;
  __shared__ float *x, *y, *xw, *yw, *corr_rm, *corr_cm;
  int tix = threadIdx.x; int bix = blockIdx.x;
  float dist_ij, dist_ji, tmp, diff; int i_ix, j_ix, n_corr_c, n_corr_r;
  if (tix == 0) {
    xdim = xdims[bix];
    ydim = ydims[bix];

    x = x_ptr[bix];
    y = y_ptr[bix];

    xw = xw_ptr[bix];
    yw = yw_ptr[bix];

    corr_cm = corr_ptr_cm[bix];
    corr_rm = corr_ptr_rm[bix];
  }
  __syncthreads();
  n_corr_c = xdim + 1;
  n_corr_r = ydim + 1;
  if (tix < MAX_DIM){
    for (int i = 0; i < DATA_DIM; ++i){
      s_x[rMInd(tix, i, DATA_DIM)]  = x[rMInd(tix, i, DATA_DIM)];
      s_xw[rMInd(tix, i, DATA_DIM)] = xw[rMInd(tix, i, DATA_DIM)];
    }
  }
  if (tix < DATA_DIM){
    for (int i = 0; i < DATA_DIM; ++i){
      s_y[rMInd(tix, i, DATA_DIM)]  = y[rMInd(tix, i, DATA_DIM)];
      s_yw[rMInd(tix, i, DATA_DIM)] = yw[rMInd(tix, i, DATA_DIM)];
    }
  }
  //Initialize the bottom right
  if (tix == 0){
    corr_rm[rMInd(xdim, ydim, n_corr_c)] = p * sqrt((float) (xdim * ydim));
    corr_cm[cMInd(xdim, ydim, n_corr_r)] = p * sqrt((float) (xdim * ydim));
    m_dim = MAX(xdim, ydim);
  }
  __syncthreads();

  i_ix = rMInd(tix, 0, DATA_DIM);
  for( int j = 0; j < m_dim; ++j){      
    j_ix = rMInd(j, 0, DATA_DIM);

    tmp = 0;
    for (int k = 0; k < DATA_DIM; ++k){
      diff= s_xw[i_ix + k] - s_y[j_ix + k];
      tmp += diff * diff;
    }
    dist_ij = sqrt(tmp);

    tmp = 0;
    for (int k = 0; k < DATA_DIM; ++k){
      diff= s_yw[i_ix + k] - s_x[j_ix + k];
      tmp += diff * diff;
    }
    dist_ji = sqrt(tmp);

    tmp = 0;
    for (int k = 0; k < DATA_DIM; ++k){
      diff = s_x[i_ix + k] - s_yw[j_ix + k];
      tmp += diff * diff;
    }

    tmp = 0;
    for (int k = 0; k < DATA_DIM; ++k){
      diff = s_y[i_ix + k] - s_xw[j_ix + k];
      tmp += diff * diff;
    }
    dist_ji += sqrt(tmp);

    if (tix < xdim) corr_cm[cMInd(tix, j, n_corr_r)] = exp( -1 * dist_ij / (float) (2 * r));      
    if (tix < ydim) corr_rm[rMInd(j, tix, n_corr_c)] = exp( -1 * dist_ji / (float) (2 * r));      
  }
  if (tix < xdim) {
    corr_cm[cMInd(tix, ydim, n_corr_r)] = p;
    corr_rm[rMInd(tix, ydim, n_corr_c)] = p;
  }
  if (tix < ydim) {
    corr_cm[cMInd(xdim, tix, n_corr_r)] = p;
    corr_rm[rMInd(ydim, tix, n_corr_c)] = p;
  }
}


__global__ void _normProbNM(float* corr_ptr_cm[], float* corr_ptr_rm[], int* xdims, int* ydims,
			    int N, float outlierfrac, int norm_iters){
  /*  row - column normalizes prob_nm
   *  Launch with 1 block per matrix, store xdims, ydims, stride, N in constant memory
   *  Thread.idx governs which row/column to normalize
   *  Assumed to have more than 7 threads
   *  ---Might be able to run without synchronization
   *  1. Set up shared memory
   *  2. Sums rows
   *  3. Norm rows
   *  4. Sum Columns
   *  5. Norm Colums -- repeat
   */
  //set up shared variables to be read once
  __shared__ int n_corr_r, n_corr_c;
  __shared__ float *corr_rm, *corr_cm;
  __shared__ float col_coeffs[MAX_DIM], row_coeffs[MAX_DIM];
  float r_sum, c_sum; int ix_r, ix_c;
  int bix = blockIdx.x; int tix = threadIdx.x;
  if (tix == 0) { 
    n_corr_r = xdims[bix] + 1;
    n_corr_c = ydims[bix] + 1;
    corr_cm = corr_ptr_cm[bix];
    corr_rm = corr_ptr_rm[bix];
  }
  row_coeffs[tix] = 1;
  col_coeffs[tix] = 1;
  __syncthreads();

  //do normalization
  for(int ctr = 0; ctr < norm_iters; ++ctr){
    r_sum = 0;
    c_sum = 0;
    if (tix < n_corr_r){
      //sum rows and divide
      for (int i = 0; i < n_corr_c; ++i) {
  	r_sum = r_sum + corr_cm[cMInd(tix, i, n_corr_r)] * col_coeffs[i];
      }
      if (tix == n_corr_r - 1) {
  	row_coeffs[threadIdx.x] = ((n_corr_c-1) * outlierfrac) / r_sum;
      } else {
  	row_coeffs[threadIdx.x] = 1 / r_sum;
      } 
    }
    __syncthreads();
    if (tix < n_corr_c){
      //sum cols and divide
      for (int i = 0; i < n_corr_r; ++i) {
  	c_sum = c_sum + corr_rm[rMInd(i, tix, n_corr_c)] * row_coeffs[i];
      }
      if (tix == n_corr_c - 1) {
  	col_coeffs[tix] = ((n_corr_r-1) * outlierfrac) / c_sum;
      } else {
  	col_coeffs[tix] = 1 / c_sum;
      } 
    }
    __syncthreads();
  }
  //copy results back
  for(int i = 0; i < MAX_DIM; ++i){
    ix_r = rMInd(i, tix, n_corr_c);
    ix_c = cMInd(tix, i, n_corr_r);
    if (tix < n_corr_c && i < n_corr_r) {
      corr_rm[ix_r] = corr_rm[ix_r] * row_coeffs[i] * col_coeffs[tix];
    } if (tix < n_corr_r && i < n_corr_c) {
    corr_cm[ix_c] = corr_cm[ix_c] * row_coeffs[tix] * col_coeffs[i];
    }
  }
}

__global__ void  _getTargPts(float* x_ptr[], float* y_ptr[], float* xw_ptr[], float*yw_ptr[],
			     float* corr_ptr_cm[], float* corr_ptr_rm[],
			     int* xdims, int* ydims, float cutoff,
			     int N, float* xt_ptr[], float* yt_ptr[]){
  /*  Computes the target points for x and y when warped
   *  Launch with 1 block per item
   *  Thread.idx governs which row/column we are dealing with
   *  Assumed to have more than 4 threads
   *  
   *  1. set up shared memory
   *  2. Norm rows of corr, detect source outliers
   *  3. Update xt with correct value (0 pad other areas
   *  4. Norm cols of corr, detect target outliers
   *  5. Update yt with correct value (0 pad other areas
   */
  __shared__ int xdim, ydim; int n_corr_r, n_corr_c;
  __shared__ float s_y[MAX_DIM * DATA_DIM], s_x[MAX_DIM * DATA_DIM];
  __shared__ float *x, *y, *xw, *yw, *xt, *yt, *corr_rm, *corr_cm;
  int tix = threadIdx.x; int bix = blockIdx.x;
  float targ;
  if (threadIdx.x == 0){
    xdim = xdims[bix];
    ydim = ydims[bix];
    x = x_ptr[bix];
    y = y_ptr[bix];
    xw = xw_ptr[bix];
    yw = yw_ptr[bix];
    xt = xt_ptr[bix];
    yt = yt_ptr[bix];
    corr_cm = corr_ptr_cm[bix];
    corr_rm = corr_ptr_rm[bix];
  }
  __syncthreads();  
  n_corr_r = xdim + 1; n_corr_c = ydim + 1;

  if (tix < xdim){
    for(int i = 0; i < DATA_DIM; ++i){
      s_x[rMInd(tix, i, DATA_DIM)] = x[rMInd(tix, i, DATA_DIM)];
    }
  }
  if (tix < ydim){
    for(int i = 0; i < DATA_DIM; ++i){
      s_y[rMInd(tix, i, DATA_DIM)] = y[rMInd(tix, i, DATA_DIM)];
    }
  }
  __syncthreads();

  if (tix < xdim){
    float r_sum = 0; 
    for(int i = 0; i < ydim; ++i){
      r_sum = r_sum + corr_cm[cMInd(tix, i, n_corr_r)];
    }
    // if the point is an outlier map it to its current warp
    if (r_sum < cutoff){      
      for(int i = 0; i < DATA_DIM; ++i){	
    	xt[rMInd(tix, i, DATA_DIM)] = xw[rMInd(tix, i, DATA_DIM)];
      }
    } else {
      for(int i = 0; i < DATA_DIM; ++i){
    	targ = 0;
    	for(int j = 0; j < ydim; ++j){
    	  targ = targ + corr_cm[cMInd(tix, j, n_corr_r)] 
    	    * s_y[rMInd(j, i, DATA_DIM)] / r_sum;
    	}
    	xt[rMInd(tix, i, DATA_DIM)] = targ;
      }
    }
  } else if (tix < MAX_DIM){
    for(int i = 0; i < DATA_DIM; ++i){
      xt[rMInd(tix, i, DATA_DIM)] = 0;
    }
  }
  if (tix < ydim){
    float c_sum = 0; 
    for(int i = 0; i < xdim; ++i){
      c_sum = c_sum + corr_rm[rMInd(i, tix, n_corr_c)];
    }
    if (c_sum < cutoff){
      for(int i = 0; i < DATA_DIM; ++i){
  	yt[rMInd(tix, i, DATA_DIM)] = yw[rMInd(tix, i, DATA_DIM)];
      }
    } else {
      for(int i = 0; i < DATA_DIM; ++i){
  	targ = 0;
  	for(int j = 0; j < xdim; ++j){
  	  targ = targ + corr_rm[rMInd(j, tix, n_corr_c)] 
  	    * s_x[rMInd(j, i, DATA_DIM)] / c_sum;
  	}
  	yt[rMInd(tix, i, DATA_DIM)] = targ;
      }
    }
  } else if (tix < MAX_DIM){
    for(int i = 0; i < DATA_DIM; ++i){
      yt[rMInd(tix, i, DATA_DIM)] = 0;
    }
  }
}


/*****************************************************************************
 *******************************Wrappers**************************************
 *****************************************************************************/

void gpuPrintArr(float* x, int N){
  int n_threads = 10;
  int n_blocks = N / 10;
  if (n_blocks * n_threads < N){
    n_blocks += 1;
  }
  _gpuFloatPrintArr<<<n_blocks, n_threads>>>(x, N);
  cudaDeviceSynchronize();
}

void gpuPrintArr(int* x, int N){
  int n_threads = 10;
  int n_blocks = N / 10;
  if (n_blocks * n_threads < N){
    n_blocks += 1;
  }
  _gpuIntPrintArr<<<n_blocks, n_threads>>>(x, N);
  cudaDeviceSynchronize();
}

void fillMat(float* dest_ptr[], float* val_ptr[], int* dims, int N){
  int n_threads = MAX_DIM;
  int n_blocks = N;
  _fillMat<<<n_blocks, n_threads>>>(dest_ptr, val_ptr, dims);
}

void initProbNM(float* x[], float* y[], float* xw[], float* yw[],
		int N, int* xdims, int* ydims, float outlier_prior, 
		float r, float* corr_cm[], float* corr_rm[]){
  int n_threads = MAX_DIM;
  int n_blocks = N;
  // printf("Launching Initlization Kernel with %i blocks and %i threads\n", n_blocks, n_threads);
  _initProbNM<<<n_blocks, n_threads>>>(x, y, xw, yw, xdims, ydims, outlier_prior, r, N, 
				       corr_cm, corr_rm);
}

void normProbNM(float* corr_cm[], float* corr_rm[], int* xdims, int* ydims, int N, 
		float outlier_frac, int norm_iters){
  int n_blocks = N;
  int n_threads = MAX_DIM;
  // printf("Launching Normalization Kernel with %i blocks and %i threads\n", n_blocks, n_threads);
  _normProbNM<<<n_blocks, n_threads>>>(corr_cm, corr_rm, xdims, ydims, 
				       N, outlier_frac, norm_iters);
}


void getTargPts(float* x[], float* y[], float* xw[], float* yw[], 
		float* corr_cm[], float* corr_rm[], 
		int* xdims, int* ydims, float cutoff, int N,
		float* xt[], float* yt[]){
  int n_blocks = N;
  int n_threads = MAX_DIM;
  // printf("Launching Get Targ Pts Kernel with %i blocks and %i threads\n", n_blocks, n_threads);
  _getTargPts<<<n_blocks, n_threads>>>(x, y, xw, yw, corr_cm, corr_rm, xdims, ydims, 
				       cutoff, N, xt, yt);
}

void checkCudaErr(){
  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess){
    printf("Error Detected:\t%s\n", cudaGetErrorString(err));
    cudaDeviceReset();
    exit(1);
  }
  printf("No Error Detected!!\n");
}
