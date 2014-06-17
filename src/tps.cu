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

__device__ float dataNormDev(int x_ix, int y_ix, float* xdata, float* ydata){
  float outval = 0;
  float diff;
  for(int i = 0; i < DATA_DIM; ++i){
    diff = xdata[x_ix + i] - ydata[y_ix + i];
    outval += diff * diff;
  }
  return sqrt(outval);
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
    printf("GPU Print:\t arr[%i] = %i\n", ix, x[2*ix]);
  }
}

__global__ void _initProbNM(float* x, float* y, float* x_warped, float* y_warped, 
			   int N, int stride, int* xdims, int* ydims, 
			   float p, float r, float* z) {

  int stride_sq = stride * stride;
  int ix = blockIdx.x * blockDim.x + threadIdx.x;
  // printf("test printing ix %i\n", ix);
  if (ix > N * stride_sq) return;
  
  //get the block index and the dimensions
  int block_ix = ix / stride_sq;
  int xdim = xdims[2*block_ix];
  int ydim = ydims[2*block_ix];
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

__global__ void _normProbNM(float* prob_nm, int* xdims_all, int* ydims_all,
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
  __shared__ float col_coeffs[MAX_DIM], row_coeffs[MAX_DIM];
  int ix; float r_sum, c_sum;
  if (threadIdx.x == 0) xdim = xdims_all[2*blockIdx.x] + 1;
  if (threadIdx.x == 1) ydim = ydims_all[2*blockIdx.x] + 1;
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

__global__ void _getTargPts(float* x, float* y, float* xw, float*yw,
			    float* prob_nm, int* xdims, int* ydims, float cutoff,
			    int stride, int N, float* xt, float* yt){
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
  if (threadIdx.x == 0) xdim = xdims[2*bix];
  if (threadIdx.x == 1) ydim = ydims[2*bix];
  if (threadIdx.x == 2) nm_stride = ydims[2*bix] + 1;
  if (threadIdx.x == 3) nm_offset = bix * stride * stride;
  if (threadIdx.x == 4) d_offset = bix * stride * DATA_DIM;
  __syncthreads();
  if (tix < xdim){
    float r_sum = 0; 
    for(int i = 0; i < ydim; ++i){
      r_sum = r_sum + prob_nm[rMInd(nm_offset, tix, i, nm_stride)];
    }
    // if the point is an outlier map it to its current warp
    if (r_sum < cutoff){      
      // printf("Block %i Row %i is an outlier\n", bix, tix);
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
}

void gpuPrintArr(int* x, int N){
  int n_threads = 10;
  int n_blocks = N / 10;
  if (n_blocks * n_threads < N){
    n_blocks += 1;
  }
  _gpuIntPrintArr<<<n_blocks, n_threads>>>(x, N);
}


void initProbNM(float* x, float* y, float* xw, float* yw,
		int N, int stride, int* xdims, int* ydims,
		float outlier_prior, float r, float* corr){
  if (stride > MAX_DIM){
    fprintf(stderr, "Matrix Size Exceeds Max Dimensions\n");
    exit(1);
  }
  const int n_threads = 50;
  int n_blocks = (N * stride * stride)/n_threads;
  if (n_threads * n_blocks < N * stride * stride)
    n_blocks += 1;
  printf("calling initProbNM with %i blocks and %i threads\n", n_blocks, n_threads);
  _initProbNM<<<n_blocks, n_threads>>>(x, y, xw, yw,N, stride, xdims, ydims,
					outlier_prior, r, corr);
}

void normProbNM(float* corr, int* xdims, int* ydims, int N, 
		int stride, float outlier_frac, int norm_iters){
  if (stride > MAX_DIM){
    fprintf(stderr, "Matrix Size Exceeds Max Dimensions\n");
    exit(1);
  }
  int n_blocks = N;
  int n_threads = stride;
  printf("Launching Normalization Kernel with %i blocks and %i threads\n", n_blocks, n_threads);
  _normProbNM<<<n_blocks, n_threads>>>(corr, xdims, ydims, stride, N, outlier_frac, norm_iters);
}


void getTargPts(float* x, float* y, float* xw, float* yw, float* corr, 
		int* xdims, int* ydims, float cutoff, int stride, int N,
		float* xt, float* yt){
  if (stride > MAX_DIM){
    fprintf(stderr, "Matrix Size Exceeds Max Dimensions\n");
    exit(1);
  }
  int n_blocks = N;
  int n_threads = stride;
  printf("Launching Get Targ Pts Kernel with %i blocks and %i threads\n", n_blocks, n_threads);
  _getTargPts<<<n_blocks, n_threads>>>(x, y, xw, yw, corr, xdims, ydims, 
				       cutoff, stride, N, xt, yt);
  // if ( cudaSuccess != cudaGetLastError() )
  //   printf( "Error!\n" );
  // int xt_size = N * stride * DATA_DIM;
  // float* xt_cpu = new float[xt_size];
  // cudaError_t err = cudaMemcpy(xt_cpu, xt, xt_size * sizeof(float), cudaMemcpyDeviceToHost);
  // if (err != cudaSuccess){
  //   printf("Error Detected:\t%s\n", cudaGetErrorString(err));
  //   cudaDeviceReset();
  //   exit(1);
  // }
}
