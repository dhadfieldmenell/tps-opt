#include <cuda_runtime.h>
#include <cublas_v2.h>

#define MAX_DIM 150
#define DATA_DIM 3

void gpuPrintArr(float* x, int N);
void gpuPrintArr(int* x, int N);

void initProbNM(float* x, float* y, float* xw, float* yw,
		int N, int stride, int* xdims, int* ydims,
		float outlier_prior, float r, float* corr);

void normProbNM(float* corr, int* xdims, int* ydims, int N, 
		int stride, float outlier_frac, int norm_iters);

void getTargPts(float* x, float* y, float* xw, float* yw, float* corr, 
		int* xdims, int* ydims, float cutoff, int stride, int N,
		float* xt, float* yt);
