#include <cuda_runtime.h>
#include <cublas_v2.h>

#define MAX_DIM 150
#define DATA_DIM 3

#define MIN(a, b) ((a < b) ? a : b)
#define MAX(a, b) ((a > b) ? a : b)

void gpuPrintArr(float* x, int N);
void gpuPrintArr(int* x, int N);

void fillMat(float* dest_ptr[], float* val_ptr[], int* dims, int N);

void initProbNM(float* x[], float* y[], float* xw[], float* yw[],
		int N, int* xdims, int* ydims, float outlier_prior, 
		float r, float* corr_cm[], float* corr_rm[]);

void normProbNM(float* corr_cm[], float* corr_rm[], int* xdims, int* ydims, int N, 
		float outlier_frac, int norm_iters);

void getTargPts(float* x[], float* y[], float* xw[], float* yw[], 
		float* corr_cm[], float* corr_rm[], 
		int* xdims, int* ydims, float cutoff, int N,
		float* xt[], float* yt[]);

void checkCudaErr();
