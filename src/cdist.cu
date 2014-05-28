#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "cdist.cuh"


__global__ void sqdistKernel(float* x, float* y, int dim, float* z) {
  int ix = blockIdx.x * blockDim.x + threadIdx.x;
  if (ix < dim * dim) {
    int x_ix = ix / dim;
    int y_ix = ix - x_ix * dim;
    float diff = x[x_ix] - y[y_ix];
    z[ix] = diff * diff;
  }
}

float* sqdistWrapper(float* x, float* y, int xdim, int ydim)
{
  int x_size = xdim * sizeof(float);
  int y_size = ydim * sizeof(float);
  int z_size = x_size * y_size / sizeof(float);

  float* x_gpu = 0;
  float* y_gpu = 0;
  float* z_gpu = 0;

  cudaError_t err_x = cudaMalloc((void **) &x_gpu, x_size);
  cudaError_t err_y = cudaMalloc((void **) &y_gpu, y_size);
  cudaError_t err_z = cudaMalloc((void **) &z_gpu, z_size);

  if ((err_x != cudaSuccess) ||
      (err_y != cudaSuccess) ||
      (err_z != cudaSuccess))
    {
      if (x_gpu) cudaFree(x_gpu);
      if (y_gpu) cudaFree(y_gpu);
      if (z_gpu) cudaFree(z_gpu);
      fprintf(stderr, "!!!! GPU memory allocation error\n");
      return 0;
    }

  err_x = cudaMemcpy(x_gpu, x, x_size, cudaMemcpyHostToDevice);
  err_y = cudaMemcpy(y_gpu, y, y_size, cudaMemcpyHostToDevice);  

  if ((err_x != cudaSuccess) ||
      (err_y != cudaSuccess))
    {
      if (x_gpu) cudaFree(x_gpu);
      if (y_gpu) cudaFree(y_gpu);
      if (z_gpu) cudaFree(z_gpu);
      fprintf(stderr, "!!!! GPU memory allocation error\n");
      return 0;
    }

  sqdistKernel<<<xdim, ydim>>>(x_gpu, y_gpu, xdim, z_gpu);
  
  float* z = new float[xdim*ydim];
  err_z = cudaMemcpy(z, z_gpu, z_size, cudaMemcpyDeviceToHost);

  if (x_gpu) cudaFree(x_gpu);
  if (y_gpu) cudaFree(y_gpu);
  if (z_gpu) cudaFree(z_gpu);

  return z;
}

int main(void)
{
  float* x = new float[10];
  for(int i = 0; i < 10; ++i)
    x[i] = (float) i;

  float* z = sqdistWrapper(x, x, 10, 10);
  for(int i = 0; i < 10; ++i){
    for(int j = 0; j < 10; ++j)
      printf("z[%i, %i] = %2f", i, j, z[i*10 + j]);
    printf("\n");
  }

  free(x);
  delete[] z;
  
  return 0;
}
