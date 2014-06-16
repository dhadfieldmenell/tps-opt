#include <cuda_runtime.h>
#include <cublas_v2.h>


float* initProbNMWrapper(float* x, float* y, float* xw, float* yw,
			 int N, int stride, int* xdims, int* ydims,
			 float outlier_prior, float r);

float* initAndNormProbNMWrapper(float* x, float* y, float* xw, float* yw,
				int N, int stride, int* xdims, int* ydims,
				float outlier_prior, float r, 
				float outlier_frac, int norm_iters);

void getTargPtsWrapper(float* x, float* y, float* xw, float* yw,
			 int N, int stride, int* xdims, int* ydims,
			 float outlier_prior, float r, 
			 float outlier_frac, int norm_iters, float outlier_cutoff,
			 float* xt, float* yt, float* corr);

void gpuPrintArr(float* x, int N);


static const int DATA_DIM = 3;
/*************************************************************************************
 ***************************Memory Management and Utilities***************************
 *************************************************************************************/
struct TPSContext {
  /*
   * Struct to handle memory management on the GPU    
   */
  static const int DATA_DIM = 3;
  //Number of splines and storage spacing
  int N, stride;
  //xdims[i] is the number of items for x[i, :, :]
  int* xdims; int* ydims; int* xdims_gpu; int* ydims_gpu;
  bool xdims_set, ydims_set; 
  //the actual points, warped points, target points
  float* x; float* xw; float* xt; 
  float* y; float* yw; float* yt;  
  //clones on the gpu
  float* x_gpu; float* y_gpu; float* xw_gpu; 
  float* yw_gpu; float* xt_gpu; float* yt_gpu;
  //true if arrays are set on the gpu
  bool x_set, y_set, xw_set, yw_set, xt_set, yt_set;
  //correspondence matrices
  float* corr; float* corr_gpu; bool corr_set;
  //kernels
  float* kx; float* ky; float* kx_gpu; float* ky_gpu;
  bool kx_set, ky_set;
  //fitting a thin plate spine to xt with x
  // is theta = dot(P, xt) - q
  float* P; float* q; float* P_gpu; float* q_gpu;
  bool P_set, q_set;
  //TPS-RPM params
  float outlier_prior, r;

  //cublas context
  cublasHandle_t cublasHandle;
  
  TPSContext(float* _x, float* _y, int* _xdims, int* _ydims,
	     float* _P, float* _q, int _N, int _stride,
	     float _outlier_prior, float _r);
  
  //functions to get indices into data
  int dataInd(int n, int i, int j);
  int dataInd(int n, int i)  {return dataInd(n, i, 0);}
  float corrVal(int n, int i, int j) {return corr[corrInd(n, i, j)];}
  int corrInd(int n, int i, int j);
  int corrInd(int n)  {return corrInd(n, 0, 0);}
  int kInd(int n, int i, int j);

  //data management functions
  void gpuAllocate();
  void sendToGPU();
  void getCorr(float* arr);
  void getCorr();
  void getXT(float* arr);
  void getYT(float* arr);
  void freeData();
  void freeDataGPU();
  void freeDataHost();
  void freeAndExit()  {fprintf(stderr, "!!!!!!INVALID ACCESS!!!!!!!!"); freeData(); exit(1);}

  //desctructor
  ~TPSContext() {freeData();}
  
  //size of transferred data functions
  //size of x*, y*
  int dataSize()  {return N * stride * DATA_DIM * sizeof(float);}
  //size of corr, kx, ky
  int corrSize()   {return N * stride * stride * sizeof(float);}
  //size of xdims, ydims
  int dimSize()   {return N * sizeof(int);}
  //TODO figure out sizes for the rest of the items that we'll need totransfer  
  
};
