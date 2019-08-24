

#ifndef _CUDA_TV
#define _CUDA_TV

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cmath>
#include "util.h"


float sumSquaresGPU(float *d_arr, int len);

__global__ void sumSquaresKernel(float *arr, int len, float *sum);

__global__ void subtractionKernel(float * result, float *a, float *b, int len);

void TVStep(float *d_image, int nx, int ny, int nz, float scaleFactor);

__global__ void tvKernel(float * image, int nx, int ny, int nz, float ambdala, float scaleFactor, float *beebee);


#endif