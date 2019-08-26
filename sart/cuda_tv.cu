

#include "cuda_tv.cuh"


float sumSquaresGPU(float *d_arr, int len)
{
	dim3 block(64);
	dim3 grid(iDivUp(len, block.x));

	float h_sum;
	float *d_sum;
	HANDLE_ERROR(cudaMalloc((void **)&d_sum, sizeof(float)));
	HANDLE_ERROR(cudaMemset(d_sum, 0, sizeof(float)));

	sumSquaresKernel <<<grid, block>>> (d_arr, len, d_sum);

	HANDLE_ERROR(cudaMemcpy(&h_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaFree(d_sum));

	return h_sum;
}

__global__ void sumSquaresKernel(float *arr, int len, float *sum)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	if (idx >= len)
		return;

	__shared__ float s_sum;

	if (threadIdx.x == 0)
		s_sum = 0;

	__syncthreads();

	float val = arr[idx];
	atomicAdd(&s_sum, val * val);

	__syncthreads();

	if (threadIdx.x == 0)
		atomicAdd(sum, s_sum);
}

__global__ void subtractionKernel(float * result, float *a, float *b, int len)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	if (idx >= len)
		return;

	result[idx] = a[idx] - b[idx];
}

void TVStep(float *d_image, int nx, int ny, int nz, float scaleFactor)
{
	dim3 block3d(4, 4, 4);
	dim3 grid3d(iDivUp(nx, block3d.x), iDivUp(ny, block3d.y), iDivUp(ny, block3d.z));
	float *d_beebee;
	HANDLE_ERROR(cudaMalloc((void **)&d_beebee, sizeof(float)));
	HANDLE_ERROR(cudaMemset(d_beebee, 0, sizeof(float)));

	tvKernel <<<grid3d, block3d>>> (d_image, nx, ny, nz, 0.0f, scaleFactor, d_beebee);

	HANDLE_ERROR(cudaFree(d_beebee));
}

__global__ void tvKernel(float * image, int nx, int ny, int nz, float ambdala, float scaleFactor, float *beebee)
{
	int tidx = blockDim.x * blockIdx.x + threadIdx.x;
	int tidy = blockDim.y * blockIdx.y + threadIdx.y;
	int tidz = blockDim.z * blockIdx.z + threadIdx.z;
	int gidx = nx * ny * tidz + nx * tidy + tidx;

	if (tidx >= nx || tidy >= ny || tidz >= nz)
		return;

	__shared__ float s_beebee;

	if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0)
		s_beebee = 0;

	__syncthreads();

	float tvgmat = 0;
	float eps = 0.000000001f;
	float w[27] = { 0 };
	float t[5] = { 0 };

	if (tidx > 0 && tidx < nx - 1 && tidy > 0 && tidy < ny - 1 && tidz > 0 && tidz < nz - 1)
	{
		//w[0] = image[nx * ny * (tidz - 1) + nx * (tidy - 1) + tidx - 1];
		//w[1] = image[nx * ny * (tidz - 1) + nx * (tidy - 1) + tidx];
		//w[2] = image[nx * ny * (tidz - 1) + nx * (tidy - 1) + tidx + 1];

		//w[3] = image[nx * ny * (tidz - 1) + nx * tidy + tidx - 1];
		w[4] = image[nx * ny * (tidz - 1) + nx * tidy + tidx];
		w[5] = image[nx * ny * (tidz - 1) + nx * tidy + tidx + 1];

		//w[6] = image[nx * ny * (tidz - 1) + nx * (tidy + 1) + tidx - 1];
		w[7] = image[nx * ny * (tidz - 1) + nx * (tidy + 1) + tidx];
		//w[8] = image[nx * ny * (tidz - 1) + nx * (tidy + 1) + tidx + 1];

		//w[9] = image[nx * ny * tidz + nx * (tidy - 1) + tidx - 1];
		w[10] = image[nx * ny * tidz + nx * (tidy - 1) + tidx];
		w[11] = image[nx * ny * tidz + nx * (tidy - 1) + tidx + 1];

		w[12] = image[nx * ny * tidz + nx * tidy + tidx - 1];
		w[13] = image[nx * ny * tidz + nx * tidy + tidx];
		w[14] = image[nx * ny * tidz + nx * tidy + tidx + 1];

		w[15] = image[nx * ny * tidz + nx * (tidy + 1) + tidx - 1];
		w[16] = image[nx * ny * tidz + nx * (tidy + 1) + tidx];
		//w[17] = image[nx * ny * tidz + nx * (tidy + 1) + tidx + 1];

		//w[18] = image[nx * ny * (tidz + 1) + nx * (tidy - 1) + tidx - 1];
		w[19] = image[nx * ny * (tidz + 1) + nx * (tidy - 1) + tidx];
		//w[20] = image[nx * ny * (tidz + 1) + nx * (tidy - 1) + tidx + 1];

		w[21] = image[nx * ny * (tidz + 1) + nx * tidy + tidx - 1];
		w[22] = image[nx * ny * (tidz + 1) + nx * tidy + tidx];
		//w[23] = image[nx * ny * (tidz + 1) + nx * tidy + tidx + 1];

		//w[24] = image[nx * ny * (tidz + 1) + nx * (tidy + 1) + tidx - 1];
		//w[25] = image[nx * ny * (tidz + 1) + nx * (tidy + 1) + tidx];
		//w[26] = image[nx * ny * (tidz + 1) + nx * (tidy + 1) + tidx + 1];

		t[0] = (w[13] - w[22]) / sqrtf((w[13] - w[22]) * (w[13] - w[22]) + (w[19] - w[22]) * (w[19] - w[22]) + (w[21] - w[22]) * (w[21] - w[22]) + eps);
		t[1] = (-w[10] - w[12] + 3 * w[13] - w[4]) / sqrtf((w[10] - w[13]) * (w[10] - w[13]) + (w[12] - w[13]) * (w[12] - w[13]) + (-w[13] + w[4]) * (-w[13] + w[4]) + eps);
		t[2] = (w[13] - w[14]) / sqrtf((w[11] - w[14]) * (w[11] - w[14]) + (w[13] - w[14]) * (w[13] - w[14]) + (-w[14] + w[5]) * (-w[14] + w[5]) + eps);
		t[3] = (w[13] - w[16]) / sqrtf((w[13] - w[16]) * (w[13] - w[16]) + (w[15] - w[16]) * (w[15] - w[16]) + (-w[16] + w[7]) * (-w[16] + w[7]) + eps);
		t[4] = ambdala * w[13] / (eps + fabsf(w[13]));

		tvgmat = t[0] + t[1] + t[2] + t[3] + t[4];
	}
	
	atomicAdd(&s_beebee, tvgmat * tvgmat);

	__syncthreads();

	if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0)
		atomicAdd(beebee, s_beebee);

	__syncthreads();

	float bobo = sqrtf(*beebee);

	if (fabsf(bobo) < 0.001f)
		bobo = 1.0f;

	tvgmat = scaleFactor * tvgmat / bobo;
	image[gidx] -= tvgmat;
}