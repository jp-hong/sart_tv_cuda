

#include "cuda_sart.cuh"


texture <float, cudaTextureType3D, cudaReadModeElementType> tex;

cudaArray *d_volumeArray;

void setTextureFilterMode(bool bLinearFilter)
{
	tex.filterMode = bLinearFilter ? cudaFilterModeLinear : cudaFilterModePoint;
}

void createCudaArray(cudaExtent volumeSize)
{
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
	HANDLE_ERROR(cudaMalloc3DArray(&d_volumeArray, &channelDesc, volumeSize));
}

void bind3DTexture(const float *d_volume, cudaExtent volumeSize)
{
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();

	// copy data to 3D array
	cudaMemcpy3DParms copyParams = { 0 };
	copyParams.srcPtr = make_cudaPitchedPtr((void *)d_volume, volumeSize.width*sizeof(float), volumeSize.width, volumeSize.height);
	copyParams.dstArray = d_volumeArray;
	copyParams.extent = volumeSize;
	copyParams.kind = cudaMemcpyDeviceToDevice;
	HANDLE_ERROR(cudaMemcpy3D(&copyParams));

	// set texture parameters
	tex.normalized = true;                      // access with normalized texture coordinates
	tex.filterMode = cudaFilterModeLinear;      // linear interpolation
	tex.addressMode[0] = cudaAddressModeBorder;   // clamp texture coordinates to border
	tex.addressMode[1] = cudaAddressModeBorder;
	tex.addressMode[2] = cudaAddressModeBorder;

	// bind array to 3D texture
	HANDLE_ERROR(cudaBindTextureToArray(tex, d_volumeArray, channelDesc));
}

void free3DTexture()
{
	HANDLE_ERROR(cudaUnbindTexture(tex));
	HANDLE_ERROR(cudaFreeArray(d_volumeArray));
}

__global__ void rotationKernel(float *img, int nx, int ny, int nz, float x0, float y0, float z0, float dx, float dy, float dz, float xlen, float ylen, float zlen, float delta_angle)
{
	int tidx = blockDim.x * blockIdx.x + threadIdx.x;
	int tidy = blockDim.y * blockIdx.y + threadIdx.y;
	int tidz = blockDim.z * blockIdx.z + threadIdx.z;
	int gidx = nx * ny * tidz + nx * tidy + tidx;

	if (tidx >= nx || tidy >= ny || tidz >= nz)
		return;

	float sinval = sinf(delta_angle);
	float cosval = cosf(delta_angle);

	float x = x0 + dx * (tidx + 0.5);
	float y = y0 + dy * (tidy + 0.5);
	float z = z0 + dz * (tidz + 0.5);

	float prev_x = x * cosval - y * sinval;
	float prev_y = x * sinval + y * cosval;

	float norm_x = prev_x / xlen + 0.5f;
	float norm_y = prev_y / ylen + 0.5f;
	float norm_z = z / zlen + 0.5f;

	img[gidx] = tex3D(tex, norm_x, norm_y, norm_z);
}

__global__ void forwardProjectionTexKernel(float *worksino, float sinval, float cosval, float r, float s, int nu, int nv, float du, float dv, float u0, float v0,
	float dx, float dy, float dz, float x0, float y0, float z0, int nx, int ny, int nz, float xlen, float ylen, float zlen)
{
	int tidu = blockDim.x * blockIdx.x + threadIdx.x;
	int tidv = blockDim.y * blockIdx.y + threadIdx.y;
	int gidx = (nv - 1 - tidv) * nu + tidu;

	if (tidu >= nu || tidv >= nv)
		return;

	float xSource = r * cosval;
	float ySource = r * sinval;
	float zSource = 0;

	float xDetCenter = (r - s) * cosval;
	float yDetCenter = (r - s) * sinval;
	float zDetCenter = 0;

	float eux = -sinval;
	float euy = cosval;
	float euz = 0;

	float evx = 0;
	float evy = 0;
	float evz = 1;

	float u = u0 + (tidu + 0.5f) * du;
	float v = v0 + (tidv + 0.5f) * dv;

	float xbin = xDetCenter + eux * u + evx * v;
	float ybin = yDetCenter + euy * u + evy * v;
	float zbin = zDetCenter + euz * u + evz * v;

	float xdiff = xbin - xSource;
	float ydiff = ybin - ySource;
	float zdiff = zbin - zSource;

	float xad = fabsf(xdiff) / dx;
	float yad = fabsf(ydiff) / dy;
	float zad = fabsf(zdiff) / dz;

	float total = 0;

	float x, y, z, normx, normy, normz, travVoxlen;

	if (xad > yad && xad > zad)
	{
		float yox = ydiff / xdiff;
		float zox = zdiff / xdiff;
		travVoxlen = dx * sqrtf(1 + yox * yox + zox * zox);

		for (int ix = 0; ix < nx; ix++)
		{
			x = x0 + dx * (ix + 0.5);
			y = ySource + yox * (x - xSource);
			z = zSource + zox * (x - xSource);

			if (y >= y0 && y <= -y0 && z >= z0 && z <= -z0)
			{
				normx = x / xlen + 0.5f;
				normy = y / ylen + 0.5f;
				normz = z / zlen + 0.5f;

				total += tex3D(tex, normx, normy, normz);
			}
		}
	}
	else if (yad > zad)
	{
		float xoy = xdiff / ydiff;
		float zoy = zdiff / ydiff;
		travVoxlen = dy * sqrtf(1 + xoy * xoy + zoy * zoy);

		for (int iy = 0; iy < ny; iy++)
		{
			y = y0 + dy * (iy + 0.5f);
			x = xSource + xoy * (y - ySource);
			z = zSource + zoy * (y - ySource);

			if (x >= x0 && x <= -x0 && z >= z0 && z <= -z0)
			{
				normx = x / xlen + 0.5f;
				normy = y / ylen + 0.5f;
				normz = z / zlen + 0.5f;

				total += tex3D(tex, normx, normy, normz);
			}
		}
	}
	else
	{
		float xoz = xdiff / zdiff;
		float yoz = ydiff / zdiff;
		travVoxlen = dz * sqrtf(1 + xoz * xoz + yoz * yoz);

		for (int iz = 0; iz < nz; iz++)
		{
			z = z0 + dz * (iz + 0.5f);
			x = xSource + xoz * (z - zSource);
			y = ySource + yoz * (z - zSource);

			if (x >= x0 && x <= -x0 && y >= y0 && y <= -y0)
			{
				normx = x / xlen + 0.5f;
				normy = y / ylen + 0.5f;
				normz = z / zlen + 0.5f;

				total += tex3D(tex, normx, normy, normz);
			}
		}
	}

	worksino[gidx] = total * travVoxlen;
}

__global__ void forwardProjectionIndicatorKernel(float *indsino, float *worksino, float sinval, float cosval, float r, float s, int nu, int nv, float du, float dv, float u0, float v0,
	float dx, float dy, float dz, float x0, float y0, float z0, int nx, int ny, int nz, float xlen, float ylen, float zlen)
{
	int tidu = blockDim.x * blockIdx.x + threadIdx.x;
	int tidv = blockDim.y * blockIdx.y + threadIdx.y;
	int gidx = (nv - 1 - tidv) * nu + tidu;

	if (tidu >= nu || tidv >= nv)
		return;

	if (indsino[gidx] == 0)
	{
		worksino[gidx] = 0;
		return;
	}

	float xSource = r * cosval;
	float ySource = r * sinval;
	float zSource = 0;

	float xDetCenter = (r - s) * cosval;
	float yDetCenter = (r - s) * sinval;
	float zDetCenter = 0;

	float eux = -sinval;
	float euy = cosval;
	float euz = 0;

	float evx = 0;
	float evy = 0;
	float evz = 1;

	float u = u0 + (tidu + 0.5f) * du;
	float v = v0 + (tidv + 0.5f) * dv;

	float xbin = xDetCenter + eux * u + evx * v;
	float ybin = yDetCenter + euy * u + evy * v;
	float zbin = zDetCenter + euz * u + evz * v;

	float xdiff = xbin - xSource;
	float ydiff = ybin - ySource;
	float zdiff = zbin - zSource;

	float xad = fabsf(xdiff) / dx;
	float yad = fabsf(ydiff) / dy;
	float zad = fabsf(zdiff) / dz;

	float total = 0;

	float x, y, z, normx, normy, normz, travVoxlen;

	if (xad > yad && xad > zad)
	{
		float yox = ydiff / xdiff;
		float zox = zdiff / xdiff;
		travVoxlen = dx * sqrtf(1 + yox * yox + zox * zox);

		for (int ix = 0; ix < nx; ix++)
		{
			x = x0 + dx * (ix + 0.5);
			y = ySource + yox * (x - xSource);
			z = zSource + zox * (x - xSource);

			if (y >= y0 && y <= -y0 && z >= z0 && z <= -z0)
			{
				normx = x / xlen + 0.5f;
				normy = y / ylen + 0.5f;
				normz = z / zlen + 0.5f;

				total += tex3D(tex, normx, normy, normz);
			}
		}
	}
	else if (yad > zad)
	{
		float xoy = xdiff / ydiff;
		float zoy = zdiff / ydiff;
		travVoxlen = dy * sqrtf(1 + xoy * xoy + zoy * zoy);

		for (int iy = 0; iy < ny; iy++)
		{
			y = y0 + dy * (iy + 0.5f);
			x = xSource + xoy * (y - ySource);
			z = zSource + zoy * (y - ySource);

			if (x >= x0 && x <= -x0 && z >= z0 && z <= -z0)
			{
				normx = x / xlen + 0.5f;
				normy = y / ylen + 0.5f;
				normz = z / zlen + 0.5f;

				total += tex3D(tex, normx, normy, normz);
			}
		}
	}
	else
	{
		float xoz = xdiff / zdiff;
		float yoz = ydiff / zdiff;
		travVoxlen = dz * sqrtf(1 + xoz * xoz + yoz * yoz);

		for (int iz = 0; iz < nz; iz++)
		{
			z = z0 + dz * (iz + 0.5f);
			x = xSource + xoz * (z - zSource);
			y = ySource + yoz * (z - zSource);

			if (x >= x0 && x <= -x0 && y >= y0 && y <= -y0)
			{
				normx = x / xlen + 0.5f;
				normy = y / ylen + 0.5f;
				normz = z / zlen + 0.5f;

				total += tex3D(tex, normx, normy, normz);
			}
		}
	}

	worksino[gidx] = total * travVoxlen;
}

__global__ void backProjectionNormKernel(float *d_image, float *d_cons3, float F, float cosbeta, float sinbeta, float deltaZZ,
	float centernZZ, float deltaS, float centerBins, int nZZ, int nBins, int deltaBeta, int nx, int ny, int nz, float dx, float dy, float dz)
{
	int tidx = blockDim.x * blockIdx.x + threadIdx.x;
	int tidy = blockDim.y * blockIdx.y + threadIdx.y;
	int tidz = blockDim.z * blockIdx.z + threadIdx.z;
	int gidx = nx * ny * (nz - 1 - tidz) + nx * tidy + (nx - 1 - tidx);
	//int gidx = nx * ny * (nz - 1 - tidz) + nx * tidy + tidx;

	if (tidx >= nx || tidy >= ny || tidz >= nz)
		return;

	float d_x2 = (tidx - (nx - 1) / 2.0f) * dx;
	float d_y2 = (tidy - (ny - 1) / 2.0f) * dy;

	float S1 = F*(d_x2 * sinbeta + d_y2 * cosbeta) / (F + d_x2 * cosbeta - d_y2 * sinbeta);
	float U = (F + d_x2 * cosbeta - d_y2 * sinbeta) / F;

	float Z = (tidz - (nz - 1) / 2.0f) * dz / U;

	float Zind = Z / deltaZZ + centernZZ;
	float Gind = S1 / deltaS + centerBins;

	int y0 = int(Zind);
	int x0 = int(Gind);

	int y1 = y0 + 1;
	int x1 = x0 + 1;

	float frac_y = Zind - y0;
	float frac_x = Gind - x0;

	float between;

	if (x0<0 || x1 >= nBins || y0<0 || y1 >= nZZ)
		between = 0.0;
	else {
		int aa = x0 + y0*nBins;
		int cc = x1 + y0*nBins;
		float d_aa = d_cons3[aa];
		float d_cc = d_cons3[cc];
		float top = d_aa + frac_y * (d_cons3[x0 + y1*nBins] - d_aa);
		float bottom = d_cc + frac_y * (d_cons3[x1 + y1*nBins] - d_cc);
		between = top + frac_x * (bottom - top);
	}

	d_image[gidx] += between * deltaBeta / (U * U);
}

__global__ void backProjectionKernel(float *d_image, float *d_cons3, float F, float cosbeta, float sinbeta, float deltaZZ,
	float centernZZ, float deltaS, float centerBins, int nZZ, int nBins, int deltaBeta, float *d_norm, int nx, int ny, int nz, float dx, float dy, float dz)
{
	int tidx = blockDim.x * blockIdx.x + threadIdx.x;
	int tidy = blockDim.y * blockIdx.y + threadIdx.y;
	int tidz = blockDim.z * blockIdx.z + threadIdx.z;
	int gidx = nx * ny * (nz - 1 - tidz) + nx * tidy + (nx - 1 - tidx);
	//int gidx = nx * ny * (nz - 1 - tidz) + nx * tidy + tidx;

	if (tidx >= nx || tidy >= ny || tidz >= nz)
		return;

	float d_x2 = (tidx - (nx - 1) / 2.0f) * dx;
	float d_y2 = (tidy - (ny - 1) / 2.0f) * dy;

	float S1 = F*(d_x2 * sinbeta + d_y2 * cosbeta) / (F + d_x2 * cosbeta - d_y2 * sinbeta);
	float U = (F + d_x2 * cosbeta - d_y2 * sinbeta) / F;

	float Z = (tidz - (nz - 1) / 2.0f) * dz / U;

	float Zind = Z / deltaZZ + centernZZ;
	float Gind = S1 / deltaS + centerBins;

	int y0 = int(Zind);
	int x0 = int(Gind);

	int y1 = y0 + 1;
	int x1 = x0 + 1;

	float frac_y = Zind - y0;
	float frac_x = Gind - x0;

	float between;

	if (x0<0 || x1 >= nBins || y0<0 || y1 >= nZZ)
		between = 0.0;
	else {
		int aa = x0 + y0*nBins;
		int cc = x1 + y0*nBins;
		float d_aa = d_cons3[aa];
		float d_cc = d_cons3[cc];
		float top = d_aa + frac_y * (d_cons3[x0 + y1*nBins] - d_aa);
		float bottom = d_cc + frac_y * (d_cons3[x1 + y1*nBins] - d_cc);
		between = top + frac_x * (bottom - top);
	}

	float norm = d_norm[gidx];

	if (norm != 0)
		d_image[gidx] += between * deltaBeta / (U * U) / norm;
}

__global__ void correctiveImageKernel(float *data, float *proj, float *corImg, int len)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	if (idx >= len)
		return;

	corImg[idx] = data[idx] - proj[idx];
}

__global__ void divisionKernel(float *a, float *b, int len)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	if (idx >= len)
		return;

	if (b[idx] != 0)
		a[idx] /= b[idx];
}

__global__ void additionKernel(float *a, float *b, int len)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	if (idx >= len)
		return;

	a[idx] += b[idx];
}

__global__ void imageUpdatekernel(float *a, float *b, int len, float lamda)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	if (idx >= len)
		return;

	a[idx] += lamda * b[idx];
}

__global__ void makePositive(float *image, int len)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	if (idx >= len)
		return;

	if (image[idx] < 0)
		image[idx] = 0;
}

__global__ void nanAndInfCheck(float *f, int len)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	if (idx >= len)
		return;

	if (isinf(f[idx]) || isnan(f[idx]))
		f[idx] = 0;
}

__global__ void setArrVal(float *arr, int len, float val)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	if (idx >= len)
		return;

	arr[idx] = val;
}

__global__ void dataFlipV(float *data, int nu, int nv)
{
	int tidu = blockDim.x * blockIdx.x + threadIdx.x;
	int tidv = blockDim.y * blockIdx.y + threadIdx.y;
	int gidx = tidv * nu + tidu;
	
	if (tidu >= nu || tidv >= nv)
		return;

	float temp = data[gidx];
	int new_gidx = (nv - 1 - tidv) * nu + tidu;
	__syncthreads();

	data[new_gidx] = temp;
}

__global__ void dataFlipU(float *data, int nu, int nv)
{
	int tidu = blockDim.x * blockIdx.x + threadIdx.x;
	int tidv = blockDim.y * blockIdx.y + threadIdx.y;
	int gidx = tidv * nu + tidu;

	if (tidu >= nu || tidv >= nv)
		return;

	float temp = data[gidx];
	int new_gidx = tidv * nu + (nu - 1 - tidu);
	__syncthreads();

	data[new_gidx] = temp;
}

__global__ void zeroEdges(float *arr, int nx, int ny, int nz)
{
	int tidx = blockDim.x * blockIdx.x + threadIdx.x;
	int tidy = blockDim.y * blockIdx.y + threadIdx.y;
	int tidz = blockDim.z * blockIdx.z + threadIdx.z;
	int gidx = nx * ny * tidz + nx * tidy + tidx;

	if (tidx >= nx || tidy >= ny || tidz >= nz)
		return;

	if (tidx <= 1 || tidx >= nx - 2)
		arr[gidx] = 0;

	if (tidy <= 1 || tidy >= ny - 2)
		arr[gidx] = 0;

	if (tidz <= 1 || tidz >= nz - 2)
		arr[gidx] = 0;
}

__global__ void zeroOutsideFOV(float *img, int nx, int ny, int nz)
{
	int tidx = blockDim.x * blockIdx.x + threadIdx.x;
	int tidy = blockDim.y * blockIdx.y + threadIdx.y;
	int tidz = blockDim.z * blockIdx.z + threadIdx.z;
	int gidx = nx * ny * tidz + nx * tidy + tidx;

	if (tidx >= nx || tidy >= ny || tidz >= nz)
		return;

	tidx -= nx / 2;
	tidy -= ny / 2;
	int r = (int)sqrtf(tidx * tidx + tidy * tidy);

	if (r > nx / 2)
		img[gidx] = 0;
}