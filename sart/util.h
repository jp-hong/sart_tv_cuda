

#ifndef _MY_UTIL
#define _MY_UTIL

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define MAX(a, b) (a > b? a : b)

#define MIN(a, b) (a < b? a : b)

#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__))

static void HandleError(cudaError_t err, const char *file, int line){
	if (err != cudaSuccess){
		printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
		exit(EXIT_FAILURE);
	}
}

inline int iDivUp(int a, int b)
{
	return ((a % b) != 0) ? (a / b + 1) : (a / b);
}

void** generateArray(int x, int y, int size_t);
void*** generateArray(int x, int y, int z, int size_t);

void freeArray(void **f, int x);
void freeArray(void ***f, int x, int y);

void copyArray(void **dst, void **src, int size_t, int x, int y);
void copyArray(void ***dst, void ***src, int size_t, int x, int y, int z);

void readArrayFromFile(void *arr, int x, char *fileName, int size_t);
void readArrayFromFile(void ***arr, int x, int y, int z, char *fileName, int size_t);
void readArrayFromFile(void **arr, int x, int y, char *fileName, int size_t);

void writeArrayToFile(void *arr, int x, char *fileName, int size_t);
void writeArrayToFile(void **arr, int x, int y, char *fileName, int size_t);
void writeArrayToFile(void ***arr, int x, int y, int z, char *fileName, int size_t);

void generateFileName(char *dst, char *directory, char *fileName);
void generateFileName(char *dst, char *directory, char *fileNamePrefix, char *fileNameSuffix, char *fileNameExtension);
void generateFileName(char *dst, char *directory, char *fileNamePrefix, int idx, char *fileNameExtension);

//CUDA thread grid dimension calculator
void getGridDim(dim3 *GRID, dim3 BLOCK, int len);


#endif