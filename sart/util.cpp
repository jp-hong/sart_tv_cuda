

#include "util.h"


void** generateArray(int x, int y, int size_t){
	void** f = (void**)malloc(x * sizeof(void*));

    for (int i = 0; i < x; i++)
		f[i] = malloc(y * size_t);

	return f;
}

void*** generateArray(int x, int y, int z, int size_t){
	void*** f = (void***)malloc(x * sizeof(void*));

	for(int i = 0; i < x; i++){
		f[i] = (void**)malloc(y * sizeof(void*));

		for (int j = 0; j < y; j++)
			f[i][j] = malloc(z * size_t);
	}

	return f;
}

void freeArray(void **f, int x){
	for(int i = 0; i < x; i++)
		free(f[i]);

    free(f);
}

void freeArray(void ***f, int x, int y){
	for(int i = 0; i < x; i++){
		for(int j = 0; j < y; j++)
			free(f[i][j]);

		free(f[i]);
	}

	free(f);
}

void copyArray(void **dst, void **src, int size_t, int x, int y){
	for (int i = 0; i < x; i++)
		memcpy((void*)dst[i], (void*)src[i], y * size_t);
}

void copyArray(void ***dst, void ***src, int size_t, int x, int y, int z){
	for (int i = 0; i < x; i++)
		for (int j = 0; j < y; j++)
			memcpy((void*)dst[i][j], (void*)src[i][j], z * size_t);
}

void readArrayFromFile(void *arr, int x, char *fileName, int size_t){
	FILE *fp = fopen(fileName, "rb");
	
	if(fp == NULL){
		printf("Failed to open file \"%s\"\n", fileName);
		exit(EXIT_FAILURE);
	}

	fread(arr, size_t, x, fp);
	fclose(fp);
}

void readArrayFromFile(void **arr, int x, int y, char *fileName, int size_t){
	FILE *fp = fopen(fileName, "rb");

	if(fp == NULL){
		printf("Failed to open file \"%s\"\n", fileName);
		exit(EXIT_FAILURE);
	}

	for(int i = 0; i < x; i++)
		fread(arr[i], size_t, y, fp);

	fclose(fp);
}

void readArrayFromFile(void ***arr, int x, int y, int z, char *fileName, int size_t){
	FILE *fp = fopen(fileName, "rb");

	if(fp == NULL){
		printf("Failed to open file \"%s\"\n", fileName);
		exit(EXIT_FAILURE);
	}

	for (int i = 0; i < x; i++)
		for (int j = 0; j < y; j++)
			fread(arr[i][j], size_t, z, fp);

	fclose(fp);
}

void writeArrayToFile(void ***arr, int x, int y, int z, char *fileName, int size_t){
	FILE *fp = fopen(fileName, "wb");

	for (int i = 0; i < x; i++)
		for (int j = 0; j < y; j++)
			fwrite(arr[i][j], size_t, z, fp);

	fclose(fp);
}

void writeArrayToFile(void **arr, int x, int y, char *fileName, int size_t){
	FILE *fp = fopen(fileName, "wb");

	for (int i = 0; i < x; i++)
		fwrite(arr[i], size_t, y, fp);

	fclose(fp);
}

void writeArrayToFile(void *arr, int x, char *fileName, int size_t){
	FILE *fp = fopen(fileName, "wb");
	fwrite(arr, size_t, x, fp);
	fclose(fp);
}

void generateFileName(char *dst, char *directory, char *fileName){
	sprintf(dst, directory);
	strcat(dst, fileName);
}

void generateFileName(char *dst, char *directory, char *fileNamePrefix, char *fileNameSuffix, char *fileNameExtension){
	sprintf(dst, directory);
	strcat(dst, fileNamePrefix);
	strcat(dst, fileNameSuffix);
	strcat(dst, fileNameExtension);
}

void generateFileName(char *dst, char *directory, char *fileNamePrefix, int idx, char *fileNameExtension){
	char fidx[256];
	sprintf(fidx, "%04d", idx);
	sprintf(dst, directory);
	strcat(dst, fileNamePrefix);
	strcat(dst, fidx);
	strcat(dst, fileNameExtension);
}

void getGridDim(dim3 *GRID, dim3 BLOCK, int len){
	if (len % BLOCK.x == 0)
		GRID->x = len / BLOCK.x;
	else
		GRID->x = len / BLOCK.x + 1;
}