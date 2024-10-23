#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>
#define N 1024
__global__ void bitonicSort(int *arr, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    



    

}

__global__ void bitonicMerge(int *arr, int n) {

}

void fillRandom(float *arr, int n) {
    for ( int i = 0; i < n; i++) {
        arr[i] = rand() / (float)RAND_MAX;
    }
}

void printarr(float * h) {
    for ( int i = 0; i < N; i++) {
        printf("%f ", h[i]);
        printf("\n");
    }
}

int main() {
    float * ha, *da;
    int size = N * sizeof(float);
    ha = (float *)malloc(size);
    cudaMalloc((void **)&da, size);
    
    fillRandom(ha, N);
    cudaMemcpy(da, ha, size, cudaMemcpyHostToDevice);
    bitonicSort(da, N);
    cudaMemcpy(ha, da, size, cudaMemcpyDeviceToHost);

    // ha contains teh sorted array
    printarr(ha);

    free(ha);
    cudaFree(da);

}

