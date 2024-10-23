#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>
#define N 1024

__device__ void compareAndSwap(int *arr, int index1, int index2, bool ascending) {
    if ((ascending && arr[index1] > arr[index2]) || (!ascending && arr[index1] < arr[index2])) {
        int temp = arr[index1];
        arr[index1] = arr[index2];
        arr[index2] = temp;
    }
}

__global__ void bitonicMerge(int *arr, int subsequenceSize, int compareDistance) {
    int threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
    int partnerIndex = threadIndex ^ compareDistance;
    
    if (partnerIndex < N && threadIndex < partnerIndex) {
        bool ascending = (threadIndex & subsequenceSize) == 0;
        compareAndSwap(arr, threadIndex, partnerIndex, ascending);
    }
}

__host__ void bitonicSort(int *arr, int n) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    for (int k = 2; k <= n; k *= 2) {
        for (int j = k / 2; j > 0; j /= 2) {
            bitonicMerge<<<blocksPerGrid, threadsPerBlock>>>(arr, k, j);
            cudaDeviceSynchronize();
        }
    }
}

void fillRandom(int *arr, int n) {
    for (int i = 0; i < n; i++) {
        arr[i] = rand();
    }
}

void printArray(int *arr, int n) {
    for (int i = 0; i < n; i++) {
        printf("%d ", arr[i]);
        if ((i + 1) % 16 == 0) printf("\n");
    }
    printf("\n");
}

int main() {
    int *ha, *da;
    int size = N * sizeof(int);
    ha = (int *)malloc(size);
    cudaMalloc((void **)&da, size);
    
    fillRandom(ha, N);
    
    cudaMemcpy(da, ha, size, cudaMemcpyHostToDevice);
    
    printf("Unsorted array:\n");
    printArray(ha, N);
    
    bitonicSort(da, N);
    
    cudaMemcpy(ha, da, size, cudaMemcpyDeviceToHost);

    printf("\nSorted array:\n");
    printArray(ha, N);

    free(ha);
    cudaFree(da);

    return 0;
}
