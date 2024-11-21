#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <unistd.h> // time.sleep equivalent
#include <time.h>
#include <math.h>
#include <thread>
#include <chrono>
#include <iostream>
#include <cstdio>
#include <cmath>
#include <algorithm>

#define N 50

// kernel params
#define RADIUS 5
#define SIGMA 3.0

// Growth function parameters
#define GROWTH_CENTER 0.5
#define GROWTH_WIDTH 0.5

// time step between states
#define DT 10

// number of steps
#define NUMSTEPS 100


__device__ void convolution(double *C, double *kernel, int i, int j, double *result) {
    int kernelSize = 2 * RADIUS + 1;
    double sum = 0.0;
    for (int ki = -RADIUS; ki <= RADIUS; ki++) {
        for (int kj = -RADIUS; kj <= RADIUS; kj++) {
            int ni = i + ki;
            int nj = j + kj;
            if (ni >= 0 && ni < N && nj >= 0 && nj < N) {
                sum += C[ni * N + nj] * kernel[(ki + RADIUS) * kernelSize + (kj + RADIUS)];
            }
        }
    }
    *result = sum;
}

// CUDA kernel for the growth function (Gaussian function)
__device__ double growthFunc(double x) {
    return 2.0 * exp(-0.5 * pow((x - GROWTH_CENTER) / GROWTH_WIDTH, 2)) - 1.0;
}

// CUDA kernel for one step of the simulation
__global__ void leniaKernel(double *C, double *kernel, double *newC) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < N && j < N) {
        double U = 0.0;
        convolution(C, kernel, i, j, &U);
        double A = growthFunc(U);
        newC[i * N + j] = C[i * N + j] + DT * A;
        newC[i * N + j] = fmin(fmax(newC[i * N + j], 0.0), 1.0); // Clamp values to [0, 1]
    }
}

// Host function to initialize the Gaussian kernel
void initGaussianKernel(double *kernel, double kernelMax) {
    int kernelSize = 2 * RADIUS + 1;


    for (int i = -RADIUS; i <= RADIUS; i++) {
        for (int j = -RADIUS; j <= RADIUS; j++) {
            int idx = (i + RADIUS) * kernelSize + (j + RADIUS);
            kernel[idx] = exp(-0.5 * (i * i + j * j) / (SIGMA * SIGMA));
            kernelMax += kernel[idx];
        }
    }
}

// not in use
// use either this or the gaussian kernel
void initDonutKernel(double *kernel, int kernelSize, double radius, double sigma) {
    int center = kernelSize / 2;

    for (int i = 0; i < kernelSize; i++) {
        for (int j = 0; j < kernelSize; j++) {
            // Calculate distance from center
            double dist = sqrt(pow(i - center, 2) + pow(j - center, 2));

            // Calculate donut shape with peak at specified radius
            kernel[i * kernelSize + j] = exp(-pow(dist - radius, 2) / (2 * sigma * sigma));
        }
    }
}

// random initialization
void initGrid(double *C) {
    srand(time(NULL));  // Initialize random seed with current time
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            C[i * N + j] = (double)rand() / RAND_MAX;  // Random value between 0 and 1
        }
    }
}










// Helper function: Interpolate between two values
double lerp(double a, double b, double t) {
    return a + t * (b - a);
}

// Map value to Viridis gradient color
void valueToViridisColor(double value) {
    // Ensure value is between 0 and 1
    value = fmin(fmax(value, 0.0), 1.0);

    // Viridis RGB control points (normalized from 0 to 1)
    const double viridis[6][3] = {
        {0.267, 0.004, 0.329},  // Dark blue (low)
        {0.282, 0.140, 0.457},
        {0.253, 0.265, 0.529},
        {0.163, 0.471, 0.558},
        {0.134, 0.658, 0.517},
        {0.993, 0.906, 0.144}   // Yellow (high)
    };

    // Map value to one of the 5 segments
    int idx = (int)(value * 5.0);
    double segmentT = (value * 5.0) - idx;

    // Linearly interpolate between the two colors in the segment
    double r = lerp(viridis[idx][0], viridis[idx + 1][0], segmentT);
    double g = lerp(viridis[idx][1], viridis[idx + 1][1], segmentT);
    double b = lerp(viridis[idx][2], viridis[idx + 1][2], segmentT);

    // Scale RGB values to 0-255
    int R = (int)(r * 255);
    int G = (int)(g * 255);
    int B = (int)(b * 255);

    // Use ANSI escape codes for 24-bit RGB color
    printf("\033[48;2;%d;%d;%dm  \033[0m", R, G, B);
}

// Print grid using Viridis gradient colors
void printGridViridis(double *C) {
    // Clear screen using ANSI escape codes
    printf("\033[2J\033[H");
    // printf("\n");
    // Print grid content without borders
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            valueToViridisColor(C[i * N + j]);  // Print colored block
        }
        printf("\n");  // Newline after each row
    }

    // Reset color to default after printing
    printf("\033[0m");
}




// Main function
int main() {
    // ha is C on host, da is C on device, dKernel is the kernel on device, dNewC is the new C on device
    double *ha, *da, *dKernel, *dNewC;

    // size of grid C , given NxN grid and double type
    int size = N * N * sizeof(double);

    // kernel measured from convolution center with radius, thats why size of kernel arry is 2*RADIUS+1
    size_t kernelSize = (2 * RADIUS + 1) * (2 * RADIUS + 1) * sizeof(double);
    double kernelMax = 0.0;


    ha = (double *)malloc(size);
    cudaMalloc((void **)&da, size);
    cudaMalloc((void **)&dKernel, kernelSize);
    cudaMalloc((void **)&dNewC, size);

    // Initialize grid and kernel on host
    // initGrid(ha);
    //initGridGlider(ha);
    initGridChunks(ha);
    double *hostKernel = (double *)malloc(kernelSize);
    initGaussianKernel(hostKernel , kernelMax);
    // initDonutKernel(hostKernel, (2 * RADIUS + 1), RADIUS / 2.0, SIGMA);

    // Copy data to device
    cudaMemcpy(da, ha, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dKernel, hostKernel, kernelSize, cudaMemcpyHostToDevice);

    // Print initial state
    printf("Initial state:\n");
    // printGridASCII(ha);
    // printGridThermal(ha);
    printGridViridis(ha);

    // Run sim 100 times
    // given by ai, not sure optimal values
    int threadsPerBlock = 16;
    dim3 blockSize(threadsPerBlock, threadsPerBlock);
    dim3 gridSize((N + threadsPerBlock - 1) / threadsPerBlock, (N + threadsPerBlock - 1) / threadsPerBlock);



    for (int step = 0; step < NUMSTEPS; step++) {
        leniaKernel<<<gridSize, blockSize>>>(da, dKernel, dNewC);
        cudaDeviceSynchronize();

        // std::this_thread::sleep_for(std::chrono::milliseconds(200));

        // printGridASCII(ha);
        // printGridThermal(ha);
        printGridViridis(ha);
        cudaMemcpy(ha, dNewC, size, cudaMemcpyDeviceToHost);
    }

    // Print the final state
    printf("\nFinal state after 100 steps:\n");
    // printGridASCII(ha);
    // printGridThermal(ha);
    printGridViridis(ha);
    // Free memory
    free(ha);
    free(hostKernel);
    cudaFree(da);
    cudaFree(dKernel);
    cudaFree(dNewC);

    return 0;
}
