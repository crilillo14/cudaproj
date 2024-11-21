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


#define N 54
// kernel params
#define RADIUS 12
#define SIGMA 2.5
// Growth function parameters
#define GROWTH_CENTER 0.18
#define GROWTH_WIDTH 0.06
// time step between states
#define DT 0.1
// number of steps
#define NUMSTEPS 90




// device functions

__device__ void convolution(double *C, double *kernel, int i, int j, double *result , double kernelMax) {
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
    *result = sum / kernelMax;
}



// Gaussian based growthfunc.
__device__ double growthFunc(double x) {
    double normalized = exp(-0.5 * pow((x - GROWTH_CENTER) / GROWTH_WIDTH, 2));
    return 2 * normalized - 1;
}




// main kernel

__global__ void leniaKernel(double *C, double *kernel, double *newC, double kernelMax) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < N && j < N) {
        double U = 0.0;
        convolution(C, kernel, i, j, &U, kernelMax);

        // Debugging convolution result
        // printf("Thread (%d, %d): Convolution result U = %f\n", i, j, U);

        double A = growthFunc(U);
        // printf("Thread (%d, %d): Growth result A = %f\n", i, j, A);
        // printf("Thread (%d, %d) ; previous state = %f , next state :%f" , i , j, C[i*N+j] , C[i * N + j] + DT * A);
        newC[i * N + j] = C[i * N + j] + DT * A ;

        // Clamp values to [0, 1] and debug final state
        newC[i * N + j] = fmin(fmax(newC[i * N + j], 0.0), 1.0);
        // printf("Thread (%d, %d): Updated newC = %f\n", i, j, newC[i * N + j]);
    }
}



// _____________________________  KERNELS __________________________




// Host function to initialize the Gaussian kernel
void initGaussianKernel(double *kernel) {
    int kernelSize = 2 * RADIUS + 1;

    for (int i = -RADIUS; i <= RADIUS; i++) {
        for (int j = -RADIUS; j <= RADIUS; j++) {
            int idx = (i + RADIUS) * kernelSize + (j + RADIUS);
            kernel[idx] = exp(-0.5 * (i * i + j * j) / (SIGMA * SIGMA));
        }
    }
}

double getKernelMax(double *kernel) {
    int kernelSize = 2 * RADIUS + 1;
    double kernelMax = 0.0;

    for (int i = -RADIUS; i <= RADIUS; i++) {
        for (int j = -RADIUS; j <= RADIUS; j++) {
            int idx = (i + RADIUS) * kernelSize + (j + RADIUS);
            kernelMax += kernel[idx];
        }
    }

    return kernelMax;
}

void initRippleKernel(double *kernel) {
    int kernelSize = 2 * RADIUS + 1;
    double strength = 1.0;
    double radius = RADIUS;

    for (int i = -RADIUS; i <= RADIUS; i++) {
        for (int j = -RADIUS; j <= RADIUS; j++) {
            int idx = (i + RADIUS) * kernelSize + (j + RADIUS);
            double distance = sqrt(i * i + j * j);

            // Create wave-like effects using sine function
            if (distance < radius) {
                kernel[idx] = strength * sin(distance / radius * M_PI);
            } else {
                kernel[idx] = 0.0;
            }
        }
    }
}


void initDonutKernel(double *kernel) {
    int kernelSize = 2 * RADIUS + 1;

    double innerSigma = SIGMA / 2.0; // Controls the "hole" size
    double outerSigma = SIGMA;      // Controls the overall radius of the donut

    for (int i = -RADIUS; i <= RADIUS; i++) {
        for (int j = -RADIUS; j <= RADIUS; j++) {
            int idx = (i + RADIUS) * kernelSize + (j + RADIUS);
            double distanceSquared = i * i + j * j;

            // Outer Gaussian
            double outer = exp(-0.5 * distanceSquared / (outerSigma * outerSigma));
            // Inner Gaussian (negative contribution)
            double inner = exp(-0.5 * distanceSquared / (innerSigma * innerSigma));

            // Donut shape by subtracting inner Gaussian from outer Gaussian
            kernel[idx] = outer - inner;
        }
    }

}


// ___________________________ GRID CONFIGS ______________________


void initGrid(double *C) {
    srand(time(NULL));  // Initialize random seed with current time
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            C[i * N + j] = (double)rand() / RAND_MAX;  // Random value between 0 and 1
        }
    }
}

void initRandomSparseGrid(double *grid) {
    for (int i = 0; i < N * N; i++) {
        grid[i] = (rand() % 10 == 0) ? 1.0 : 0.0; // Randomly place "cells" at sparse intervals
    }
}

void initLobeKernel(double *kernel) {
    int kernelSize = 2 * RADIUS + 1;
    double strength = 1.0;
    double lobeAngle = M_PI / 4.0;  // Width of each lobe

    for (int i = -RADIUS; i <= RADIUS; i++) {
        for (int j = -RADIUS; j <= RADIUS; j++) {
            int idx = (i + RADIUS) * kernelSize + (j + RADIUS);
            double distance = sqrt(i * i + j * j);
            double angle = atan2(j, i);

            // Create lobe-like structures
            if (distance < RADIUS) {
                if (fmod(angle, lobeAngle) < 0.2) {
                    kernel[idx] = strength;
                } else {
                    kernel[idx] = 0.0;
                }
            } else {
                kernel[idx] = 0.0;
            }
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
    // printf("\033[2J\033[H");
    printf("\n");
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


    ha = (double *)malloc(size);
    cudaMalloc((void **)&da, size);
    cudaMalloc((void **)&dKernel, kernelSize);
    cudaMalloc((void **)&dNewC, size);

    // Initialize grid and kernel on host
    initGrid(ha);
    // initRandomSparseGrid(ha);
    //initGridGlider(ha);
    // initGridChunks(ha);



    double *hostKernel = (double *)malloc(kernelSize);
    // initGaussianKernel(hostKernel);
    // initDonutKernel(hostKernel);
    // initLobeKernel(hostKernel);
    initRippleKernel(hostKernel);
    double kernelMax = getKernelMax(hostKernel);
    printf("kernel max: %f\n" , kernelMax);
    // initDonutKernel(hostKernel, (2 * RADIUS + 1), RADIUS / 2.0, SIGMA);

    // Copy data to device
    cudaMemcpy(da, ha, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dKernel, hostKernel, kernelSize, cudaMemcpyHostToDevice);

    // Print initial state
    printf("Initial state:\n");
    // printGridASCII(ha);
    // printGridThermal(ha);
    printGridViridis(ha);

    // Run sim NUMSTEPS times
    // given by ai, not sure optimal values
    int threadsPerBlock = 16;
    dim3 blockSize(threadsPerBlock, threadsPerBlock);
    dim3 gridSize((N + threadsPerBlock - 1) / threadsPerBlock, (N + threadsPerBlock - 1) / threadsPerBlock);

    for (int step = 0; step < NUMSTEPS; step++) {
        leniaKernel<<<gridSize, blockSize>>>(da, dKernel, dNewC, kernelMax);
        cudaDeviceSynchronize();

        // std::this_thread::sleep_for(std::chrono::milliseconds(20));

        double *temp = dNewC;
        da = dNewC;
        dNewC = temp;

        // printGridASCII(ha);
        // printGridThermal(ha);
        printGridViridis(ha);

        cudaMemcpy(ha, da, size, cudaMemcpyDeviceToHost);
    }

    // Print the final state
    printf("\nFinal state after %d steps:\n", NUMSTEPS);
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