#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <unistd.h> // time.sleep equivalent
#include <time.h>
#include <math.h>

#define N 100

// kernel params
#define RADIUS 5
#define SIGMA 1.0

// Growth function parameters
#define GROWTH_CENTER 0.15
#define GROWTH_WIDTH 0.02

// time step between states
#define DT 0.1

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
__global__ void leniaKernel(double *C, double *kernel, double *newC, double dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i < N && j < N) {
        double U = 0.0;
        convolution(C, kernel, i, j, &U);
        double A = growthFunc(U);
        newC[i * N + j] = C[i * N + j] + dt * (A - 0.5) * 2;
        newC[i * N + j] = fmin(fmax(newC[i * N + j], 0.0), 1.0); // Clamp values to [0, 1]
    }
}

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


void initGrid(double *C) {
    srand(time(NULL));  // Initialize random seed with current time
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            C[i * N + j] = (double)rand() / RAND_MAX;  // Random value between 0 and 1
        }
    }
}


void printGrid(double *C) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%.2f ", C[i * N + j]);
        }
        printf("\n");
    }
}


// Maps a double value between 0 and 1 to an ASCII character
char valueToAscii(double value) {
    // Define ASCII characters from least dense to most dense
    const char *density = " .:-=+*#%@";
    const int numChars = 10;  // Length of the density string

    // Ensure value is between 0 and 1
    value = fmin(fmax(value, 0.0), 1.0);

    // Map the value to an index in our density string
    int index = (int)(value * (numChars - 1));

    return density[index];
}

// Print the grid using ASCII characters
void printGridASCII(double *C) {
    // Clear screen using ANSI escape code
    printf("\033[2J\033[H");

    // Print top border
    printf("+");
    for (int j = 0; j < N; j++) printf("-");
    printf("+\n");

    // Print grid content
    for (int i = 0; i < N; i++) {
        printf("|");  // Left border
        for (int j = 0; j < N; j++) {
            char ascii = valueToAscii(C[i * N + j]);
            printf("%c", ascii);
        }
        printf("|\n");  // Right border
    }

    // Print bottom border
    printf("+");
    for (int j = 0; j < N; j++) printf("-");
    printf("+\n");

        // Print legend
    printf("\nDensity Scale: ");
    for (int i = 0; i < 10; i++) {
        printf("%c", valueToAscii(i / 9.0));
    }
    printf("\n0.0 -------- 1.0\n");

}
// Main function
int main() {
    // ha is C on host, da is C on device, dKernel is the kernel on device, dNewC is the new C on device
    double *ha, *da, *dKernel, *dNewC;

    // size of grid C , given NxN grid and double type
    int size = N * N * sizeof(double);

    // kernel measured from convolution center with radius, thats why size of kernel arry is 2*RADIUS+1
    double kernelSize = (2 * RADIUS + 1) * (2 * RADIUS + 1) * sizeof(double);


    ha = (double *)malloc(size);
    cudaMalloc((void **)&da, size);
    cudaMalloc((void **)&dKernel, kernelSize);
    cudaMalloc((void **)&dNewC, size);
    
    // Initialize grid and kernel on host
    initGrid(ha);
    double *hostKernel = (double *)malloc(kernelSize);
    // initGaussianKernel(hostKernel);
    initDonutKernel(hostKernel, kernelSize, RADIUS/2.0, SIGMA);
    // Copy data to device
    cudaMemcpy(da, ha, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dKernel, hostKernel, kernelSize, cudaMemcpyHostToDevice);
    
    // Print initial state
    printf("Initial state:\n");
    printGridASCII(ha);
    
    // Run sim 100 times
    // given by ai, not sure optimal values
    int threadsPerBlock = 16;
    dim3 blockSize(threadsPerBlock, threadsPerBlock);
    dim3 gridSize((N + threadsPerBlock - 1) / threadsPerBlock, (N + threadsPerBlock - 1) / threadsPerBlock);

    
    
    for (int step = 0; step < 100; step++) {
        leniaKernel<<<gridSize, blockSize>>>(da, dKernel, dNewC);
        cudaDeviceSynchronize();
        
        printGrid(ha);
        cudaMemcpy(ha, dNewC, size, cudaMemcpyDeviceToHost);
    }
    
    // Print the final state
    printf("\nFinal state after 100 steps:\n");
    printGrid(ha);
    
    // Free memory
    free(ha);
    free(hostKernel);
    cudaFree(da);
    cudaFree(dKernel);
    cudaFree(dNewC);
    
    return 0;
}
