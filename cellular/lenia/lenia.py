import numpy as np
import matplotlib.pyplot as plt
import math
import time
        # Your kernel implementation


class Kernel: 
    def __init__(self, radius, sigma) -> None:
        self.radius = radius
        self.sigma = sigma
        # self.kernel = self.initiateGaussianKernel(radius * 2 + 1, sigma)
        self.kernel = self.initiateDonutKernel(radius * 2 + 1, radius, sigma)
        
        self.kernel_max = self.kernel.sum()
        
        # for ease of computation
    
    def initiateGaussianKernel(self, kernel_size, sigma):
        kernel = np.zeros((kernel_size , kernel_size))
        for i in range(kernel_size):
            for j in range(kernel_size):
                dist_squared = (i - self.radius)**2 + (j - self.radius)**2
                # Apply Gaussian function
                kernel[i, j] = np.exp(-dist_squared / (2 * sigma**2))
        return kernel
    
    def initiateDonutKernel(self, kernel_size, radius, sigma):
        kernel = np.zeros((kernel_size, kernel_size))
        center = kernel_size // 2  # Center of the kernel
        for i in range(kernel_size):
            for j in range(kernel_size):
                dist = math.sqrt((i - center)**2 + (j - center)**2)
                # Apply donut-shaped function
                kernel[i, j] = np.exp(-((dist - radius)**2) / (2 * sigma**2))
        return kernel

     
    def getKernel(self):
        return self.kernel

    def convolute(self, A , i , j):
        
        kernel_size = self.kernel.shape[0]
        result = 0.0
        for ki in range(kernel_size):
            for kj in range(kernel_size):
                    
                ai = i + ki - self.radius
                aj = j + kj - self.radius
            
            if 0 <= ai < A.shape[0] and 0 <= aj < A.shape[1]:
                result += A[ai][aj] * self.kernel[ki][kj]
        
        return result
    
    def visualize(self):
        plt.imshow(self.kernel, cmap='viridis', interpolation='nearest')
        plt.colorbar()
        plt.show()
        


class GrowthFunc:
    def __init__(self, mu , sigma) -> None:
        self.mu = mu
        self.sigma = sigma
    
    def gaussianFunc(self, x): 
        
        # should yield a gaussian with amplitude 2, translated down 1. Either cell increases or decreases
        # not exactly gaussian, just enough so that it replicates 
        
        return 2 * np.exp(-0.5 * math.pow((x - self.mu) / self.sigma , 2)) - 1
    
    def __call__(self, *args: np.any, **kwds: np.any) -> np.any:
        return self.gaussianFunc(*args, **kwds)



class Lenia:
    def __init__(self, C_0, kernel_radius, kernel_sigma, growth_center, growth_width) -> None:
        self.C = C_0
        self.G = GrowthFunc(growth_center, growth_width)
        self.K = Kernel(kernel_radius, kernel_sigma)
        self.N = C_0.shape[0]
        self.kernel_max = self.K.kernel.sum()  # Normalize kernel
    
    def run(self, dt, steps, visualize=False):
        if visualize:
            plt.ion()  # Turn on interactive mode
            fig, ax = plt.subplots()
            img = ax.imshow(self.C, cmap='viridis', interpolation='nearest')
            plt.colorbar(img, ax=ax)

        for step in range(steps):
            time.sleep(1)
            # Evolve the grid
            new_C = np.zeros_like(self.C)
            for i in range(self.N):
                for j in range(self.N):
                    # Convolute the kernel over the cell state matrix
                    U = self.K.convolute(self.C, i, j) / self.kernel_max
                    # Get the activation function value
                    A = self.G.gaussianFunc(U)
                    # Update cell state with activation value
                    new_C[i, j] = self.C[i, j] + dt * A
                    # Restrict values to [0, 1]
                    new_C[i, j] = max(0, min(1, new_C[i, j]))
            self.C = new_C
            
            # Update visualization
            if visualize:
                img.set_data(self.C)
                plt.title(f"Step {step + 1}")
                plt.draw()
                plt.pause(0.1)  # Pause to create animation effect

        if visualize:
            plt.ioff()  # Turn off interactive mode
            plt.show()
            

# Example usage:
if __name__ == "__main__":
    N = 40
    dt = 0.1
    steps = 200
    C_0 = np.random.rand(N, N)
    # C_0 = np.zeros((N, N))
    # center = N // 2
    # C_0[center - 2:center + 3, center - 2:center + 3] = 0.5  # Initialize with a small square

    lenia = Lenia(
        C_0=C_0,
        kernel_radius=5,
        kernel_sigma=2.0,  # Larger sigma for smoother spread
        growth_center=0.3,  # Shift growth function to align with kernel range
        growth_width=0.05,  # Wider growth width for smoother dynamics
    )
    
    # debug kernel
    # lenia.K.visualize()

    lenia.run(dt=dt, steps=steps, visualize=True)

