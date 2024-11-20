import numpy as np
import matplotlib.pyplot as plt
import math
import time

class Kernel: 
    def __init__(self, radius, sigma) -> None:
        self.radius = radius
        self.sigma = sigma
        self.kernel = self.initiateDonutKernel(radius * 2 + 1, radius, sigma)
        
    def initiateDonutKernel(self, kernel_size, radius, sigma):
        kernel = np.zeros((kernel_size, kernel_size))
        center = kernel_size // 2
        for i in range(kernel_size):
            for j in range(kernel_size):
                dist = math.sqrt((i - center)**2 + (j - center)**2)
                # Softer donut kernel with peak at the ring
                kernel[i, j] = np.exp(-((dist - radius)**2) / (2 * sigma**2))
        
        # Normalize kernel to sum to 1
        kernel /= kernel.sum()
        return kernel
     
    def convolute(self, A, i, j):
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
        plt.figure(figsize=(8,6))
        plt.imshow(self.kernel, cmap='viridis', interpolation='nearest')
        plt.colorbar()
        plt.title("Kernel Visualization")
        plt.show()

class GrowthFunc:
    def __init__(self, mu, sigma) -> None:
        self.mu = mu
        self.sigma = sigma
    
    def gaussianFunc(self, x): 
        # Smoother growth function centered around mu
        # Returns values between 0 and 1
        return np.exp(-((x - self.mu)**2) / (2 * self.sigma**2))
    
    def __call__(self, *args, **kwargs):
        return self.gaussianFunc(*args, **kwargs)

class Lenia:
    def __init__(self, C_0, kernel_radius, kernel_sigma, growth_center, growth_width) -> None:
        self.C = C_0.copy()
        self.G = GrowthFunc(growth_center, growth_width)
        self.K = Kernel(kernel_radius, kernel_sigma)
        self.N = C_0.shape[0]
    
    def run(self, dt, steps, visualize=False):
        if visualize:
            
            fig, ax = plt.subplots(figsize=(8,6))
            img = ax.imshow(self.C, cmap='viridis', interpolation='nearest', vmin=0, vmax=1)
            plt.colorbar(img, ax=ax)

        for step in range(steps):
            # Evolve the grid
            new_C = np.zeros_like(self.C)
            for i in range(self.N):
                for j in range(self.N):
                    # Convolute the kernel over the cell state matrix
                    U = self.K.convolute(self.C, i, j)
                    
                    # Get the activation function value
                    A = self.G.gaussianFunc(U)
                    
                    # Update cell state with activation value
                    new_C[i, j] = self.C[i, j] + dt * (A - 0.5) * 2
                    
                    # Restrict values to [0, 1]
                    new_C[i, j] = np.clip(new_C[i, j], 0, 1)
            
            self.C = new_C
            
            # Update visualization
            if visualize:
                img.set_data(self.C)
                plt.title(f"Step {step + 1}")
                plt.draw()
                plt.pause(0.1)

        if visualize:
            plt.ioff()
            plt.show()

# Example usage:
if __name__ == "__main__":
    N = 50  # Increased grid size
    dt = 0.2
    steps = 200
    

    C_0 = np.random.rand(N, N)
    
    # Initialize C_0 with a circular pattern in the center
    

    lenia = Lenia(
        C_0=C_0,
        kernel_radius=4,  # Increased radius
        kernel_sigma=3.0,  # Adjusted sigma
        growth_center=0.5,  # Centered growth
        growth_width=0.1,  # Reasonable growth width
    )
    
    # Uncomment to debug kernel
    # lenia.K.visualize()

    lenia.run(dt=dt, steps=steps, visualize=True)