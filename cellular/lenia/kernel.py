import numpy as np
import math

class Kernel: 
    def __init__(self, radius, sigma) -> None:
        self.radius = radius
        self.sigma = sigma
        self.kernel = self.initiateGaussianKernel(radius * 2 , sigma)
        
        
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
        