import math
import numpy as np



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