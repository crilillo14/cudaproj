
### Lenia cell state: 

L^(t+dt) = [L^t + dt*G(K*L^t)] 

A is used in the paper, but L is more fitting, since I want to name the Activation of the growth function A. 

## Steps to get next state L^(t + dt): 

1. Convolute the weighted kernel K over A^t
    1. given the sum K*A^t, you then normalize to U = K * A^t / K_total, where K_total is the sum of all cells in the kernel, and is constant, but is also not equal to the number of cells, as that is smoothLife. In Lenia's case, the maximum activation is the sum of all the kernel cells, which is less than the number of cells (kernel values between 0 and 1)

2. The normalized kernel activation U is then passed into the Growth Function,

Activation = G(U)

3. given some delta t time interval between discrete states, L^(t+dt) = [L^t + Adt], which is finally clamped between 0 and 1. 
