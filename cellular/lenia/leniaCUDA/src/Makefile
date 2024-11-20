# Makefile for bitonic CUDA program
# csc596 (231) fall 2022-2023

L = /usr/local/cuda/bin
GIT_NAME = crilillo14

# Program name
P = bitonic

# Additional flags
VERBOSE = -DIS_VERBOSE
N_ELEM = -DN_ELEM=1000  # Define number of elements (you can adjust this)

# Test target
test:
        $(MAKE) clean
        $(MAKE) $(P)
        $(MAKE) test-bitonic

# Rule to compile bitonic.cu
$(P): $(P).cu
        $(L)/nvcc $(N_ELEM) $(VERBOSE) -lm -Xcompiler -fPIC -o $(P) $(P).cu

# Rule to run the program
test-bitonic: $(P)
        ./$(P)

# Clean up compiled files
clean:
        -rm $(P)