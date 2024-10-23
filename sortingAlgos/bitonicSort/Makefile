# Compiler and flags
NVCC = nvcc
CFLAGS = -arch=sm_30 -O3

# Executable name
TARGET = bitonic_sort

# Source files
SRC = bitonic_sort.cu

# Default rule to build the target
all: $(TARGET)

# Rule to compile the CUDA program
$(TARGET): $(SRC)
	$(NVCC) $(CFLAGS) -o $(TARGET) $(SRC)

# Rule to clean up generated files
clean:
	rm -f $(TARGET)

# Rule to run the program
run: $(TARGET)
	./$(TARGET)

