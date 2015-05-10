include ../../common/make.config

CC := $(CUDA_DIR)/bin/nvcc -O3 -use_fast_math -arch=sm_32

INCLUDE := $(CUDA_DIR)/include

SRC = gaussian.cu
EXE = gaussian

release: $(SRC)
	$(CC) $(KERNEL_DIM) $(SRC) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR) 

UMA: gaussianUMA.cu
	$(CC) $(KERNEL_DIM) gaussianUMA.cu -o UMAgaussian -I$(INCLUDE) -L$(CUDA_LIB_DIR) 

clean:
	rm gaussian
