include ../../common/make.config

CC := $(CUDA_DIR)/bin/nvcc -O3 -arch=sm_32
INCLUDE := $(CUDA_DIR)/include

SRC = pathfinder.cu
UMASRC = pathfinderUMA.cu

UMAEXE = UMApathfinder
EXE = pathfinder

release:
	$(CC) $(SRC) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR) 

UMA:
	$(CC) $(UMASRC) -o $(UMAEXE) -DUMA -I$(INCLUDE) -L$(CUDA_LIB_DIR) 

enum:
	$(CC) -deviceemu $(SRC) -o $(EXE) -I$(INCLUDE) -L$$(CUDA_LIB_DIR) 

debug:
	$(CC) -g $(SRC) -o $(EXE) -I$(INCLUDE) -L$$(CUDA_LIB_DIR) 

debugenum:
	$(CC) -g -deviceemu $(SRC) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR) 

clean:
	rm -f pathfinder
