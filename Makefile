# Adapted from Anil Shanbhag's Makefile at https://github.com/anilshanbhag/crystal

CUDA_PATH       ?= /usr/local/cuda
CUDA_INC_PATH   ?= $(CUDA_PATH)/include
CUDA_BIN_PATH   ?= $(CUDA_PATH)/bin

NVCC = nvcc

# For RTX 3090
SM_TARGETS   = -gencode=arch=compute_86,code=\"sm_86,compute_86\" 
SM_DEF     = -DSM860

# For A100
# SM_TARGETS   = -gencode=arch=compute_80,code=\"sm_80,compute_80\" 
# SM_DEF     = -DSM800

NVCCFLAGS += --std=c++17 --expt-relaxed-constexpr --expt-extended-lambda --extended-lambda $(SM_DEF) -Xptxas="-v" -lineinfo -Xcudafe -\# 

SRC = src
BIN = bin
BUILD_DIR = obj
CU_SRC := $(shell find $(SRC) -name "*.cu")
OBJ = $(CU_SRC:src/%.cu=$(BUILD_DIR)/%.o)
DEP = $(OBJ:%.o=%.d)

CUB_DIR = cub/

INCLUDES = -I$(CUB_DIR) -I.

$(BIN)/%: $(BUILD_DIR)/%.o
	$(NVCC) $(SM_TARGETS) -Xlinker -lgomp $< -o $@

-include $(DEP)

$(BUILD_DIR)/%.o: $(SRC)/%.cu
	$(NVCC) -Xcompiler -fopenmp -MMD $(SM_TARGETS) $(NVCCFLAGS) $(CPU_ARCH) $(INCLUDES) $(LIBS) -O3 --compile $< -o $@

setup:
	mkdir -p bin/volcano obj/volcano

clean:
	rm -rf bin/* obj/*
