ARCH ?= sm_80
NVCC ?= nvcc
NVCCFLAGS = -O2 -std=c++17 -arch=$(ARCH) -lineinfo

BINS = reduce_v1_to_v5 scan histogram

.PHONY: all clean run
all: $(BINS)
%: %.cu ../common/cuda_utils.h
	$(NVCC) $(NVCCFLAGS) $< -o $@
run: all
	@./reduce_v1_to_v5
	@echo; ./scan
	@echo; ./histogram
clean: ; rm -f $(BINS)
