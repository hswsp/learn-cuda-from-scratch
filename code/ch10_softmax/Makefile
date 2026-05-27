ARCH ?= sm_80
NVCC ?= nvcc
NVCCFLAGS = -O2 -std=c++17 -arch=$(ARCH) -lineinfo

BINS = softmax online_softmax rmsnorm

.PHONY: all clean run
all: $(BINS)
%: %.cu ../common/cuda_utils.h
	$(NVCC) $(NVCCFLAGS) $< -o $@
run: all
	@./softmax; echo; ./online_softmax; echo; ./rmsnorm
clean: ; rm -f $(BINS)
