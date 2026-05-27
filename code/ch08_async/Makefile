ARCH ?= sm_80
NVCC ?= nvcc
NVCCFLAGS = -O2 -std=c++17 -arch=$(ARCH) -lineinfo

BINS = multi_stream cuda_graph_demo

.PHONY: all clean run
all: $(BINS)
%: %.cu ../common/cuda_utils.h
	$(NVCC) $(NVCCFLAGS) $< -o $@
run: all
	@./multi_stream; echo; ./cuda_graph_demo
clean: ; rm -f $(BINS)
