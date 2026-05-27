ARCH ?= sm_80
NVCC ?= nvcc
NVCCFLAGS = -O2 -std=c++17 -arch=$(ARCH) -lineinfo --expt-relaxed-constexpr

BINS = matmul_naive matmul_tiled transpose

.PHONY: all clean run bench
all: $(BINS)
%: %.cu ../common/cuda_utils.h
	$(NVCC) $(NVCCFLAGS) $< -o $@

run: all
	@./matmul_naive --M=512 --N=512 --K=512
	@./matmul_tiled --M=512 --N=512 --K=512
	@./transpose

bench: all
	@for K in 128 512 1024 2048; do \
	  ./matmul_naive --M=$$K --N=$$K --K=$$K; \
	  ./matmul_tiled --M=$$K --N=$$K --K=$$K; \
	done

clean: ; rm -f $(BINS)
