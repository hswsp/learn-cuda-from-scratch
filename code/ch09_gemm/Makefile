ARCH ?= sm_80
NVCC ?= nvcc
NVCCFLAGS = -O2 -std=c++17 -arch=$(ARCH) -lineinfo --expt-relaxed-constexpr

BINS = gemm_register_tile gemm_wmma gemm_cublas

.PHONY: all clean run bench
all: $(BINS)

gemm_register_tile: gemm_register_tile.cu ../common/cuda_utils.h
	$(NVCC) $(NVCCFLAGS) $< -o $@
gemm_wmma: gemm_wmma.cu ../common/cuda_utils.h
	$(NVCC) $(NVCCFLAGS) $< -o $@
gemm_cublas: gemm_cublas.cu ../common/cuda_utils.h
	$(NVCC) $(NVCCFLAGS) $< -o $@ -lcublas

run: all
	@./gemm_register_tile --M=1024 --N=1024 --K=1024
	@./gemm_wmma         --M=1024 --N=1024 --K=1024
	@./gemm_cublas       --M=1024 --N=1024 --K=1024

bench: all
	@for S in 512 1024 2048 4096; do \
	  ./gemm_register_tile --M=$$S --N=$$S --K=$$S; \
	  ./gemm_wmma         --M=$$S --N=$$S --K=$$S; \
	  ./gemm_cublas       --M=$$S --N=$$S --K=$$S; \
	done

clean: ; rm -f $(BINS)
