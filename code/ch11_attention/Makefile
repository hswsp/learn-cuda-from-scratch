ARCH ?= sm_80
NVCC ?= nvcc
NVCCFLAGS = -O2 -std=c++17 -arch=$(ARCH) -lineinfo --expt-relaxed-constexpr

BINS = attention_naive fused_qkv

.PHONY: all clean run
all: $(BINS)
%: %.cu ../common/cuda_utils.h
	$(NVCC) $(NVCCFLAGS) $< -o $@
run: all
	@./fused_qkv; echo; ./attention_naive --T=128 --D=64
clean: ; rm -f $(BINS)
