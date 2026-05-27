ARCH ?= sm_80
NVCC ?= nvcc
NVCCFLAGS = -O2 -std=c++17 -arch=$(ARCH) -lineinfo

BINS = flash_attn_v1

.PHONY: all clean run bench
all: $(BINS)
%: %.cu ../common/cuda_utils.h
	$(NVCC) $(NVCCFLAGS) $< -o $@

run: all
	@./flash_attn_v1 --T=256
	@./flash_attn_v1 --T=1024
	@./flash_attn_v1 --T=4096
bench: run
clean: ; rm -f $(BINS)
