ARCH ?= sm_80
NVCC ?= nvcc
NVCCFLAGS = -O2 -std=c++17 -arch=$(ARCH) -lineinfo --expt-relaxed-constexpr

BINS = mini_llm

.PHONY: all clean run
all: $(BINS)
mini_llm: mini_llm.cu ../common/cuda_utils.h
	$(NVCC) $(NVCCFLAGS) $< -o $@

run: all
	@./mini_llm --weights=../../data/gpt2-small.bin --tokens=15496,11,612,318 --max_new=10
clean: ; rm -f $(BINS)
