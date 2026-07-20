ARCH ?= sm_80
NVCC ?= nvcc
NVCCFLAGS = -O2 -std=c++17 -arch=$(ARCH) -lineinfo --expt-relaxed-constexpr

BINS = mini_llm mini_llm_adv1 mini_llm_adv2

.PHONY: all clean run run_adv1 run_adv2
all: $(BINS)

mini_llm: mini_llm.cu ../common/cuda_utils.h
	$(NVCC) $(NVCCFLAGS) $< -o $@

mini_llm_adv1: mini_llm_adv1.cu ../common/cuda_utils.h
	$(NVCC) $(NVCCFLAGS) $< -o $@

mini_llm_adv2: mini_llm_adv2.cu ../common/cuda_utils.h
	$(NVCC) $(NVCCFLAGS) $< -o $@

run: all
	@./mini_llm --weights=../../data/gpt2-small.bin --tokens=15496,11,612,318 --max_new=10

run_adv1: mini_llm_adv1
	@./mini_llm_adv1 --weights=../../data/gpt2-small.bin --tokens=15496,11,612,318 --max_new=64 --temperature=0.8 --top_k=40 --seed=42

run_adv2: mini_llm_adv2
	@./mini_llm_adv2 --weights=../../data/gpt2-small.bin --tokens=15496,11,612,318 --max_new=64 --temperature=0.8 --top_k=40 --seed=42

clean: ; rm -f $(BINS)
