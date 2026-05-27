# Top-level Makefile for CUDA + LLM tutorial.
# Recurses into each chapter. Override ARCH on the command line:
#   make ARCH=sm_75 all     # T4 (Colab free tier)
#   make ARCH=sm_80 all     # A100, also OK on 3090
#   make ARCH=sm_89 all     # 4090
#   make ARCH=sm_90a all    # H100

ARCH ?= sm_80
export ARCH

CHAPTERS := \
	ch01_intro ch02_hello ch03_threads ch04_arch \
	ch05_memory ch06_tile ch07_reduce ch08_async \
	ch09_gemm ch10_softmax ch11_attention ch12_flashattn \
	ch13_llm_parts ch14_mini_llm

.PHONY: all clean run bench help $(CHAPTERS)

help:
	@echo "Targets:"
	@echo "  make all                   build every chapter (ARCH=$(ARCH))"
	@echo "  make clean                 clean every chapter"
	@echo "  make run                   build + execute every chapter's demos"
	@echo "  make bench                 run benchmarks (core chapters only)"
	@echo "  make ch09_gemm             build only ch09"
	@echo "  make ARCH=sm_75 all        target T4"
	@echo ""
	@echo "Per-chapter Makefiles live in code/<chapter>/Makefile"

all:
	@for c in $(CHAPTERS); do \
		echo "==> $$c"; \
		$(MAKE) -C code/$$c ARCH=$(ARCH) all || exit 1; \
	done

clean:
	@for c in $(CHAPTERS); do \
		$(MAKE) -C code/$$c clean; \
	done

run:
	@for c in $(CHAPTERS); do \
		echo "==> running $$c"; \
		$(MAKE) -C code/$$c ARCH=$(ARCH) run || exit 1; \
	done

bench:
	@for c in ch05_memory ch06_tile ch09_gemm ch12_flashattn ch14_mini_llm; do \
		echo "==> benching $$c"; \
		$(MAKE) -C code/$$c ARCH=$(ARCH) bench; \
	done

$(CHAPTERS):
	$(MAKE) -C code/$@ ARCH=$(ARCH) all
