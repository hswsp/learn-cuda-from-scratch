# Chapter 01 — Intro & environment.
ARCH ?= sm_80
NVCC ?= nvcc
NVCCFLAGS = -O2 -std=c++17 -arch=$(ARCH) -lineinfo --expt-relaxed-constexpr

BINS = device_query bandwidth_estimate

.PHONY: all clean run
all: $(BINS)

%: %.cu ../common/cuda_utils.h
	$(NVCC) $(NVCCFLAGS) $< -o $@

run: all
	@echo "--- device_query ---";       ./device_query
	@echo;
	@echo "--- bandwidth_estimate ---"; ./bandwidth_estimate

clean:
	rm -f $(BINS) *.o
