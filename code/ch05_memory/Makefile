ARCH ?= sm_80
NVCC ?= nvcc
NVCCFLAGS = -O2 -std=c++17 -arch=$(ARCH) -lineinfo --expt-relaxed-constexpr

BINS = coalesce_vs_strided mem_modes shared_demo constant_demo

.PHONY: all clean run bench
all: $(BINS)
%: %.cu ../common/cuda_utils.h
	$(NVCC) $(NVCCFLAGS) $< -o $@

run: all
	@for b in $(BINS); do echo "--- $$b ---"; ./$$b; echo; done

bench: all
	@./coalesce_vs_strided
	@./mem_modes

clean: ; rm -f $(BINS)
