ARCH ?= sm_80
NVCC ?= nvcc
NVCCFLAGS = -O2 -std=c++17 -arch=$(ARCH) -lineinfo

BINS = rope swiglu kv_cache sampling

.PHONY: all clean run
all: $(BINS)
%: %.cu ../common/cuda_utils.h
	$(NVCC) $(NVCCFLAGS) $< -o $@
run: all
	@for b in $(BINS); do echo "--- $$b ---"; ./$$b; done
clean: ; rm -f $(BINS)
