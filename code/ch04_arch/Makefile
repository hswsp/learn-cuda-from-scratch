ARCH ?= sm_80
NVCC ?= nvcc
NVCCFLAGS = -O2 -std=c++17 -arch=$(ARCH) -lineinfo

BINS = warp_divergence occupancy_probe

.PHONY: all clean run
all: $(BINS)
%: %.cu ../common/cuda_utils.h
	$(NVCC) $(NVCCFLAGS) $< -o $@

run: all
	@echo "--- warp_divergence ---"; ./warp_divergence
	@echo; echo "--- occupancy_probe ---"; ./occupancy_probe

clean: ; rm -f $(BINS)
