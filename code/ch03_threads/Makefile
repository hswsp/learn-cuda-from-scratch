ARCH ?= sm_80
NVCC ?= nvcc
NVCCFLAGS = -O2 -std=c++17 -arch=$(ARCH) -lineinfo --expt-relaxed-constexpr

BINS = vec_add matrix_add thread_id_map

.PHONY: all clean run
all: $(BINS)
%: %.cu ../common/cuda_utils.h ../common/cpu_ref.h ../common/check.h
	$(NVCC) $(NVCCFLAGS) $< -o $@

run: all
	@echo "--- thread_id_map ---"; ./thread_id_map | head -40
	@echo; echo "--- vec_add ---"; ./vec_add
	@echo; echo "--- matrix_add ---"; ./matrix_add

clean: ; rm -f $(BINS)
