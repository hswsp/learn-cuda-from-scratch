ARCH ?= sm_80
NVCC ?= nvcc
NVCCFLAGS = -O2 -std=c++17 -arch=$(ARCH) -lineinfo --expt-relaxed-constexpr

BINS = hello host_device_memcpy error_handling

.PHONY: all clean run
all: $(BINS)

%: %.cu ../common/cuda_utils.h
	$(NVCC) $(NVCCFLAGS) $< -o $@

run: all
	@echo "--- hello ---";              ./hello
	@echo;
	@echo "--- host_device_memcpy ---"; ./host_device_memcpy
	@echo;
	@echo "--- error_handling ---";     ./error_handling || true

clean:
	rm -f $(BINS) *.o
