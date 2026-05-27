FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    python3 python3-pip \
    vim less \
    cuda-nsight-compute-12-4 cuda-nsight-systems-12-4 \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir numpy torch --index-url https://download.pytorch.org/whl/cpu
# torch CPU build is enough for reference checks; on GPU host use cu124 wheel.

WORKDIR /work
CMD ["bash"]
