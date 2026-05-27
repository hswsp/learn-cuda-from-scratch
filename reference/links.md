# 外部链接

## 官方文档
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/) — 看 mma.sync / cp.async 必读
- [Nsight Compute User Guide](https://docs.nvidia.com/nsight-compute/NsightCompute/index.html)
- [cuBLAS / cuDNN / NCCL / CUTLASS](https://developer.nvidia.com/gpu-accelerated-libraries)

## 教程与博客
- [CUDA Hands-on Tutorials (NVIDIA)](https://developer.nvidia.com/blog/even-easier-introduction-cuda/)
- [Simon Boehm: How to Optimize a CUDA Matmul Kernel](https://siboehm.com/articles/22/CUDA-MMM) — 从 naive 到接近 cuBLAS
- [Lei Mao: CUDA Notes](https://leimao.github.io/) — 大量精炼小文
- [Tri Dao: FlashAttention talks](https://github.com/Dao-AILab/flash-attention) — 论文 + slides
- [Horace He: Making Deep Learning Go Brrrr](https://horace.io/brrr_intro.html)

## 开源仓库（按学习价值）
- [llama.cpp](https://github.com/ggerganov/llama.cpp) — 极致 CPU/ARM + GGUF 量化；CUDA 实现也精简
- [vLLM](https://github.com/vllm-project/vllm) — PagedAttention、continuous batching、Triton kernels
- [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) — NVIDIA 官方产线，CUTLASS 重度用户
- [FlashAttention](https://github.com/Dao-AILab/flash-attention) — fa v1/v2/v3 源码
- [CUTLASS](https://github.com/NVIDIA/cutlass) — GEMM 模板库；CuTe DSL
- [Triton](https://github.com/triton-lang/triton) — OpenAI 的 GPU DSL
- [llm.c](https://github.com/karpathy/llm.c) — Karpathy 的纯 C/CUDA GPT-2 训练/推理（本仓库 Ch14 的灵感来源）
- [picoGPT](https://github.com/jaymody/picoGPT) — 60 行 Python 写 GPT-2，便于对照

## 课程
- [CMU 15-418: Parallel Computer Architecture and Programming](https://www.cs.cmu.edu/~418/)
- [Stanford CS149: Parallel Computing](https://gfxcourses.stanford.edu/cs149)
- [UIUC ECE408: Applied Parallel Programming](http://ece408.web.engr.illinois.edu/)
- [Modal: GPU MODE](https://github.com/cuda-mode/lectures) — 极活跃的 GPU 编程社区 + lectures

## 中文资源
- [《CUDA C 编程权威指南》](https://book.douban.com/subject/27006429/) — 中文教材，覆盖到 sm_30 时代，但基础概念扎实
- [BBuf 的 GiantPandaCV 公众号](https://github.com/BBuf/how-to-optim-algorithm-in-cuda) — 大量 CUDA 优化技巧整理

## 在线工具
- [Godbolt Compiler Explorer (CUDA)](https://godbolt.org) — 看 nvcc 输出的 PTX/SASS
- [Triton Playground](https://triton-lang.org) — 在浏览器写 Triton kernel
