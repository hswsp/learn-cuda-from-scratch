# 论文清单（按章节匹配）

## 基础架构与编程模型
- *Programming Massively Parallel Processors* (Kirk, Hwu) — 4th ed. 2022. 教材级。
- *Patterns for Parallel Programming* — Mattson et al. — reduce/scan/histogram 模式的鼻祖
- *Optimizing Parallel Reduction in CUDA* — Mark Harris, NVIDIA — 第 7 章五阶段优化的原文

## GEMM / Tensor Core
- *NVIDIA Tensor Core Performance: The Ultimate Guide* — Markidis et al. 2018
- *CUTLASS: Fast Linear Algebra in CUDA C++* — NVIDIA OSS docs
- *Volta GV100 Whitepaper / Ampere A100 Whitepaper / Hopper H100 Whitepaper* — 硬件细节
- *Cooperative Matrix Multiply* (mma PTX, sm_80) — PTX 手册章节

## Attention 与 FlashAttention
- **Attention Is All You Need** — Vaswani et al. 2017 — Transformer 鼻祖
- **FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness** — Tri Dao et al. 2022
- **FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning** — Tri Dao 2023
- **FlashAttention-3** — Shah et al. 2024（Hopper / fp8）

## 推理系统
- **Efficient Memory Management for Large Language Model Serving with PagedAttention** — vLLM, Kwon et al. 2023
- *Orca: A Distributed Serving System for Transformer-Based Generative Models* — Yu et al. OSDI 2022
- **Fast Inference from Transformers via Speculative Decoding** — Leviathan et al. 2023
- *Mixed Precision Training* — Micikevicius et al. 2018 — fp16/bf16 训练
- *FP8 Formats for Deep Learning* — Micikevicius et al. 2022 — Hopper FP8

## 量化
- **GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers** — Frantar et al. 2022
- **AWQ: Activation-aware Weight Quantization for LLM Compression** — Lin et al. 2023
- *SmoothQuant: Accurate and Efficient Post-Training Quantization for LLMs* — Xiao et al. 2022
- *LLM.int8()* — Dettmers et al. 2022

## 模型架构（用于 Capstone 扩展）
- *Language Models are Unsupervised Multitask Learners* — GPT-2, Radford et al. 2019
- *Llama 2 / Llama 3 Technical Report* — Touvron et al.
- *RoFormer: Enhanced Transformer with Rotary Position Embedding* — Su et al. 2021 — RoPE
- *GLU Variants Improve Transformer* — Shazeer 2020 — SwiGLU

## 工具
- *Performance Tuning of Scientific Codes with the Roofline Model* — Williams et al. — Roofline 原文
- *Nsight Compute Profiling Guide* — NVIDIA 官方文档
