# Chapter 14 — Capstone: Mini-LLM Inference

> 把前面 13 章写过的所有 kernel 串成一个能生成文本的 GPT-2 small 推理引擎。

## 流程

1. 下载权重（一次性）：
```bash
pip install transformers torch
python ../../data/download_gpt2.py --out ../../data/gpt2-small.bin
```

2. 编译并运行：
```bash
make ARCH=sm_80
./mini_llm --weights=../../data/gpt2-small.bin --tokens=15496,11,612,318 --max_new=10
```

3. token id ↔ 文本：本程序输入输出都是 GPT-2 BPE token id。
   解码/编码用 Python：
```python
from transformers import GPT2Tokenizer
tok = GPT2Tokenizer.from_pretrained("gpt2")
print(tok.encode("Hello, there is"))     # -> [15496, 11, 612, 318]
print(tok.decode([15496, 11, 612, 318, 1043, 257, 1810, 2950, 287]))
```

## 进阶版 1: `mini_llm_adv1.cu`

在基础版之上实现两个常见升级：

1. **支持更长序列**：
   - 一次性预分配最大序列长度 (`max_seq`) 的 Workspace，生成过程中不再反复 `cudaMalloc`。
   - MHA 直接从 `(T, 3*D)` 的 `qkv` 缓冲区读取，省去每 token 3 次 `cudaMemcpy`。
   - 可测试 `--max_new=64`（总长度轻松超过 16）。

2. **`temperature + top-k` 采样**：
   - 默认 `temperature=0.8`、`top_k=40`，替代 greedy argmax。
   - 采样逻辑在 CPU 完成（单 token 拷贝开销远小于 forward，且方便调试策略）。
   - 可用 `--seed` 复现结果。

```bash
make mini_llm_adv1
./mini_llm_adv1 --weights=../../data/gpt2-small.bin \\
                --tokens=15496,11,612,318 \\
                --max_new=64 --temperature=0.8 --top_k=40 --seed=42
```

## 进阶版 2: `mini_llm_adv2.cu`

在 v1 基础上再做三个生产级升级：

1. **KV cache**：每层维护 `K_cache/V_cache`（形状 `n_layer × n_head × T_max × D_head`）。
   - prefill 阶段一次性处理全部 prompt 并填充 cache；
   - decode 阶段每步只 forward 1 个 token，attention 从 `O(T²)` 降到 `O(T)`。
2. **fp16 + WMMA**：`fc/proj/qkv` 的权重以 `__half` 存储，GEMM 走 Tensor Core；
   - 激活值保持 fp32，GEMM 前实时 `fp32→fp16` 转换；
   - `M` 不足 16 时自动 pad 到 16 的倍数（decode 的 `M=1` 也能走 WMMA）；
   - 仅当 `N/K` 未 16 对齐时才回退到简单 GEMM。
3. **FlashAttention**：prefill 阶段用 Ch12 的 `flash_attn_v1` 替换朴素 attention，
   不物化 `T×T` 中间矩阵；decode 阶段使用单 query 的 KV-cache attention。

预期性能（A100 fp16, prompt=4）：单步约 5 ms（v1 约 50 ms）。

```bash
make mini_llm_adv2
./mini_llm_adv2 --weights=../../data/gpt2-small.bin \\
                --tokens=15496,11,612,318 \\
                --max_new=64 --temperature=0.8 --top_k=40 --seed=42
```

注意：`wmma` 需要 sm_70+，Makefile 默认 `ARCH=sm_80`。

## 简化说明

为了把代码控制在单个 .cu 文件、专注教学：
- 全程 fp32（生产用 fp16+TC，第 9/12 章模板可移植过来）
- 朴素 attention，非 FlashAttention（O(T²) 显存，但实现简单）
- 无 KV cache（每步重算 prompt+answer 全前向 → 慢，O(T³) total，但代码短）
- 只支持 batch=1

## 改进路径（练习题）

1. **加 KV cache**：把每层的 K, V 缓存到 `(n_layer, n_head, T_max, D_head)`，每步只 forward 1 token，从 O(T²) 单步成本降到 O(T)。
2. **fp16 + Tensor Core**：把 fc/proj/qkv 用 Ch9 的 WMMA 替换。
3. **FlashAttention 替换朴素 attention**：直接用 Ch12 的 kernel。
4. **Triton 端到端**：用 Triton 重写整个 forward，对比代码量。
5. **接入 sentencepiece tokenizer**：让程序接受 prompt 字符串。
