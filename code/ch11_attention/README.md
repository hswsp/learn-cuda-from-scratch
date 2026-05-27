# Chapter 11 — Attention 入门

| 文件 | 说明 |
|---|---|
| `fused_qkv.cu` | 单次 GEMM 同时算出 Q/K/V，再 split |
| `attention_naive.cu` | 三阶段 (QKᵀ → softmax → ·V) 朴素实现，物化 T×T 中间矩阵 |

```bash
make ARCH=sm_80 run
```

教程: [`docs/ch11-attention/index.html`](../../docs/ch11-attention/index.html)
