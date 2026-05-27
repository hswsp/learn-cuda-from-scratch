# Chapter 10 — Softmax & Norm

| 文件 | 说明 |
|---|---|
| `softmax.cu` | 三阶段数值稳定 softmax，每 row 一个 block |
| `online_softmax.cu` | 单遍合并 (m, l) — FlashAttention 的基础 |
| `rmsnorm.cu` | Llama / GPT-NeoX 用的 RMSNorm |

```bash
make ARCH=sm_80 run
```

教程: [`docs/ch10-softmax/index.html`](../../docs/ch10-softmax/index.html)
