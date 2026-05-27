# Chapter 12 — FlashAttention

| 文件 | 说明 |
|---|---|
| `flash_attn_v1.cu` | 教学版 single-head FA v1，fp32，tile + online softmax 融合 |

```bash
make ARCH=sm_80 run
```

不物化 T×T 中间矩阵；HBM 流量 O(T·D)（朴素是 O(T²)）。

教程: [`docs/ch12-flashattn/index.html`](../../docs/ch12-flashattn/index.html)
