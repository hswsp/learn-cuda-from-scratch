# Chapter 07 — Reduction / Scan / Atomics

| 文件 | 说明 |
|---|---|
| `reduce_v1_to_v5.cu` | Mark Harris 经典 5 阶段 reduction 优化 |
| `scan.cu` | block 内 inclusive prefix-sum (warp shuffle) |
| `histogram.cu` | 全局 atomic vs shared-mem 私有 + 合并 |

```bash
make ARCH=sm_80 run
```

教程：[`docs/ch07-reduce/index.html`](../../docs/ch07-reduce/index.html)
