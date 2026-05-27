# Chapter 06 — Shared Memory & Tile

| 文件 | 说明 |
|---|---|
| `matmul_naive.cu` | 一线程一 cell 的朴素 GEMM，memory-bound |
| `matmul_tiled.cu` | 经典 tile 版（BM=BN=BK=32），全局读取量降 32× |
| `transpose.cu` | naive / shared / +padding 三版 transpose，bank conflict 直观演示 |

```bash
make ARCH=sm_80 run
make ARCH=sm_80 bench    # 多种 size 的 GFLOPS 对比
```

教程: [`docs/ch06-tile/index.html`](../../docs/ch06-tile/index.html)
