# Chapter 04 — GPU 硬件架构

| 文件 | 说明 |
|---|---|
| `warp_divergence.cu` | warp-uniform vs warp-divergent 分支的耗时对比 |
| `occupancy_probe.cu` | 调用 occupancy API 看 block / shared mem / 寄存器约束 |

```bash
make ARCH=sm_80 run
```

教程：[`docs/ch04-arch/index.html`](../../docs/ch04-arch/index.html)
