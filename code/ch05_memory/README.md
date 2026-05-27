# Chapter 05 — 内存层级

| 文件 | 说明 |
|---|---|
| `coalesce_vs_strided.cu` | stride=1/2/4/8/32 访问的吞吐对比 |
| `mem_modes.cu` | device-only / unified / zero-copy 三种分配的差距 |
| `shared_demo.cu` | 第一次用 `__shared__` 做 block 内 reduction |
| `constant_demo.cu` | `__constant__` 内存广播加速 |

```bash
make ARCH=sm_80 run
make ARCH=sm_80 bench
```

教程: [`docs/ch05-memory/index.html`](../../docs/ch05-memory/index.html)
