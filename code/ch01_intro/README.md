# Chapter 01 — 导论与环境

> 配套教程：[`docs/ch01-intro/index.html`](../../docs/ch01-intro/index.html)

## 示例

| 文件 | 说明 |
|---|---|
| `device_query.cu` | 打印 GPU 全部硬件能力（SM 数、显存、Tensor Core 代际等） |
| `bandwidth_estimate.cu` | 测量 H2D / D2H / D2D 内存拷贝带宽，对比 pageable vs pinned |

## 运行

```bash
make ARCH=sm_80 all
./device_query
./bandwidth_estimate --MB=128 --iters=10
```

## 练习

见 `exercises/`：
1. 修改 `device_query.cu`，新增打印每个 SM 上的 CUDA core 数（提示：与 compute capability 相关，要查表）。
2. 测出在你的机器上"H2D pageable vs pinned"的加速比是多少？典型值在 1.5×–4× 之间。
