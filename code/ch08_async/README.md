# Chapter 08 — 性能分析与异步

| 文件 | 说明 |
|---|---|
| `multi_stream.cu` | 用 4 个 stream 让 H2D / 计算 / D2H 三阶段重叠 |
| `cuda_graph_demo.cu` | 1000 次 launch vs 1 次 CUDA Graph 重放 |

```bash
make ARCH=sm_80 run
# 用 Nsight Systems 看 timeline
nsys profile --stats=true ./multi_stream
# 用 Nsight Compute 看 kernel 详情
ncu --set full ./cuda_graph_demo
```

教程: [`docs/ch08-async/index.html`](../../docs/ch08-async/index.html)
