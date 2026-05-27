# Chapter 03 — 线程模型与索引

| 文件 | 说明 |
|---|---|
| `thread_id_map.cu` | 打印 (block, thread) → warp/lane/global id 映射 |
| `vec_add.cu` | 1D 向量加：一线程一元素 vs grid-stride loop 对比 |
| `matrix_add.cu` | 2D 索引下的矩阵加，处理非整数倍尺寸 |

```bash
make ARCH=sm_80 run
```

教程: [`docs/ch03-threads/index.html`](../../docs/ch03-threads/index.html)
