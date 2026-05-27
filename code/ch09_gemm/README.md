# Chapter 09 — GEMM 深入

| 文件 | 说明 |
|---|---|
| `gemm_register_tile.cu` | 2D 寄存器 tile，128×128 block × 8×8/thread，fp32 |
| `gemm_wmma.cu` | Tensor Core WMMA 16×16×16 fp16，fp32 累加 |
| `gemm_cublas.cu` | cuBLAS sgemm baseline（注意 row/col-major 转换） |

```bash
make ARCH=sm_80 run        # 单一 size 对比
make ARCH=sm_80 bench      # 多 size 性能曲线
```

教程: [`docs/ch09-gemm/index.html`](../../docs/ch09-gemm/index.html)
