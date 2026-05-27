# Chapter 02 — Hello CUDA

| 文件 | 说明 |
|---|---|
| `hello.cu` | 第一个 `__global__` kernel，每个线程打印自己的坐标 |
| `host_device_memcpy.cu` | cudaMalloc + cudaMemcpy + kernel 修改数组 |
| `error_handling.cu` | 演示三种错误如何被 CUDA_CHECK / KERNEL_CHECK 抓住 |

```bash
make ARCH=sm_80 run
```

配套教程: [`docs/ch02-hello/index.html`](../../docs/ch02-hello/index.html)
