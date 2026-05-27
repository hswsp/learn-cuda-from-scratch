# Chapter 13 — LLM 必备零件

| 文件 | 说明 |
|---|---|
| `rope.cu` | Rotary Position Embedding（Llama、Mistral 用） |
| `swiglu.cu` | SiLU/SwiGLU 激活，Llama 风格 FFN 的 gate × up |
| `kv_cache.cu` | decode 时把新 (K, V) 追加到预分配 cache |
| `sampling.cu` | greedy argmax（top-k / top-p 在练习题） |

```bash
make ARCH=sm_80 run
```

教程: [`docs/ch13-llm-parts/index.html`](../../docs/ch13-llm-parts/index.html)
