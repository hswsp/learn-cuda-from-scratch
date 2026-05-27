# Chapter 14 — Capstone: Mini-LLM Inference

> 把前面 13 章写过的所有 kernel 串成一个能生成文本的 GPT-2 small 推理引擎。

## 流程

1. 下载权重（一次性）：
```bash
pip install transformers torch
python ../../data/download_gpt2.py --out ../../data/gpt2-small.bin
```

2. 编译并运行：
```bash
make ARCH=sm_80
./mini_llm --weights=../../data/gpt2-small.bin --tokens=15496,11,612,318 --max_new=10
```

3. token id ↔ 文本：本程序输入输出都是 GPT-2 BPE token id。
   解码/编码用 Python：
```python
from transformers import GPT2Tokenizer
tok = GPT2Tokenizer.from_pretrained("gpt2")
print(tok.encode("Hello, there is"))     # -> [15496, 11, 612, 318]
print(tok.decode([15496, 11, 612, 318, 1043, 257, 1810, 2950, 287]))
```

## 简化说明

为了把代码控制在单个 .cu 文件、专注教学：
- 全程 fp32（生产用 fp16+TC，第 9/12 章模板可移植过来）
- 朴素 attention，非 FlashAttention（O(T²) 显存，但实现简单）
- 无 KV cache（每步重算 prompt+answer 全前向 → 慢，O(T³) total，但代码短）
- 只支持 batch=1

## 改进路径（练习题）

1. **加 KV cache**：把每层的 K, V 缓存到 `(n_layer, n_head, T_max, D_head)`，每步只 forward 1 token，从 O(T²) 单步成本降到 O(T)。
2. **fp16 + Tensor Core**：把 fc/proj/qkv 用 Ch9 的 WMMA 替换。
3. **FlashAttention 替换朴素 attention**：直接用 Ch12 的 kernel。
4. **Triton 端到端**：用 Triton 重写整个 forward，对比代码量。
5. **接入 sentencepiece tokenizer**：让程序接受 prompt 字符串。
