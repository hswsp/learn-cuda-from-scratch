# 模型权重

## GPT-2 small (124M, fp16 ~250 MB)

```bash
pip install transformers torch
python data/download_gpt2.py --out data/gpt2-small.bin
```

输出文件 `gpt2-small.bin` 的格式见 `download_gpt2.py` 顶部块注释 (16 int32 header + 一连串 fp16 张量)。
Ch14 的 `mini_llm` 程序直接 fread 这个文件。

权重布局对应：HuggingFace `gpt2` 模型，1024 上下文，50257 词表。

## 也可以用更小的：

- `distilgpt2` (~82M)
- `EleutherAI/pythia-70m`
- 自己训练的 tiny GPT
