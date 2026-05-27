#!/usr/bin/env python3
"""
Download GPT-2 small weights from HuggingFace and dump them as a raw fp16
binary blob that the Ch14 C++ inference engine can mmap / fread.

Layout (little-endian, fp16 except header):
    int32 magic        = 0x47505432   ("GPT2")
    int32 version      = 1
    int32 n_layer      = 12
    int32 n_head       = 12
    int32 d_model      = 768
    int32 d_ff         = 3072
    int32 max_seq      = 1024
    int32 vocab_size   = 50257
    int32 padding[8]   = 0
    fp16  wte          [vocab_size, d_model]
    fp16  wpe          [max_seq,    d_model]
    for layer in 0..n_layer-1:
        fp16 ln1_w     [d_model]
        fp16 ln1_b     [d_model]
        fp16 c_attn_w  [d_model, 3*d_model]
        fp16 c_attn_b  [3*d_model]
        fp16 c_proj_w  [d_model, d_model]
        fp16 c_proj_b  [d_model]
        fp16 ln2_w     [d_model]
        fp16 ln2_b     [d_model]
        fp16 mlp_fc_w  [d_model, d_ff]
        fp16 mlp_fc_b  [d_ff]
        fp16 mlp_proj_w[d_ff,    d_model]
        fp16 mlp_proj_b[d_model]
    fp16 ln_f_w        [d_model]
    fp16 ln_f_b        [d_model]
    (output head shares wte)

Usage:
    pip install transformers torch
    python data/download_gpt2.py --out data/gpt2-small.bin
"""
import argparse
import struct
import numpy as np


def write_tensor(f, t):
    arr = t.detach().cpu().to(dtype=__import__("torch").float16).numpy()
    f.write(arr.tobytes(order="C"))


def main():
    import torch
    from transformers import GPT2LMHeadModel

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2", help="HF model id")
    ap.add_argument("--out",   default="data/gpt2-small.bin")
    args = ap.parse_args()

    print(f"[gpt2-dl] loading {args.model} ...")
    m = GPT2LMHeadModel.from_pretrained(args.model).eval()
    cfg = m.config
    print(f"[gpt2-dl] n_layer={cfg.n_layer} n_head={cfg.n_head} "
          f"d={cfg.n_embd} vocab={cfg.vocab_size} max_seq={cfg.n_positions}")

    with open(args.out, "wb") as f:
        # header (16 int32 = 64 bytes)
        f.write(struct.pack("<i", 0x47505432))
        f.write(struct.pack("<i", 1))
        f.write(struct.pack("<i", cfg.n_layer))
        f.write(struct.pack("<i", cfg.n_head))
        f.write(struct.pack("<i", cfg.n_embd))
        f.write(struct.pack("<i", 4 * cfg.n_embd))
        f.write(struct.pack("<i", cfg.n_positions))
        f.write(struct.pack("<i", cfg.vocab_size))
        f.write(b"\x00" * (8 * 4))   # padding

        write_tensor(f, m.transformer.wte.weight)
        write_tensor(f, m.transformer.wpe.weight)
        for blk in m.transformer.h:
            write_tensor(f, blk.ln_1.weight); write_tensor(f, blk.ln_1.bias)
            # HF GPT-2 c_attn is Conv1D (out_features=3*d, in_features=d, weight (d, 3d))
            write_tensor(f, blk.attn.c_attn.weight); write_tensor(f, blk.attn.c_attn.bias)
            write_tensor(f, blk.attn.c_proj.weight); write_tensor(f, blk.attn.c_proj.bias)
            write_tensor(f, blk.ln_2.weight); write_tensor(f, blk.ln_2.bias)
            write_tensor(f, blk.mlp.c_fc.weight);   write_tensor(f, blk.mlp.c_fc.bias)
            write_tensor(f, blk.mlp.c_proj.weight); write_tensor(f, blk.mlp.c_proj.bias)
        write_tensor(f, m.transformer.ln_f.weight); write_tensor(f, m.transformer.ln_f.bias)

    import os
    print(f"[gpt2-dl] wrote {args.out}  ({os.path.getsize(args.out)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
