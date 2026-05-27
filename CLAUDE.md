# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repository is

A from-scratch **CUDA + LLM inference tutorial** for Chinese-speaking learners. Pairs HTML tutorials (`docs/`) with runnable CUDA C++ code (`code/`). Final deliverable is a hand-written GPT-2 small inference engine.

**Target reader**: solid C/C++ background, zero GPU experience. Tutorial is in 中文; code identifiers and inline comments stay in English so they read naturally with NVIDIA/CUTLASS docs.

## Repo layout (load-bearing conventions)

- `code/common/` — shared C++ headers (`cuda_utils.h`, `cpu_ref.h`, `check.h`, `tensor.h`). Every chapter `#include "../common/..."`. Touch carefully — a change here can break every example.
- `code/chNN_xxx/` — one directory per chapter. Each has its own `Makefile`, 2–5 `.cu` examples, an `exercises/` subdir (`_starter.cu` + `_solution.cu` pairs), an optional `bench/` subdir, and a `README.md` that indexes the examples.
- `docs/chNN-xxx/index.html` — chapter tutorials. Hyphen-separated dir names (not underscore) so URLs look clean. Each page imports `../assets/style.css` + `../assets/tutorial.js` + Prism + Mermaid via CDN.
- `docs/_template/chapter.html` — copy this when creating a new chapter page; the placeholders `<!-- CH_TITLE -->`, `<!-- CH_PREV -->`, `<!-- CH_NEXT -->`, `<!-- CH_BODY -->` are search-and-replace targets.
- `docs/index.html` — landing page with the 14-chapter roadmap. Must be updated whenever a chapter status changes from 🚧 to ✅.
- `scripts/` — `docker/Dockerfile` (nvidia/cuda:12.4), `setup_colab.ipynb` (Colab template), `run_chapter.sh`, `bench.sh`.
- `data/download_gpt2.py` — produces a raw fp16 weight blob from HuggingFace `gpt2`. Used only by Ch14.
- `reference/papers.md` + `links.md` — paper/link registry, append-only.

## Build & run

The host running Claude Code is **macOS arm64 with no NVIDIA GPU**. Do not attempt to invoke `nvcc` locally — it does not exist here. Code is authored to compile/run on Linux + NVIDIA GPU (Colab T4/A100, on-prem 3090/4090).

```bash
# Per-chapter build (target a specific architecture)
cd code/chNN_xxx
make ARCH=sm_80              # default sm_80 (A100, also works on T4 sm_75/4090 sm_89)
make run                      # build + execute all examples
make bench                    # core chapters only (5, 6, 9, 12, 14)
make clean

# Full repo
make -C code all
make -C code clean

# Local docs preview (works on macOS)
python3 -m http.server -d docs 8000

# CPU-only sanity check of reference implementations (works on macOS)
clang++ -O2 -std=c++17 -DCPU_ONLY code/common/cpu_ref_demo.cpp -o /tmp/ref && /tmp/ref
```

Single-test pattern: each `.cu` file builds to a same-named executable. To run only one:

```bash
cd code/ch06_tile
make tiled_matmul && ./tiled_matmul --M=1024 --N=1024 --K=1024
```

## Authoring conventions

### CUDA code
- All kernels include a top-of-file block comment with: 学习目标 / 参数 / 预期输出 / 对应 HTML 锚点（`docs/chNN-xxx/index.html#section`）。
- Wrap every CUDA API call with `CUDA_CHECK(...)` from `common/cuda_utils.h`. After every kernel launch call `KERNEL_CHECK()`.
- Default kernel API style: row-major, fp32 unless the example is specifically about fp16/bf16/fp8 (chapters 9, 12, 14 use fp16 a lot).
- Every non-trivial kernel must have a CPU reference (in `common/cpu_ref.h`) and an `allclose` check (in `common/check.h`).
- Use `GpuTimer` from `cuda_utils.h` for timing — never raw `cudaEventCreate` boilerplate in examples.
- Default block size: 256 unless a comment justifies otherwise. Default tile size for matmul: 32×32.

### Chapter HTML
- Copy `docs/_template/chapter.html`, fill in placeholders.
- Page structure must follow: 学习目标 → 前置知识 → 核心概念 → 关键代码 → 运行结果 → 性能数据（核心章节）→ 自检清单 → 练习题 → 常见坑 → 下一章导览.
- Embed code via `<pre><code class="language-cuda">...</code></pre>` (Prism handles `cuda`, `cpp`, `py`, `bash`).
- Architecture diagrams: prefer Mermaid (`<pre class="mermaid">graph TD; ...</pre>`). Use raw SVG only for warp-scheduling animations (Ch 4, 8).
- Cross-reference code with relative links: `<a href="../../code/chNN_xxx/file.cu">file.cu</a>`.

### Glossary discipline
When a Chinese term has a standard English form, mark first occurrence as `<span class="term" data-en="warp">线程束 (warp)</span>`. The `tutorial.js` script renders a tooltip on hover. Add new terms to `docs/glossary.html`.

## Verification before declaring a chapter done

A chapter is "done" when:
1. All `.cu` examples have block comments + `CUDA_CHECK` + CPU `allclose` check.
2. `code/chNN_xxx/README.md` lists every file with one-line description.
3. `docs/chNN-xxx/index.html` has all 10 standard sections + working prev/next links.
4. `docs/index.html` chapter card status flipped from 🚧 to ✅.
5. Top-level `README.md` table entry updated if needed.

For chapters whose verification truly requires a GPU (e.g. WMMA output check), add `⚠️ 需 GPU 验证` banner at top of the HTML and leave numeric performance results as `TODO(on GPU)` rather than fabricating.

## What NOT to do

- Don't introduce CMake or other build systems — Makefiles are intentionally simple for learners to read.
- Don't add JS frameworks (React/Vue/Svelte) to `docs/`. Vanilla HTML + Prism + Mermaid only.
- Don't depend on external CUDA libraries beyond cuBLAS/cuDNN/CUTLASS (and only as comparison baselines, never as the main implementation — the whole point is to write the kernels yourself).
- Don't write kernels that only compile with one specific SM version unless the chapter is explicitly about that hardware feature (WMMA → sm_70+, `cp.async` → sm_80+, FP8 → sm_89/90).
- Don't translate code comments to Chinese — keep them English for searchability against official docs.
