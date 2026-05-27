# CUDA × LLM 推理 实战教程

> 从零开始学习 CUDA GPU 编程，一步步手撸大模型推理引擎。
> Hand-written CUDA kernels, from "Hello, GPU" to a working GPT-2 inference engine.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![CUDA: 11.8+](https://img.shields.io/badge/CUDA-11.8%2B-green)](#)
[![Tutorial: 14 chapters](https://img.shields.io/badge/Chapters-14-orange)](docs/index.html)

---

## 这个项目是什么？

一个**完全开源、零基础友好**的 CUDA + LLM 推理学习仓库，包含：

- 📖 **14 章中文 HTML 教程**（`docs/`）—— 逐章讲解，配 Mermaid 架构图与可交互导航
- 💻 **配套可运行 CUDA C++ 代码**（`code/`）—— 每章 2–5 个示例 + 3–5 道练习题
- 🧪 **CPU reference 实现** —— 不依赖 GPU 也能验证算子正确性
- 🚀 **Google Colab 一键运行模板** —— 没有 NVIDIA 显卡也能学
- 🐳 **Docker 镜像** —— Linux 用户一行命令复现

**最终成果**：能在自己写的 kernel 上跑通 GPT-2 small 的端到端推理。

---

## 学习路径（14 章）

| 阶段 | 章节 | 主题 | 关键产出 |
|---|---|---|---|
| **A · 基础** | 1 | 导论与环境 | `deviceQuery` |
|   | 2 | Hello CUDA | 第一个 kernel |
|   | 3 | 线程模型与索引 | 向量/矩阵加 |
|   | 4 | GPU 硬件架构 | SM/warp 可视化 |
| **B · 核心** | 5 | 内存层级 | 带宽实测 |
|   | 6 | Shared Memory + Tile | Tiled MatMul |
|   | 7 | Reduction / Scan | Sum / Histogram |
|   | 8 | 性能分析与异步 | Streams + Nsight |
| **C · LLM** | 9 | GEMM 深入 | Tensor Core GEMM |
|   | 10 | Softmax 与 Norm | Online Softmax / RMSNorm |
|   | 11 | Attention 入门 | Scaled Dot-Product |
|   | 12 | FlashAttention | Tile + Online Softmax 融合 |
|   | 13 | LLM 必备零件 | RoPE / SwiGLU / KV Cache |
|   | 14 | **Capstone** | **GPT-2 small 端到端推理** |

📍 **从首页开始**：[`docs/index.html`](docs/index.html)（本地预览：`python3 -m http.server -d docs 8000`）

---

## 快速开始

### 方式 1：Google Colab（无需本地 GPU，**推荐初学者**）

打开 `scripts/setup_colab.ipynb`，点击 "Open in Colab" → Runtime → Change runtime type → **T4 GPU** → 运行所有 cell。

### 方式 2：本地 Linux + NVIDIA GPU

```bash
# 1. 确认环境
nvcc --version          # 需要 CUDA 11.8+
nvidia-smi              # 看到你的 GPU 即可

# 2. 编译某一章
cd code/ch02_hello
make ARCH=sm_80          # A100/3090/4090 用 sm_80/sm_86/sm_89
./hello

# 3. 编译全部
make -C code all
```

### 方式 3：Docker

```bash
docker build -t cuda-tutorial scripts/docker
docker run --gpus all -it -v $PWD:/work cuda-tutorial bash
# 容器内：
make -C code/ch09_gemm run
```

### 方式 4：本机 macOS（无 NVIDIA GPU）—— 仅做 CPU reference 与教程阅读

```bash
# 跑 CPU reference 算子（不需要 nvcc）
clang++ -O2 -std=c++17 -DCPU_ONLY code/common/cpu_ref_demo.cpp -o /tmp/ref && /tmp/ref

# 预览教程: 两种方式任选
# 方式 A (最简单, 推荐): 在 Finder 里双击 docs/index.html
open docs/index.html

# 方式 B: 起本地 server (跨章节链接更可靠, 部分浏览器对 file:// 有限制时用这个)
python3 -m http.server -d docs 8000
# 浏览器打开 http://localhost:8000
```

> **方式 A vs B 的区别**：教程纯静态 HTML + CDN 引入 Mermaid/Prism，没有 fetch/XHR/ES module 这些会被 `file://` 拦的操作，所以双击直接打开 99% 功能正常。
> 个别浏览器（旧版 Safari）的 clipboard API 在 `file://` 下可能受限——教程已加 `document.execCommand` 回退；如果出问题再切到方式 B。

---

## 目录结构

```
.
├── README.md                  本文件
├── CLAUDE.md                  Claude Code 后续会话的项目说明
├── Makefile                   顶层构建（递归到各章）
├── scripts/
│   ├── docker/                Dockerfile（nvidia/cuda:12.4 基础镜像）
│   ├── setup_colab.ipynb      Google Colab 一键运行模板
│   ├── run_chapter.sh         编译并跑某章全部示例
│   └── bench.sh               性能对照 + nsys/ncu 调用示例
├── code/
│   ├── common/                公用：错误检查、计时、CPU 参考、Tensor 封装
│   ├── ch01_intro/ ... ch14_mini_llm/
│   │   ├── *.cu               每章 2–5 个主示例
│   │   ├── exercises/         练习题（_starter.cu + _solution.cu）
│   │   ├── bench/             性能对照表（核心章节）
│   │   ├── Makefile
│   │   └── README.md
├── docs/                      HTML 教程（GitHub Pages 可直接托管）
│   ├── index.html             首页 + 学习路径图
│   ├── _template/chapter.html 章节模板
│   ├── assets/                CSS / JS / Mermaid / SVG
│   ├── ch01-intro/ ... ch14-mini-llm/
│   └── glossary.html          中英术语表
├── data/                      GPT-2 权重下载脚本
└── reference/                 论文清单 + 外部链接
```

---

## 这个教程为谁准备？

✅ **适合**：
- 会用 C/C++，但从未写过并行代码的工程师
- 想理解大模型推理底层、不只想会调 PyTorch 的 ML 学习者
- 在准备 GPU 方向面试 / 想读懂 vLLM / TensorRT-LLM 源码的人

❌ **不适合**：
- 想 5 分钟跑通 ChatGPT 的产品用户（请直接用 OpenAI API）
- 不想写代码、只想看博客的读者（教程里大量 hands-on 部分）

---

## 学习建议

1. **按章节顺序学**——后面的章节强依赖前面（尤其是第 5、6 章是性能优化的灵魂）
2. **先跑代码再看 HTML**——每个示例的 `make run` 输出会让概念更具体
3. **每章必做练习**——`exercises/` 里的题不做完不要进入下一章
4. **关键瓶颈章建议多花时间**：Ch5（内存层级）、Ch6（Tile）、Ch9（GEMM）、Ch12（FlashAttention）

---

## 致谢与参考

本教程参考了：

- NVIDIA CUDA C++ Programming Guide
- 《Programming Massively Parallel Processors》(Kirk & Hwu)
- FlashAttention v1/v2 论文 (Tri Dao et al.)
- vLLM (PagedAttention), TensorRT-LLM, llama.cpp 工程实现
- CUTLASS / CuTe 文档

完整论文与链接清单见 [`reference/papers.md`](reference/papers.md) 与 [`reference/links.md`](reference/links.md)。

---

## 许可

MIT License —— 教育用途、商业用途、二次创作均欢迎，注明出处即可。
