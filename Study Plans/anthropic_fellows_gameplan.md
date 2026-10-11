# Anthropic Fellows Program — ML Systems & Performance Gameplan

**Target workstream:** ML Systems & Performance  
**Secondary:** Reinforcement Learning  
**Timeline:** 4 months to application-ready  
**Starting point:** PyTorch basics, API calls, statistical ML theory solid through ~lecture 03 (neural net fundamentals), fuzzy on 04–07

---

## What Anthropic actually wants to see

Before the month-by-month plan, understand what makes an ML Systems candidate stand out:

1. **Deep systems intuition** — not just "I used PyTorch" but "I understood why it was slow and fixed it"
2. **Public evidence** — GitHub repos with real commit history, blog posts that show you understand tradeoffs
3. **One impressive artifact** — a single project that makes a researcher say "this person thinks like us"
4. **Research-adjacent thinking** — you don't just implement, you measure, analyze, and write up what you found

The fellowship is competitive. The bar is: could this person contribute to a real research project in week 1?

---

## Where you actually stand

**You can skip:** ML 101, "what is a neural network," basic Python/PyTorch setup  
**You need to close fast:** lectures 04–07 gaps (optimization, regularization, backprop mechanics, architectures)  
**You need to build from scratch:** systems-level understanding (GPU memory, kernels, profiling, distributed training)  
**You already have:** SWE instincts that most ML candidates lack — use this aggressively

---

## Month 1: Close the gaps, get GPU-native

**Goal:** Be fluent in everything through lecture 07, and write your first GPU-profiled training loop.

### Week 1–2: Close the lecture 04–07 gap (targeted, fast)

Don't re-read your notes end-to-end. Work through the *concepts* by implementing them.

**The fastest path: Karpathy's micrograd + makemore**
- [micrograd](https://github.com/karpathy/micrograd) — build autograd from scratch (~200 lines). This will lock in backpropagation mechanics permanently.
- [makemore parts 1–4](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ) — takes you from bigrams to MLPs to BatchNorm. At your level, skim parts 1–2 quickly and go deep on 3–4 (BatchNorm, initialization, optimization).

**Specific concepts to nail:**
- Backprop: the chain rule as a graph, why gradients vanish/explode, gradient clipping
- Optimization: SGD vs. momentum vs. Adam — not just the equations, but *when* each fails
- BatchNorm / LayerNorm: what they're actually doing to the loss landscape
- Initialization: Xavier/He init and why it matters for deep nets
- Dropout and weight decay: regularization from a bias-variance lens

**Do NOT spend more than 10 days on this.** You're not trying to master it — you're trying to stop it from being a gap.

### Week 3–4: Get GPU-native

This is where you start separating yourself from most candidates.

**Read first:**
- [PyTorch internals](http://blog.ezyang.com/2019/05/pytorch-internals/) by Edward Yang (1 hour read, read it twice)
- [Making Deep Learning Go Brrrr From First Principles](https://horace.io/brrr_intro.html) — the single best intro to ML systems thinking

**Build:**  
Your first real project: a **training profiler**.

```
Project: "What is my training loop actually doing?"
- Train a small transformer (GPT-2 small or similar) on a toy dataset
- Profile it with torch.profiler — capture a trace, open it in Chrome's trace viewer
- Identify the top 3 bottlenecks
- Fix each one and measure the speedup
- Write a blog post: "I profiled my first training loop. Here's what I found."
```

Tools: `torch.profiler`, `torch.cuda.memory_summary()`, `nvidia-smi`, `nvtop`

**Key things to measure:**
- GPU utilization (should be >90% — if not, you're CPU-bound or memory-bound)
- Memory bandwidth utilization vs. compute utilization
- Time in data loading vs. forward vs. backward

**Public output (Week 4):** Post the blog + GitHub repo. Title: "Profiling a PyTorch Training Loop: A Beginner's Audit." This is your first public artifact.

---

## Month 2: Go deep on the machinery

**Goal:** Understand what happens at the hardware level. Be able to explain memory bandwidth, kernel fusion, and mixed precision to a researcher.

### Week 5–6: GPU memory model + mixed precision

**Read:**
- [CUDA Programming Model](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#programming-model) — just chapters 1–3 (don't need to become a CUDA engineer, but need the mental model)
- [Tim Dettmers: A Gentle Introduction to 8-bit Matrix Multiplication](https://huggingface.co/blog/hf-bitsandbytes-integration)

**Papers:**
- [Flash Attention (Dao et al., 2022)](https://arxiv.org/abs/2205.14135) — read the intro + Section 2 carefully; the rest is implementation detail you can skim. The key insight: IO-awareness, tiling, and why the naive attention implementation is memory-bandwidth-bound.
- [Flash Attention 2 (Dao, 2023)](https://arxiv.org/abs/2307.08691) — read as a follow-up to see what changed

**Build:**
```
Project: "Mixed precision training from scratch"
- Implement a training loop that supports fp32, fp16, and bf16
- Use torch.cuda.amp.autocast and GradScaler
- Benchmark: measure training time and GPU memory for each precision
- Identify where numerical instability appears (try training with fp16 without GradScaler)
- Blog post: "Why bf16 is usually better than fp16, and when it isn't"
```

### Week 7–8: Scaling laws + efficient architectures

**Read:**
- [Scaling Laws for Neural Language Models (Kaplan et al., 2020)](https://arxiv.org/abs/2001.08361) — the original. Read the abstract + Section 4 carefully.
- [Training Compute-Optimal LLMs (Chinchilla, Hoffmann et al., 2022)](https://arxiv.org/abs/2203.15556) — read the key result: optimal token/parameter ratio. This is referenced constantly at Anthropic.
- [The FLOPs calculation](https://www.lesswrong.com/posts/6Fpvch8RR29qLEWNH/chinchilla-s-wild-implications) — understand the math behind "how much does it cost to train this model"

**Build:**
```
Project: "Reproduce a scaling law"
- Train a series of small transformers (1M, 10M, 50M params) on the same dataset
- Plot loss vs. compute — does it follow a power law?
- Write: "Reproducing Chinchilla scaling laws on a budget"
```

This project is excellent for an application because it shows you can read a paper and implement its core claims.

---

## Month 3: Distributed training + real infrastructure

**Goal:** Understand multi-GPU training deeply. Run something non-trivial on real hardware.

### Week 9–10: Distributed training concepts

**Read:**
- [DDP Tutorial (PyTorch official)](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html) — implement every step yourself, don't just copy-paste
- [FSDP Tutorial](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html)
- [Megatron-LM paper](https://arxiv.org/abs/1909.08053) — focus on Section 3 (tensor parallelism). This is the foundation for how large models are trained.
- [ZeRO paper (Rajbhandari et al., 2020)](https://arxiv.org/abs/1910.02054) — focus on the ZeRO stages. Understand ZeRO-1/2/3 conceptually.

**Concepts to be able to explain:**
- Data parallelism vs. tensor parallelism vs. pipeline parallelism — when to use each
- Why all-reduce is the bottleneck in DDP and how FSDP changes this
- Activation checkpointing (gradient checkpointing): the memory-compute tradeoff
- Communication overlap: how to hide network latency behind compute

### Week 11–12: Run it for real

Get GPU access. Options:
- [Lambda Labs](https://lambdalabs.com/) — cheapest, good for 1–8 GPU runs (~$1.50/hr for an A100)
- [Vast.ai](https://vast.ai/) — even cheaper, less reliable
- [Google Colab Pro+](https://colab.research.google.com/) — fine for single GPU
- Apply for compute credits: [Google TPU Research Cloud](https://sites.research.google/trc/about/), [Weights & Biases Academic](https://wandb.ai/site/research)

**Build:**
```
Project: "Distributed training from scratch to production"
- Train a 100M+ parameter model using DDP across 2+ GPUs
- Implement gradient checkpointing and measure the memory savings
- Profile communication overhead vs. compute time
- Compare DDP vs. FSDP for your model size
- Blog post: "What I learned training a 100M model on 2 GPUs for $12"
```

This project proves you can operate real infrastructure, not just run examples.

---

## Month 4: Ship something that matters

**Goal:** Produce one impressive end-to-end artifact that reads like research, not a tutorial. This is your application centerpiece.

### Week 13–14: Pick your flagship project

Choose **one** of these based on what excites you most:

**Option A: Efficient fine-tuning deep dive**  
Implement LoRA from scratch (read [the paper](https://arxiv.org/abs/2106.09685)), compare to full fine-tuning and QLoRA on the same task. Write a rigorous comparison with ablations. This is immediately practical at Anthropic.

**Option B: Kernel optimization**  
Take a real bottleneck (e.g., attention, a specific layer type), write a custom CUDA/Triton kernel, and benchmark it against PyTorch's default. Read [Triton's tutorials](https://triton-lang.org/main/getting-started/tutorials/index.html). Shows you can go below the framework level.

**Option C: Synthetic data pipeline**  
Build an end-to-end pipeline: generate synthetic training data with a language model, train on it, evaluate. This is directly relevant to Anthropic's work. Read [Phi-1 paper](https://arxiv.org/abs/2306.11644) and [Alpaca paper](https://arxiv.org/abs/2212.10560).

**Option D: Training stability investigation**  
Pick a known instability (loss spikes, gradient explosion in large models) and do a careful empirical investigation. What causes it? What fixes it? Reproduce known results, then add something new. Closest to actual research work.

### Week 15–16: Polish and apply

**Your application package:**
- 1 flagship project with a proper writeup (treat it like a short paper: motivation, method, experiments, results, discussion)
- 3–4 blog posts showing your progression
- Clean GitHub with real commit history (no "add everything" commits)
- A clear narrative: "I came from SWE, I spent 4 months going deep on ML systems, here's what I built"

---

## The reading list (in order)

### Foundations (Month 1)
- [PyTorch Internals — Edward Yang](http://blog.ezyang.com/2019/05/pytorch-internals/)
- [Making Deep Learning Go Brrrr — Horace He](https://horace.io/brrr_intro.html)
- [The Illustrated Transformer — Jay Alammar](https://jalammar.github.io/illustrated-transformer/) (if not already solid on transformers)

### Core papers (Month 2)
- Flash Attention (Dao et al., 2022)
- Flash Attention 2 (Dao, 2023)
- Scaling Laws for Neural LMs (Kaplan et al., 2020)
- Chinchilla (Hoffmann et al., 2022)

### Systems papers (Month 3)
- Megatron-LM (Shoeybi et al., 2019)
- ZeRO (Rajbhandari et al., 2020)
- PyTorch DDP (Li et al., 2020) — [arxiv](https://arxiv.org/abs/2006.15704)

### Flagship-project papers (Month 4 — pick based on your project)
- LoRA (Hu et al., 2021)
- QLoRA (Dettmers et al., 2023)
- Triton (Tillet et al., 2019)
- Phi-1 (Gunasekar et al., 2023)

### Optional but high-ROI
- [Andrej Karpathy's llm.c](https://github.com/karpathy/llm.c) — GPT-2 in ~1000 lines of C/CUDA. Reading it is worth more than most tutorials.
- [The Transformer Family](https://lilianweng.github.io/posts/2023-01-27-the-transformer-family-v2/) — Lilian Weng's blog, bookmark it

---

## Blog post topics (in order — write these as you go)

1. **Month 1:** "I profiled my first PyTorch training loop. Here's what was actually slow." *(write from real data, not theory)*
2. **Month 2:** "Why your training loop is memory-bandwidth-bound (and what to do about it)"
3. **Month 2:** "Reading Flash Attention without a PhD: what the paper is actually saying"
4. **Month 3:** "What distributed training actually is, explained by someone who just learned it"
5. **Month 4:** [Your flagship project writeup — make this one count]

**Where to post:** Substack or a personal site + cross-post to Hacker News when ready. Don't post everything immediately — let the quality accumulate.

---

## Weekly rhythm

Each week should have all three:

| Track | Time | What |
|-------|------|------|
| **Build** | ~15–20 hrs | Active coding on your current project |
| **Read** | ~5 hrs | 1–2 papers or articles, take real notes |
| **Write** | ~3–5 hrs | Draft or polish a blog post |

Papers should be read with a pen (or obsidian/notion). For each paper: (1) what problem does this solve, (2) what's the key insight, (3) what would I need to implement this, (4) what are the limitations.

---

## What to cut if you're short on time

In order of what matters least for ML Systems:
1. The RL secondary path (save for post-application)
2. Deep CUDA kernel writing (understand it conceptually, don't implement unless you choose Option B flagship)
3. The philosophy/AI safety reading (not relevant for ML Systems application)

Focus relentlessly on: **profile → measure → optimize → write it up**.

---

## The application essay framing

When you write your application, the narrative is:

> "I came from software engineering — I knew how to build things, but I didn't know how to build things that *work at scale* under real constraints. Over the past four months I went deep on ML systems: I profiled training loops, implemented distributed training from scratch, and [your flagship project]. What I found is that [specific non-obvious thing you learned]. I want to work at Anthropic because [specific problem in their systems work that genuinely interests you]."

The SWE background is an asset, not a liability. Own it.

---

*Last updated: June 2026*
