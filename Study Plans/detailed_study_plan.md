# Detailed Week-by-Week Study Plan
## Anthropic Fellows — ML Systems & Performance Track

**Daily time budget:** ~4–5 hrs/day (adjustable)  
**Split each day roughly:** 2.5 hrs build · 1.5 hrs read · 1 hr write/reflect  

---

## PHASE 1: CLOSE THE GAPS (Weeks 1–2)

### Week 1 — Backprop, Optimization, Normalization

**Goal:** Implement autograd from scratch. Never be confused by backprop again.

---

**Monday**  
- Read: [Yes you should understand backprop — Karpathy](https://karpathy.medium.com/yes-you-should-understand-backprop-e2f06eab496b) (~30 min)  
- Read: [Calculus on Computational Graphs — Olah](https://colah.github.io/posts/2015-08-Backprop/) (~30 min)  
- Code: Work through [micrograd](https://github.com/karpathy/micrograd) — read the source fully, then **delete it and rewrite it from memory**. Don't move on until your version passes its test suite.  
- Done when: `Value` class with `__add__`, `__mul__`, `__pow__`, `relu`, `backward()` all working. Can compute gradient of a simple 2-layer net manually and confirm it matches.

**Tuesday**  
- Watch: [Karpathy Zero-to-Hero — Lecture 1 (micrograd)](https://www.youtube.com/watch?v=VMj-3S1tku0) — skip anything you already implemented; watch for gaps  
- Code: Extend your micrograd to support `tanh`, `exp`, `log`. Add a working SGD optimizer on top of it.  
- Exercise: Implement XOR with your autograd. If it doesn't learn, debug until it does.  
- Done when: XOR converges in <2000 steps.

**Wednesday**  
- Read: [An overview of gradient descent optimization algorithms — Ruder](https://ruder.io/optimizing-gradient-descent/) — **Sections 1–6 only** (~1 hr)  
- Key concepts to nail: momentum intuition, Adam's bias correction (why divide by `1-β^t`), why Adam can generalize worse than SGD in some settings  
- Code: Implement SGD with momentum, RMSProp, and Adam **from scratch** in PyTorch (no `torch.optim`). Test on MNIST or a toy quadratic.  
- Done when: Your Adam matches `torch.optim.Adam` loss curve within 1%.

**Thursday**  
- Read: [Batch Normalization original paper (Ioffe & Szegedy, 2015)](https://arxiv.org/abs/1502.03167) — **Abstract + Sections 1, 2, 3** only. Skip the Inception architecture parts.  
- Read: [Layer Normalization (Ba et al., 2016)](https://arxiv.org/abs/1607.06450) — Abstract + Section 2 (~20 min)  
- Code: Implement `BatchNorm1d` from scratch (forward + backward). Test it by training a deep MLP on MNIST with and without it — plot loss curves.  
- Done when: Your BatchNorm matches `nn.BatchNorm1d` output within float precision.

**Friday**  
- Watch: [Karpathy Zero-to-Hero — makemore Part 3 (BatchNorm)](https://www.youtube.com/watch?v=P6sfmUTpUmc) — watch at 1.5x  
- Code: Follow along but implement from scratch in a parallel notebook. Focus on: why saturated activations kill gradients, the dead neuron problem, how BN fixes the loss landscape.  
- Experiment: Train a 6-layer MLP without BN, with BN, with LayerNorm. Plot: loss curves, gradient norms per layer, activation distributions per layer (use matplotlib histograms).  
- Write: 300-word journal entry: "What does BatchNorm actually do to the loss landscape?" Use your plots as evidence.

**Weekend**  
- Read: [The Marginal Value of Momentum for Small Learning Rate SGD — Gitman et al.](https://arxiv.org/abs/1910.09939) — just abstract + Section 1 (~20 min). Intuition: when does momentum help?  
- Code: Finish anything incomplete from the week. Clean up your notebooks. Push to GitHub.  
- Done when: GitHub repo `ml-foundations` exists with clean, documented implementations of: micrograd, custom optimizers, BatchNorm. Readme explains what each file does.

---

### Week 2 — Transformers, Initialization, Regularization

**Goal:** Be genuinely fluent in transformers (not just "I know what attention is").

---

**Monday**  
- Read: [The Illustrated Transformer — Alammar](https://jalammar.github.io/illustrated-transformer/) — even if you've read it, re-read it. Focus specifically on: why queries/keys/values, what the scores mean geometrically, why multi-head.  
- Read: [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762) — **Sections 1, 2, 3** (architecture). Skip experiments/results for now.  
- Done when: You can explain scaled dot-product attention without looking at anything.

**Tuesday**  
- Watch: [Karpathy Zero-to-Hero — makemore Part 5 (GPT)](https://www.youtube.com/watch?v=kCc8FmEb1nY) — this is the famous "build GPT from scratch" video. Watch it fully.  
- Code: Implement a GPT (decoder-only transformer) from scratch. Target: ~10M params, trains on tiny shakespeare or similar.  
- Done when: Model generates coherent-ish text after 5000 training steps.

**Wednesday**  
- Read: [Weight initialization in neural networks — deeplearning.ai notes](https://www.deeplearning.ai/ai-notes/initialization/) (~30 min)  
- Read: [Delving Deep into Rectifiers (He et al., 2015)](https://arxiv.org/abs/1502.01852) — **Sections 1–3** (~30 min). This is the He init paper.  
- Code: In your GPT implementation, experiment with different initializations: all zeros, random normal (bad), Xavier, He, scaled init (GPT-2 style: `1/sqrt(2 * n_layers)`). Plot: gradient norms at init for each. See which one gives "nice" gradients.  
- Done when: You can explain *why* GPT-2 uses `0.02 * 1/sqrt(2L)` for residual projections.

**Thursday**  
- Read: [Regularization in Deep Learning — distill.pub (dropout section)](https://www.dropbox.com/s/7c8a56sr5b1uoym/dropout-brief.pdf) (~20 min)  
- Code: Add dropout to your GPT. Train with and without it on a small dataset. Plot validation vs. training loss for both. Observe the gap.  
- Implement label smoothing from scratch (don't use PyTorch's built-in). Test it.  
- Experiment: What happens when you set dropout=0.5 on a small model? Explain what you see.

**Friday**  
- Code: Profile your GPT training loop for the first time.  
  - Use `torch.profiler.profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA])`  
  - Run 100 training steps with profiling enabled  
  - Export the trace: `prof.export_chrome_trace("trace.json")`  
  - Open in `chrome://tracing`  
  - **Question to answer:** Where is time being spent? What's the ratio of forward:backward:optimizer step?  
- Write: 500-word journal entry answering those questions. This becomes the skeleton of your first blog post.

**Weekend**  
- Read: [PyTorch Internals — Edward Yang](http://blog.ezyang.com/2019/05/pytorch-internals/) — **full read, no skimming** (~2 hrs). This is foundational for everything in Phase 2.  
- Done when: You can explain what a `TensorImpl` is, what `autograd` dispatch does, and why PyTorch's dynamic graph is slower than static graphs but more flexible.  
- Push all code. Your `ml-foundations` repo should now have your GPT implementation.

---

## PHASE 2: GPU-NATIVE (Weeks 3–6)

### Week 3 — Profiling & the Systems Mental Model

**Goal:** Understand *why* training loops are slow. Build your first real profiling report.

---

**Monday**  
- Read: [Making Deep Learning Go Brrrr From First Principles — Horace He](https://horace.io/brrr_intro.html) — **read it twice**. This is the most important document in your entire plan.  
- Key mental model to internalize: the roofline model. Every operation is either compute-bound or memory-bandwidth-bound. Almost nothing in transformers is compute-bound. Almost everything is memory-bandwidth-bound.  
- Done when: You can explain the arithmetic intensity of matrix multiply vs. elementwise ops without looking anything up.

**Tuesday**  
- Read: [NVIDIA A100 spec sheet](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf) — just the "Performance" section. Note: 312 TFLOPS (BF16), 2 TB/s memory bandwidth. These numbers are your reference points forever.  
- Code: Write a benchmark harness in PyTorch:  
  ```python
  # Measure achieved FLOPS and memory bandwidth for:
  # 1. Large matrix multiply (M=N=K=4096)
  # 2. Elementwise add of a large tensor
  # 3. Softmax over a large tensor
  # 4. LayerNorm over a large tensor
  ```  
  Use `torch.cuda.Event` for timing. Compare achieved vs. theoretical peak.  
- Done when: You have a table showing % of theoretical peak for each operation. Most elementwise ops should be <30% of compute peak. MatMul should be >50%.

**Wednesday**  
- Read: [CUDA Programming Model — NVIDIA docs](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html) — **Chapter 1 (Introduction) + Chapter 2 (Programming Model) only** (~1.5 hrs). Don't implement anything yet — just get the mental model: threads, blocks, warps, SM occupancy.  
- Done when: You can explain what a warp is, what warp divergence is, and why memory coalescing matters.

**Thursday**  
- Read: [GPU Memory — Tim Dettmers](https://timdettmers.com/2020/09/07/which-gpu-for-deep-learning/) — the memory section specifically  
- Code: Write a memory profiler for your GPT:  
  ```python
  torch.cuda.reset_peak_memory_stats()
  # ... training step ...
  print(torch.cuda.memory_summary())
  ```  
  For each component (embedding, attention, FFN, loss), measure peak memory. Build a pie chart.  
- Question to answer: What fraction of GPU memory is activations vs. parameters vs. gradients vs. optimizer state?  
- Done when: You can predict, before measuring, roughly how much memory a training step uses given model size and batch size.

**Friday**  
- Code: Your first real project — a thorough profiling report on your GPT.  
  - Profile with `torch.profiler`, set `record_shapes=True, with_stack=True`  
  - Identify top 5 slowest ops  
  - For each: is it compute-bound or memory-bandwidth-bound? How do you know?  
  - Implement 3 optimizations: (1) `torch.compile`, (2) increase batch size until GPU util >85%, (3) pin memory + num_workers for DataLoader  
  - Measure speedup of each  
- Write: First blog post draft — "I profiled a GPT training loop. Here's what was actually slow."  
  Use your real numbers. Structure: what I expected → what I found → what I did → the numbers.

**Weekend**  
- Polish and publish the blog post. Post to a personal site or Substack.  
- Start a GitHub repo `gpt-profiling` with clean code and a README that links to the post.  
- Done when: The post is live. Link in Twitter/X bio or LinkedIn.

---

### Week 4 — torch.compile, Kernel Fusion, Custom Ops

**Goal:** Understand what happens when PyTorch optimizes your code, and why it helps.

---

**Monday**  
- Read: [torch.compile — PyTorch 2.0 docs introduction](https://pytorch.org/docs/stable/torch.compiler_get_started.html) (~30 min)  
- Read: [TorchDynamo and FX Graphs — PyTorch blog](https://pytorch.org/blog/torchdynamo-towards-the-graph-capture/) (~30 min)  
- Done when: You can explain what TorchDynamo does, what a FX graph is, and why graph capture enables kernel fusion.

**Tuesday**  
- Code: Apply `torch.compile` to your GPT with different backends and modes:  
  ```python
  model = torch.compile(model, mode="default")
  model = torch.compile(model, mode="reduce-overhead")
  model = torch.compile(model, mode="max-autotune")  # takes longer to compile
  ```  
  Benchmark each. Note the first-step compilation overhead vs. steady-state speedup.  
- Inspect the compiled graph: `torch._dynamo.explain(model)(input)` to see what it captured and what it fell back to.

**Wednesday**  
- Read: [Kernel Fusion — distill.pub informal explainer](https://horace.io/brrr_intro.html) — re-read the kernel fusion section of the Brrrr post  
- Read: [Writing Custom CUDA Kernels — PyTorch extension tutorial](https://pytorch.org/tutorials/advanced/cpp_extension.html) — **just the intro and first example** (~45 min)  
- Code: Write a fused `gelu + linear` kernel using PyTorch's `torch.autograd.Function`. Compare to the unfused version. Does it go faster? Why or why not?

**Thursday**  
- Read: [Flash Attention (Dao et al., 2022)](https://arxiv.org/abs/2205.14135)  
  - **Section 1 (Intro):** Why is standard attention slow? What is "IO-aware"?  
  - **Section 2 (Background):** GPU memory hierarchy, arithmetic intensity of standard attention  
  - **Section 3 (FlashAttention algorithm):** Tiling. Understand the key insight: recompute vs. store activations.  
  - Skip Section 4+ on first read  
- Done when: You can explain on a whiteboard why naive attention is memory-bandwidth-bound and how FlashAttention fixes it.

**Friday**  
- Code: Benchmark standard attention vs. `torch.nn.functional.scaled_dot_product_attention` (which uses Flash Attention under the hood in PyTorch 2.0+):  
  ```python
  # Sequence lengths: 128, 512, 2048, 8192
  # Measure: FLOPS, memory usage, time
  # Plot: speedup ratio as function of sequence length
  ```  
- Write: 400-word technical note for your blog — "Flash Attention: what the paper actually says, stripped of the math"

**Weekend**  
- Read: [Flash Attention 2 (Dao, 2023)](https://arxiv.org/abs/2307.08691) — what changed from v1? (~1 hr)  
- Read: [Triton — Tillet et al. intro + tutorial 1](https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html) (~1 hr). Don't implement yet — just understand the abstraction.  
- Push all code. Week 4 repo: add the attention benchmarking code.

---

### Week 5 — Mixed Precision + Memory Optimization

**Goal:** Understand fp16/bf16 deeply, implement mixed precision from scratch, cut memory in half.

---

**Monday**  
- Read: [Mixed Precision Training (Micikevicius et al., 2018)](https://arxiv.org/abs/1710.03740) — **Sections 1–3** (~45 min). Key: loss scaling, why fp16 underflows, why bf16 doesn't need loss scaling.  
- Done when: You can explain the difference between fp16 and bf16 in terms of exponent bits vs. mantissa bits, and why that matters for training stability.

**Tuesday**  
- Code: Implement mixed precision training manually (without `torch.cuda.amp`):  
  ```python
  # Cast model to fp16 for forward pass
  # Keep master weights in fp32
  # Scale the loss before backward
  # Unscale gradients before optimizer step
  # Skip step if gradients contain inf/nan
  ```  
  Train your GPT with this. Compare: loss curve vs. fp32 baseline. Memory usage. Training time.

**Wednesday**  
- Code: Now use `torch.cuda.amp.autocast` + `GradScaler`. Verify it matches your manual implementation.  
  - Read the [AMP tutorial](https://pytorch.org/docs/stable/amp.html) — specifically the "typical use" section  
  - Experiment: What ops are cast to fp16 automatically? Which stay in fp32? Use `torch.autocast.__enter__` with dtype logging to find out.

**Thursday**  
- Code: Gradient checkpointing (activation checkpointing):  
  - Read: [PyTorch checkpoint docs](https://pytorch.org/docs/stable/checkpoint.html) (~20 min)  
  - Apply `torch.utils.checkpoint.checkpoint` to your GPT's transformer blocks  
  - Measure: memory reduction vs. time overhead  
  - Plot: peak memory and tokens/sec as you vary which layers are checkpointed  
- Done when: You can explain the memory-compute tradeoff and when you'd choose to checkpoint vs. not.

**Friday**  
- Code: Gradient accumulation — simulate a batch size of 256 with physical batch size of 16:  
  ```python
  accumulation_steps = 16
  for i, batch in enumerate(dataloader):
      loss = model(batch) / accumulation_steps
      loss.backward()
      if (i + 1) % accumulation_steps == 0:
          optimizer.step()
          optimizer.zero_grad()
  ```  
  Verify: does the loss match single large-batch training?  
- Benchmark: gradient accumulation vs. large batch. What's the time overhead?

**Weekend**  
- Write: Blog post 2 — "Why your training loop is memory-bandwidth-bound (and what to do about it)"  
  Cover: mixed precision, gradient checkpointing, gradient accumulation. Use your real benchmark numbers.  
- Push all code to `mixed-precision-training` repo.

---

### Week 6 — Scaling Laws + Efficient Architecture

**Goal:** Understand the math behind how big to make a model and how long to train it.

---

**Monday**  
- Read: [Scaling Laws for Neural Language Models (Kaplan et al., 2020)](https://arxiv.org/abs/2001.08361)  
  - **Abstract + Section 1:** The three key findings  
  - **Section 3:** The power law relationships  
  - **Section 4 (key):** Optimal allocation of compute budget  
  - Skip Sections 5–7 on first read  
- Done when: You can explain the Kaplan et al. conclusion: "given a fixed compute budget, scale model size more than data"

**Tuesday**  
- Read: [Chinchilla (Hoffmann et al., 2022)](https://arxiv.org/abs/2203.15556)  
  - **Sections 1–3:** The key result — Kaplan et al. was wrong about the optimal ratio  
  - **Figure 1 + Table 1:** The optimal tokens-per-parameter ratio (~20 tokens/param)  
  - [Chinchilla's Wild Implications — LessWrong](https://www.lesswrong.com/posts/6Fpvch8RR29qLEWNH/chinchilla-s-wild-implications) (~30 min read — the best explainer)  
- Done when: You can explain why GPT-3 was "undertrained" by Chinchilla standards.

**Wednesday**  
- Code: FLOPs calculator — write a function that takes model config and returns:  
  - Total parameters  
  - FLOPs per forward pass  
  - FLOPs per training step (fwd + bwd ≈ 3x forward)  
  - Chinchilla-optimal dataset size given those params  
  - Time to train on 1x A100 at 50% MFU  
  Use this to compute estimates for GPT-2 small, GPT-2 large, LLaMA-7B, LLaMA-70B. Sanity-check against published training times.

**Thursday**  
- Code: Scaling law project begins.  
  - Train 5 models: 1M, 5M, 20M, 50M, 100M params  
  - Same architecture family (just vary d_model, n_heads, n_layers proportionally)  
  - Same dataset, same training duration (compute-matched)  
  - Log: train loss, val loss, tokens/sec, peak memory  
  - Use wandb or tensorboard for logging  

**Friday**  
- Code: Continue training. While models train, fit a power law to your loss vs. compute curve.  
  - Use scipy curve_fit: `L = A * C^(-alpha)`  
  - Does your alpha match Kaplan et al.'s reported values?  
- Write: Start draft of blog post 3 — "Reproducing Scaling Laws on a Budget"

**Weekend**  
- Finish blog post 3 and publish  
- Review all code from weeks 3–6. Refactor anything messy. Your GitHub should now have 3 clean repos with READMEs.

---

## PHASE 3: DISTRIBUTED TRAINING (Weeks 7–10)

### Week 7 — DDP: Theory + Implementation

**Goal:** Understand distributed data parallelism deeply enough to debug it.

---

**Monday**  
- Read: [Pytorch DDP paper (Li et al., 2020)](https://arxiv.org/abs/2006.15704) — **Sections 1–4** (~1 hr). Focus on: gradient bucketing, all-reduce overlap with backward pass, the `no_sync` context manager.  
- Read: [DDP Tutorial — PyTorch official](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html) — read it fully, code along.

**Tuesday**  
- Code: Set up a single-machine multi-GPU DDP training run for your GPT.  
  - If you don't have multi-GPU locally: use Lambda Labs (1x instance with 2 A100s, ~$3/hr). Budget: $10 for this week.  
  - Use `torch.distributed.init_process_group`, `DistributedDataParallel`  
  - Verify: gradients are synchronized correctly (compare to single-GPU)  

**Wednesday**  
- Code: Benchmark DDP scaling efficiency.  
  - Train your 100M GPT on 1, 2, 4 GPUs (if available)  
  - Measure: tokens/sec per GPU for each configuration  
  - Compute linear scaling efficiency: `efficiency = (N_gpu * single_gpu_throughput) / multi_gpu_throughput`  
  - Profile communication: how much time is spent in all-reduce?

**Thursday**  
- Read: [Ring all-reduce explainer](https://andrew.gibiansky.com/blog/machine-learning/baidu-allreduce/) — understand the algorithm  
- Code: Manually implement `all_reduce` using `dist.all_reduce` and verify it gives the same gradients as DDP.  
- Experiment: Set `find_unused_parameters=True` vs. `False`. What's the performance difference? Why?

**Friday**  
- Code: Implement gradient compression — reduce communication overhead:  
  - Try `PowerSGD` (PyTorch has it in `torch.distributed.algorithms.ddp_comm_hooks`)  
  - Measure: throughput improvement vs. convergence impact  
- Write: 400-word technical note — "What DDP actually does: allreduce, bucketing, and overlap"

**Weekend**  
- Read: [Megatron-LM (Shoeybi et al., 2019)](https://arxiv.org/abs/1909.08053) — **Sections 1–3** (tensor parallelism). Understand: why tensor parallelism, what a "tensor parallel group" is, the all-reduce pattern.  
- Done when: You can draw a diagram of how a single transformer layer is split across 4 GPUs in tensor parallelism.

---

### Week 8 — FSDP: Sharding Parameters

**Goal:** Understand Fully Sharded Data Parallel — the key technique for training 70B+ models.

---

**Monday**  
- Read: [ZeRO paper (Rajbhandari et al., 2020)](https://arxiv.org/abs/1910.02054) — **Sections 1–4** (~1.5 hrs)  
  Key: ZeRO Stage 1 (shard optimizer state), Stage 2 (+ gradients), Stage 3 (+ parameters). Understand the memory savings formula for each.  
- Done when: For a 7B model on 8 GPUs, you can compute the per-GPU memory for each ZeRO stage.

**Tuesday**  
- Read: [FSDP Tutorial — PyTorch official](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html) — code along  
- Read: [Introducing PyTorch Fully Sharded Data Parallel — Meta blog](https://engineering.fb.com/2021/07/15/open-source/fsdp/)  
- Code: Wrap your GPT in FSDP. Use `FullyShardedDataParallel` with `auto_wrap_policy`.

**Wednesday**  
- Code: Compare DDP vs. FSDP for your 100M model:  
  - Measure: peak memory, tokens/sec, gradient sync time  
  - Try FSDP ShardingStrategy: `FULL_SHARD`, `SHARD_GRAD_OP`, `NO_SHARD`  
  - Plot: memory vs. throughput tradeoff for each  

**Thursday**  
- Code: Checkpointing with FSDP — this is non-trivial:  
  ```python
  # FSDP requires special save/load:
  from torch.distributed.fsdp import StateDictType
  with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT):
      state = model.state_dict()
  ```  
  Implement save and load correctly. Verify the loaded model produces identical outputs.

**Friday**  
- Read: [Activation Memory (Korthikanti et al., 2022)](https://arxiv.org/abs/2205.05198) — **Sections 1–3**. This is the paper that quantifies activation memory in transformers and proposes sequence parallelism.  
- Code: Measure activation memory in your model per layer per sequence length. Does it match the formula in the paper?

**Weekend**  
- Write: Blog post 4 — "What distributed training actually is, explained by someone who just learned it"  
  Cover DDP + FSDP + ZeRO stages. Use diagrams (even hand-drawn, photographed). This is a high-ROI post because almost nothing explains this accessibly.

---

### Week 9 — Pipeline Parallelism + Real Multi-Node Setup

**Goal:** Understand the third dimension of parallelism. Run a real multi-node job.

---

**Monday**  
- Read: [GPipe (Huang et al., 2019)](https://arxiv.org/abs/1811.06965) — **Sections 1–3** (~45 min). Understand micro-batching and pipeline bubbles.  
- Read: [PipeDream (Narayanan et al., 2019)](https://arxiv.org/abs/1806.03377) — **Abstract + Section 2** (~20 min). 1F1B schedule.  
- Done when: You can draw the pipeline schedule diagram and explain what "bubble overhead" means.

**Tuesday**  
- Read: [3D Parallelism explainer — DeepSpeed docs](https://www.deepspeed.ai/tutorials/pipeline/) (~45 min)  
- Code: Implement simple pipeline parallelism manually — split a model into 2 stages on 2 GPUs, send activations between them. This is pedagogical, not production.

**Wednesday–Thursday**  
- Compute: Set up a 2-node (or 4-GPU) training run. Get the `torchrun` launcher working.  
  ```bash
  torchrun --nproc_per_node=4 --nnodes=2 \
    --rdzv_id=100 --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:29400 \
    train.py
  ```  
  - On Lambda Labs: rent 2x 4-GPU nodes (~$6/hr). Budget ~$15 for this experiment.  
  - Train your 100M GPT with DDP across 8 GPUs total  
  - Measure: is inter-node bandwidth the bottleneck? (NVLink vs. Ethernet)

**Friday**  
- Analysis: Profile the multi-node run. How much time is spent in communication vs. compute?  
- Implement: gradient accumulation + DDP + mixed precision together. This is how real production training works.  
- Done when: You have a training script that runs efficiently on multi-GPU with all three techniques simultaneously.

**Weekend**  
- Read: [Efficient Large Scale Language Modeling with Mixtures of Experts (Artetxe et al., 2021)](https://arxiv.org/abs/2112.10684) — **Sections 1–2** (~30 min). Understand MoE as a way to scale parameters without proportionally scaling compute.  
- Clean up all distributed training code into a single repo `distributed-training-experiments` with good documentation.

---

### Week 10 — Profiling Distributed Training + Optimization

**Goal:** Profile a multi-GPU run and identify the actual bottlenecks.

---

**Monday–Tuesday**  
- Code: Distributed profiling — use PyTorch's profiler on a DDP run:  
  ```python
  with torch.profiler.profile(
      activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
      schedule=torch.profiler.schedule(wait=1, warmup=1, active=3),
      on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/ddp_run'),
  ) as prof:
      for step in training_loop():
          prof.step()
  ```  
  - Profile all-reduce time specifically: what % of total training time is communication?  
  - Try NCCL vs. Gloo. Does it matter for your setup?

**Wednesday**  
- Read: [NVIDIA Nsight Systems tutorial](https://docs.nvidia.com/nsight-systems/) — just the Quick Start (~30 min)  
- Run Nsight on your training loop. Compare to torch.profiler — what additional detail do you get?

**Thursday–Friday**  
- Optimization sprint: pick 3 bottlenecks from your profiling and fix them. Measure before/after for each.  
  Ideas: overlap communication with computation, use `gradient_as_bucket_view=True` in DDP, tune bucket sizes.  
- Document: Create a table: bottleneck → fix → before → after → explanation.

**Weekend**  
- This weekend is a review + prep weekend. Review weeks 7–10. You should now be able to:  
  - Set up DDP, FSDP training from scratch  
  - Profile distributed runs and identify communication vs. compute time  
  - Explain ZeRO stages and when to use each  
  - Estimate memory requirements for training a given model size  

---

## PHASE 4: FLAGSHIP PROJECT (Weeks 11–16)

### Week 11 — Choose + Scope Your Flagship

**Goal:** Pick the right project and scope it so it's both impressive and finishable.

---

**Monday**  
- Decide on your flagship project (see the options in the main gameplan). The best choice is the one that:
  1. Genuinely excites you
  2. Produces a result someone could cite or build on
  3. You can finish in 4 weeks

- If undecided: **Option A (LoRA investigation)** is the highest ROI for ML Systems at Anthropic right now. It's directly relevant to their fine-tuning work, has a tight scope, and lets you combine your systems profiling skills with something new.

**Tuesday–Wednesday**  
- **If Option A (LoRA):**  
  - Read: [LoRA (Hu et al., 2021)](https://arxiv.org/abs/2106.09685) — **full paper** (~2 hrs)  
  - Read: [QLoRA (Dettmers et al., 2023)](https://arxiv.org/abs/2305.14314) — **Sections 1–3** (~1 hr)  
  - Code: Implement LoRA from scratch (no PEFT library). Just: freeze base weights, add trainable `A` and `B` matrices, inject into attention projections.  

- **If Option B (Triton kernel):**  
  - Work through [Triton tutorials 1–3](https://triton-lang.org/main/getting-started/tutorials/index.html) — vector add, fused softmax, matrix multiply  

- **If Option C (Synthetic data pipeline):**  
  - Read: [Phi-1 (Gunasekar et al., 2023)](https://arxiv.org/abs/2306.11644) and [Alpaca (Taori et al., 2023)](https://arxiv.org/abs/2302.13971) — both (~2 hrs total)  

**Thursday–Friday**  
- Scope document: Write a 1-page project spec:
  - Problem statement (1 sentence)
  - What you'll build
  - Experiments you'll run (list them: at minimum 4 comparisons)
  - How you'll evaluate success
  - What you'll write up
  
  This forces you to think before you code. Share it on Twitter/X for feedback.

**Weekend**  
- Set up the project repo. Write the README with just the problem statement and planned experiments. Commit. The goal is: anyone looking at your GitHub can see what you're building and why.

---

### Weeks 12–14 — Build the Flagship

These three weeks are heads-down building. Below is guidance for Option A (LoRA investigation) as the default; adapt if you chose differently.

---

**Week 12: Implementation + Baseline**

Monday–Wednesday:
- Implement full LoRA from scratch:
  - `LoRALinear` module: `y = W_0 x + (B A) x * alpha/rank`
  - Support: choosing which layers to apply LoRA to, rank r, alpha scaling
  - Support: merging LoRA weights into base model for inference
- Test: does your LoRA implementation match HuggingFace PEFT's output? (It should.)

Thursday–Friday:
- Run baseline experiments: fine-tune a 1.3B model (GPT-2 XL or LLaMA-1.3B) three ways:
  1. Full fine-tuning
  2. Your LoRA implementation (r=8)
  3. LoRA with different ranks: r=1, 4, 8, 16, 32, 64
- Dataset: something concrete — Alpaca instructions, or a code dataset, or a specific classification task
- Log: val loss, training time, peak GPU memory, tokens/sec

**Week 13: Ablations + Surprises**

Run the experiments that turn a project into research:

- Which layers matter? Apply LoRA to: only Q/K/V, only Q/V, only FFN, only output projections, all linear layers. How does final performance differ?
- Rank vs. performance: make a plot. Is there a "knee" in the rank-performance curve?
- Initialization sensitivity: what if A is initialized to non-zero? What if B is non-zero? (Standard: A~N(0,σ), B=0 so initial delta_W=0)
- Learning rate scaling: how should LR scale with rank?
- Profile each variant: what is the memory and compute overhead as a function of rank?

**Week 14: QLoRA + Quantization**

- Read: [bitsandbytes library](https://github.com/TimDettmers/bitsandbytes) — understand 4-bit NormalFloat quantization  
- Implement (or use): 8-bit base model + LoRA adapters (QLoRA setup)  
- Experiment: Full fine-tune vs. LoRA vs. QLoRA — quality/memory/speed tradeoff  
- Surprise question to investigate: Does quantizing the base model hurt LoRA training more when rank is low vs. high?

---

### Week 15 — Write It Up Like Research

**Goal:** Produce a writeup that reads like a short paper, not a blog post.

**Structure:**
1. **Introduction** (~300 words): What problem, why it matters, what you found
2. **Background** (~400 words): LoRA mechanics, QLoRA, why efficient fine-tuning matters at scale
3. **Experimental Setup** (~300 words): Model, dataset, hardware, evaluation metric
4. **Results** (~600 words + figures): Your ablations. Be precise. "Rank 8 achieves 97% of full fine-tuning quality at 12% of the memory cost on our benchmark."
5. **Discussion** (~300 words): What surprised you. What the tradeoffs mean in practice. When you'd use LoRA vs. not.
6. **Conclusion** (~150 words)

**Figures to include:**
- Rank vs. val loss curve
- Memory usage comparison (bar chart: full FT, LoRA r=8, QLoRA r=8)
- Training throughput comparison
- Layer ablation heatmap (which layers to LoRA → performance)

**Monday–Wednesday:** Write the full draft  
**Thursday:** Revise for clarity. Cut anything that doesn't add to the argument.  
**Friday:** Polish and publish — as a blog post that mirrors the paper structure. This is your best public artifact.

---

### Week 16 — Application + Final Polish

**Monday–Tuesday**  
- Review all public outputs:
  - 5 blog posts published
  - 4+ GitHub repos with clean history and READMEs
  - 1 flagship writeup
- Polish anything rough. Make sure every repo README answers: what is this, why does it exist, how to run it.

**Wednesday**  
- Write your application essay draft. Use this structure:
  - Para 1: Who you are, what you built in the last 4 months (3–4 concrete things)
  - Para 2: The specific thing you found most interesting — the non-obvious insight from your flagship project
  - Para 3: What you want to work on at Anthropic specifically (be specific — reference their actual published work)
  - Para 4: Why you're ready now

**Thursday**  
- Review: does your application link to the flagship writeup prominently? Is your GitHub linked? Is the repos' README quality something you'd be proud to show a researcher?

**Friday**  
- Submit.  
- Take the weekend off. You earned it.

---

## Checkpoints (self-assessment)

Use these to verify you're on track. Be honest.

| Week | You should be able to... |
|------|--------------------------|
| 2 | Implement a GPT from scratch in <4 hrs |
| 4 | Profile any PyTorch model and identify compute vs. memory-bound ops |
| 6 | Compute Chinchilla-optimal model/data size for a given compute budget |
| 8 | Set up DDP and FSDP training from scratch, explain ZeRO stages |
| 10 | Profile a distributed run and identify communication bottlenecks |
| 14 | Explain your flagship project's key finding in 2 sentences |
| 16 | Explain every project in your GitHub to a researcher without prep |

---

## Resource Index (all links in one place)

**Foundational reads (do first)**
- https://karpathy.medium.com/yes-you-should-understand-backprop-e2f06eab496b
- https://colah.github.io/posts/2015-08-Backprop/
- http://blog.ezyang.com/2019/05/pytorch-internals/
- https://horace.io/brrr_intro.html

**Karpathy videos**
- https://github.com/karpathy/micrograd
- https://www.youtube.com/watch?v=VMj-3S1tku0 (micrograd)
- https://www.youtube.com/watch?v=P6sfmUTpUmc (makemore Part 3 — BatchNorm)
- https://www.youtube.com/watch?v=kCc8FmEb1nY (GPT from scratch)

**Papers (in order)**
- Attention Is All You Need: https://arxiv.org/abs/1706.03762
- Flash Attention: https://arxiv.org/abs/2205.14135
- Flash Attention 2: https://arxiv.org/abs/2307.08691
- Scaling Laws: https://arxiv.org/abs/2001.08361
- Chinchilla: https://arxiv.org/abs/2203.15556
- PyTorch DDP: https://arxiv.org/abs/2006.15704
- Megatron-LM: https://arxiv.org/abs/1909.08053
- ZeRO: https://arxiv.org/abs/1910.02054
- LoRA: https://arxiv.org/abs/2106.09685
- QLoRA: https://arxiv.org/abs/2305.14314

**Blogs**
- https://ruder.io/optimizing-gradient-descent/
- https://jalammar.github.io/illustrated-transformer/
- https://lilianweng.github.io/posts/2023-01-27-the-transformer-family-v2/
- https://www.lesswrong.com/posts/6Fpvch8RR29qLEWNH/chinchilla-s-wild-implications

**Infrastructure**
- https://lambdalabs.com/ (GPU rental)
- https://vast.ai/ (cheaper GPU rental)
- https://wandb.ai/ (experiment tracking — use this from week 3 onward)
- https://triton-lang.org/main/getting-started/tutorials/index.html

---

*Total papers: ~12 core + ~6 supplementary*  
*Total projects: 6 (profiling, mixed precision, scaling laws, distributed training, distributed profiling, flagship)*  
*Total blog posts: 5*  
*Timeline: 16 weeks*
