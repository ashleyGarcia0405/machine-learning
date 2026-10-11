# Detailed Week-by-Week Study Plan
## Anthropic Fellows — Reinforcement Learning Track

**Daily time budget:** ~4–5 hrs/day  
**Split each day roughly:** 2.5 hrs build · 1.5 hrs read · 1 hr write/reflect  
**Starting point:** PyTorch basics, statistical ML theory through ~lecture 03, SWE background

---

## What Anthropic actually wants from an RL candidate

The RL workstream at Anthropic isn't about playing Atari games. It's about:
- **RLHF** (Reinforcement Learning from Human Feedback) — how Claude is trained
- **Scalable oversight** — using RL to make models easier to supervise
- **Policy generalization** — do trained behaviors hold up out-of-distribution?
- **Reward hacking** — what happens when the reward model is wrong?
- **Training stability** — why PPO is finicky at scale, how to fix it

The through-line is: RL as a tool for aligning model behavior, not RL as a game-playing benchmark. Keep that lens on everything you read and build.

---

## Where you actually stand

**You can skip:** Basic probability, basic Python/PyTorch, intro neural nets  
**You need to build:** RL-specific math (Bellman equations, policy gradients, importance sampling), environment interaction loop, PPO from scratch  
**You already have:** SWE instincts — use them. RL codebases are notoriously messy. Clean, well-tested implementations will distinguish you immediately.

---

## PHASE 1: RL FOUNDATIONS (Weeks 1–3)

### Week 1 — MDPs, Value Functions, Q-Learning

**Goal:** Get the mathematical foundations locked in through implementation, not just reading.

---

**Monday**
- Read: [Sutton & Barto Chapter 1](http://incompleteideas.net/book/the-book-2nd.html) — free online. The whole chapter, ~1 hr. Get the vocabulary: agent, environment, reward, return, policy.
- Read: [Sutton & Barto Chapter 3](http://incompleteideas.net/book/the-book-2nd.html) — **Sections 3.1–3.5**. MDPs, the Bellman equation for V and Q. Read slowly. The Bellman equation is the foundation of everything.
- Done when: You can write the Bellman optimality equation for both V* and Q* from memory, and explain in words what each term means.

**Tuesday**
- Read: [Sutton & Barto Chapter 4](http://incompleteideas.net/book/the-book-2nd.html) — **Sections 4.1–4.4**. Dynamic programming: policy evaluation, policy improvement, policy iteration, value iteration.
- Code: Implement value iteration and policy iteration on a toy gridworld (5x5, some walls, one goal). No libraries — just numpy.
  ```python
  # Grid: states are (row, col)
  # Actions: up, down, left, right
  # Reward: -1 per step, +10 at goal
  # Implement: value_iteration(grid, gamma=0.99) -> V, pi
  ```
- Done when: Your value iteration converges and the resulting policy takes the shortest path to the goal.

**Wednesday**
- Read: [Sutton & Barto Chapter 6](http://incompleteideas.net/book/the-book-2nd.html) — **Sections 6.1–6.5**. TD learning: TD(0), SARSA, Q-learning. Understand the difference between on-policy (SARSA) and off-policy (Q-learning).
- Code: Implement Q-learning (tabular) on the gridworld. Then try FrozenLake from Gymnasium:
  ```bash
  pip install gymnasium
  ```
  ```python
  import gymnasium as gym
  env = gym.make("FrozenLake-v1", is_slippery=False)
  # Implement tabular Q-learning
  # Plot: cumulative reward vs. episode number
  ```
- Done when: Your Q-learning agent solves FrozenLake consistently (>90% success rate after training).

**Thursday**
- Read: [Playing Atari with Deep Reinforcement Learning (Mnih et al., 2013)](https://arxiv.org/abs/1312.5602) — **Sections 1–4**. DQN: the key innovations (experience replay, target network). Understand *why* each is needed — what instability they prevent.
- Code: Start implementing DQN. Today: just the replay buffer and the network architecture.
  ```python
  class ReplayBuffer:
      def __init__(self, capacity):
          ...
      def push(self, state, action, reward, next_state, done):
          ...
      def sample(self, batch_size):
          ...
  ```

**Friday**
- Code: Finish DQN. Add: target network with periodic copy, epsilon-greedy exploration schedule, training loop.
- Train on CartPole-v1 first (simpler, faster feedback than Atari):
  ```python
  env = gym.make("CartPole-v1")
  ```
- Plot: episode return vs. training step, epsilon vs. training step.
- Done when: DQN solves CartPole (average return >475 over 100 episodes).
- Experiment: Remove experience replay. Remove target network. What breaks?

**Weekend**
- Read: [Human-level control through deep reinforcement learning (Mnih et al., 2015)](https://www.nature.com/articles/nature14236) — the full DQN Nature paper (~1.5 hrs). Focus on: the tricks that make it stable, the evaluation methodology.
- Code: Clean up your DQN, add logging (use wandb — you'll use it all plan), push to GitHub repo `rl-fundamentals`.
- Done when: DQN repo is clean, READMEs explain what each file does, wandb dashboard shows training curves.

---

### Week 2 — Policy Gradients

**Goal:** Understand policy gradient methods from first principles. This is the math backbone of PPO.

---

**Monday**
- Read: [Sutton & Barto Chapter 13](http://incompleteideas.net/book/the-book-2nd.html) — **Sections 13.1–13.4**. The policy gradient theorem. Understand the derivation — not just the result. Why do we differentiate through the log probability?
- Read: [Policy Gradient Algorithms — Lilian Weng](https://lilianweng.github.io/posts/2018-04-08-policy-gradient/) — **through the REINFORCE section** (~45 min). Best practical summary of the math.
- Done when: You can derive the REINFORCE gradient estimator from scratch on paper.

**Tuesday**
- Code: Implement REINFORCE from scratch on CartPole:
  ```python
  # Policy: small MLP that outputs action probabilities
  # Collect full episode, compute returns G_t
  # Loss: -sum(log pi(a|s) * G_t)
  # Baseline: try with and without a baseline (subtract mean return)
  ```
- Plot: training curves with and without baseline. How much does variance reduction matter?
- Done when: REINFORCE solves CartPole. You understand *why* the variance is so high.

**Wednesday**
- Read: The variance reduction section of [Lilian Weng's policy gradient post](https://lilianweng.github.io/posts/2018-04-08-policy-gradient/) (~30 min)
- Read: [Advantage estimation — GAE (Schulman et al., 2015)](https://arxiv.org/abs/1506.02438) — **Sections 1–3**. The key idea: use a learned value function as baseline, interpolate between MC and TD with λ.
- Done when: You can explain the bias-variance tradeoff in GAE and what λ=0 vs. λ=1 means.

**Thursday**
- Code: Implement Actor-Critic with GAE:
  - Actor: policy network outputting action probabilities
  - Critic: value network outputting V(s)
  - Advantage: A_t = G_t - V(s_t), or GAE with λ
  - Train simultaneously on CartPole, then LunarLander-v2
- Done when: Actor-Critic is more sample-efficient than REINFORCE on the same task.

**Friday**
- Read: [Deterministic Policy Gradient (Silver et al., 2014)](https://proceedings.mlr.press/v32/silver14.pdf) — **Sections 1–3** (~45 min). Understand when deterministic policies make sense (continuous action spaces).
- Code: Implement DDPG (Deep Deterministic Policy Gradient) on Pendulum-v1 (continuous actions):
  - Target actor + critic networks
  - Ornstein-Uhlenbeck noise for exploration
  - Soft updates (polyak averaging) for targets
- Done when: DDPG stabilizes the pendulum.

**Weekend**
- Read: [TD3 (Fujimoto et al., 2018)](https://arxiv.org/abs/1802.09477) — **Sections 1–3**. What instabilities does DDPG have and how does TD3 fix them? (Clipped double Q, delayed policy updates, target policy smoothing)
- Code: Add the TD3 fixes to your DDPG. Does it train more stably?
- Write: 400-word journal entry — "The four main RL families (DP, value-based, policy gradient, actor-critic) and when you'd use each." Use your own implementations as evidence.

---

### Week 3 — Trust Regions and PPO

**Goal:** Implement PPO from scratch. This is the most important algorithm for the Anthropic path.

---

**Monday**
- Read: [Trust Region Policy Optimization (Schulman et al., 2015)](https://arxiv.org/abs/1502.05477) — **Sections 1–4**. TRPO motivates PPO. The key idea: constrain policy updates so you don't step too far. Understand: the surrogate objective, the KL constraint, why natural gradients.
- Don't implement TRPO (it's complex and mostly superseded). Just understand the *problem* it's solving.
- Done when: You can explain in one paragraph why unconstrained policy gradient methods can "fall off a cliff."

**Tuesday**
- Read: [Proximal Policy Optimization (Schulman et al., 2017)](https://arxiv.org/abs/1707.06347) — **full paper** (~1.5 hrs). This is the key paper. Read it carefully:
  - Section 2: the clipped surrogate objective — understand each case of the clip
  - Section 3: the full algorithm (PPO-Clip)
  - Section 4: compare to PPO-KL penalty variant
  - Figure 1: the intuition for why clipping works
- Done when: You can explain the clipped objective without looking at anything.

**Wednesday**
- Code: Implement PPO-Clip from scratch. No stable-baselines, no tianshou — just PyTorch.
  ```python
  # Collect T timesteps of experience
  # Compute advantages with GAE
  # For K epochs:
  #   Sample minibatches
  #   Compute clipped surrogate loss
  #   Compute value loss
  #   Compute entropy bonus
  #   Update both actor and critic
  ```
  Start with CartPole to verify correctness.

**Thursday**
- Code: Move your PPO to more challenging environments:
  - LunarLander-v2 (discrete)
  - BipedalWalker-v3 (continuous — switch to continuous action distribution)
  - MuJoCo HalfCheetah-v4 if you have a license (else use PyBullet's free version)
- Plot: learning curves. Does it match published PPO performance?
- Known pitfalls to debug:
  - Value loss overwhelming policy loss (use separate loss scaling)
  - Entropy collapsing too fast (tune entropy coefficient)
  - Advantage normalization: normalize per minibatch, not per rollout

**Friday**
- Code: Add all the implementation details that matter in practice:
  - Orthogonal initialization
  - Learning rate annealing
  - Reward normalization (running mean/std)
  - Observation normalization
  - Value function clipping
- Read: [Implementation Matters in Deep Policy Gradients (Engstrom et al., 2020)](https://arxiv.org/abs/2005.12729) — this paper shows that most PPO "improvements" in the literature are actually just implementation details. Read the full paper (~1.5 hrs).
- Done when: You can list the 9 "implementation details" from the Engstrom paper and explain which ones matter most.

**Weekend**
- Code: Benchmark your PPO against Stable Baselines 3's PPO on the same environments. Do they match? If not, debug.
- Read: [The 37 Implementation Details of PPO — Costa Huang](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/) — this is the most important practical reference for PPO. Bookmark it. Read all 37 details.
- Write: Blog post 1 — "I implemented PPO from scratch. Here are the 5 details that actually matter."
  Use your own experiments as evidence. This is a high-signal post because most PPO tutorials are wrong about what matters.

---

## PHASE 2: DEEP RL + RLHF (Weeks 4–8)

### Week 4 — Exploration, Reward Shaping, Environment Design

**Goal:** Understand the parts of RL that become critical when you move from toy envs to language models.

---

**Monday**
- Read: [Sutton & Barto Chapter 2](http://incompleteideas.net/book/the-book-2nd.html) — **Sections 2.1–2.7** (multi-armed bandits). This is the cleanest setting to understand exploration-exploitation.
- Code: Implement epsilon-greedy, UCB1, and Thompson sampling on a 10-armed bandit. Plot regret curves for each.
- Done when: You can explain *why* UCB1 works better than epsilon-greedy in terms of the uncertainty estimate.

**Tuesday**
- Read: [Curiosity-driven Exploration (Pathak et al., 2017)](https://arxiv.org/abs/1705.05363) — **Sections 1–3**. Intrinsic motivation: reward the agent for visiting novel states. The key: use a learned forward model to measure surprise.
- Read: [Random Network Distillation (Burda et al., 2018)](https://arxiv.org/abs/1810.12894) — **Sections 1–3**. Simpler: novelty = prediction error of a random target network.
- Code: Add RND-based intrinsic reward to your PPO. Test on MontezumaRevenge or a hard exploration gridworld.

**Wednesday**
- Read: [Reward shaping and potential functions — Ng et al., 1999](https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/NgHaradaRussell-reward-shaping-ICML99.pdf) — **Sections 1–3**. The formal result: when does reward shaping preserve the optimal policy?
- Read: [Concrete Problems in AI Safety (Amodei et al., 2016)](https://arxiv.org/abs/1606.06565) — **Sections 1–4**. Written by Anthropic founders. The reward hacking section is directly relevant to RLHF. Read it carefully.
- Done when: You can explain "specification gaming" and give 3 examples of reward hacking in RL.

**Thursday**
- Read: [Farama Foundation Gymnasium docs](https://gymnasium.farama.org/) — the custom environment tutorial
- Code: Build a custom Gymnasium environment:
  ```python
  class TextNavigationEnv(gym.Env):
      # Simple: agent navigates a text-based maze
      # State: current room description (string → embedding)
      # Actions: north, south, east, west
      # Reward: +1 for reaching goal, -0.01 per step
  ```
  This exercises the interface you'll use when environments become language-based.

**Friday**
- Code: Implement a simple bandit version of RLHF:
  ```python
  # "Reward model": a classifier trained on preference pairs
  # Policy: a small network that generates text (or actions)
  # Training loop:
  #   1. Generate 2 outputs from policy
  #   2. "Human" (ground truth function) picks preferred one
  #   3. Update reward model on the preference
  #   4. Update policy with RL using reward model
  ```
  This is a minimal RLHF loop. It won't scale, but it will teach you the mechanics.
- Done when: Your reward model learns a preference function and the policy improves against it.

**Weekend**
- Read: [Reward Modeling for Mitigating Overoptimization (Gao et al., 2022)](https://arxiv.org/abs/2210.10760) — **Sections 1–3**. Key result: as you optimize more against a reward model, performance eventually degrades. The reward model is an imperfect proxy. This is central to Anthropic's work.
- Write: 300-word journal entry — "What is Goodhart's Law and why does it appear in RLHF?"

---

### Week 5 — Language Models as RL Policies

**Goal:** Understand how RL applies to LLMs specifically. This is the core intellectual move for the Anthropic path.

---

**Monday**
- Read: [Learning to Summarize with Human Feedback (Stiennon et al., 2020)](https://arxiv.org/abs/2009.01325) — **Sections 1–4**. This is the paper that operationalized RLHF for language models. OpenAI + what became Anthropic worked on this. Read it carefully:
  - The SFT → reward model → RL pipeline
  - How they collect human preference data
  - The KL penalty term in the reward function (why it's there)
- Done when: You can draw the full training pipeline and explain the role of each component.

**Tuesday**
- Read: [Training Language Models to Follow Instructions with Human Feedback (InstructGPT, Ouyang et al., 2022)](https://arxiv.org/abs/2203.02155) — **Sections 1–4**. This is InstructGPT — the direct predecessor to ChatGPT and Claude's training methodology.
  - Focus on: Section 3.3 (the RL objective with KL penalty), Figure 2 (the pipeline)
  - Key formula: `R(x,y) = r_θ(x,y) − β log(π_RL(y|x) / π_SFT(y|x))`
  - Understand every term in this formula.
- Done when: You can explain why the KL penalty term is necessary and what happens if you remove it.

**Wednesday**
- Code: Build the KL-penalized RL objective in PyTorch:
  ```python
  def rlhf_reward(reward_model_score, policy_logprobs, ref_logprobs, beta=0.1):
      kl_penalty = (policy_logprobs - ref_logprobs).sum(-1)
      return reward_model_score - beta * kl_penalty
  ```
  Test: vary beta. What happens to the policy when beta=0? When beta=10?
- Understand: the reference policy is frozen SFT model. Why do we need it?

**Thursday**
- Read: [Constitutional AI (Bai et al., 2022)](https://arxiv.org/abs/2212.08073) — **Sections 1–3**. This is Anthropic's paper. The key idea: replace human feedback with AI feedback (RLAIF). The constitution is a set of principles the AI critique uses.
- Read: [RLAIF: Scaling Reinforcement Learning from Human Feedback with AI Feedback (Lee et al., 2023)](https://arxiv.org/abs/2309.00267) — **Sections 1–3** (~45 min).
- Done when: You can explain the difference between RLHF and RLAIF and when you'd prefer each.

**Friday**
- Read: [Direct Preference Optimization (Rafailov et al., 2023)](https://arxiv.org/abs/2305.18290) — **Sections 1–3**. DPO reformulates RLHF without explicit RL — you optimize directly on the preference data. This is increasingly replacing PPO-based RLHF.
- Code: Implement the DPO loss:
  ```python
  def dpo_loss(policy_chosen_logps, policy_rejected_logps,
               ref_chosen_logps, ref_rejected_logps, beta=0.1):
      pi_logratios = policy_chosen_logps - policy_rejected_logps
      ref_logratios = ref_chosen_logps - ref_rejected_logps
      logits = pi_logratios - ref_logratios
      loss = -F.logsigmoid(beta * logits).mean()
      return loss
  ```
  Make sure you understand *why* this is equivalent to RLHF.
- Done when: You can explain the DPO derivation at a high level — how does optimizing this loss correspond to maximizing reward minus KL?

**Weekend**
- Read: [A General Language Assistant as a Laboratory for Alignment (Bai et al., 2022)](https://arxiv.org/abs/2204.05862) — Anthropic's "helpful, harmless, honest" paper. **Sections 1–3**. Understand the tradeoff between helpfulness and harmlessness in RLHF.
- Write: Blog post 2 — "The RLHF pipeline, explained from first principles"
  Cover: SFT → reward model → PPO with KL penalty. Include the key formula. This post is high-value because it shows you understand Anthropic's core training methodology.

---

### Week 6 — Reward Modeling in Depth

**Goal:** Understand reward models — how they're trained, what makes them fail, and how to evaluate them.

---

**Monday**
- Read: [Reward Model Ensembles (Coste et al., 2023)](https://arxiv.org/abs/2310.02743) — **Sections 1–3**. The problem: reward models have uncertainty. Using an ensemble gives calibrated uncertainty estimates.
- Read: [Scaling Laws for Reward Model Overoptimization (Gao et al., 2022)](https://arxiv.org/abs/2210.10760) — **full paper**. The key result: the KL distance between the RL policy and the reference policy is the right measure of "how much you've optimized." Too much optimization → reward hacking.

**Tuesday**
- Code: Train a reward model from scratch on a real preference dataset.
  - Use [Anthropic's HH-RLHF dataset](https://huggingface.co/datasets/Anthropic/hh-rlhf) — this is real preference data from Anthropic
  - Model: take a pretrained LM (GPT-2 or similar), add a scalar head
  - Train with Bradley-Terry loss:
    ```python
    def reward_model_loss(r_chosen, r_rejected):
        return -F.logsigmoid(r_chosen - r_rejected).mean()
    ```
  - Evaluate: preference accuracy on a held-out test set

**Wednesday**
- Experiment: How does reward model quality affect downstream RL?
  - Train two reward models: one with 1k preference pairs, one with 10k
  - Use each as the reward signal for PPO on a toy generation task
  - Measure: does the policy optimize differently against each?
  - Plot: the Gao et al. overoptimization curve — KL vs. reward vs. ground truth quality

**Thursday**
- Read: [Calibration in Deep Learning — Guo et al., 2017](https://arxiv.org/abs/1706.04599) — **Sections 1–3**. Reward models need to be calibrated to give meaningful uncertainty estimates.
- Code: Evaluate your reward model's calibration:
  - Plot a reliability diagram
  - Apply temperature scaling
  - Does calibration matter for downstream RL? (Experiment: compare calibrated vs. uncalibrated reward in the RL loop)

**Friday**
- Read: [Debate as an Alignment Strategy (Irving et al., 2018)](https://arxiv.org/abs/1805.00899) — **Sections 1–3**. Alternative to reward modeling: have two agents debate, human judges who wins.
- Read: [Scalable Oversight (Bowman et al., 2022)](https://arxiv.org/abs/2211.03540) — **Sections 1–3**. The core problem: as models get smarter, human evaluators can't reliably judge them. What do we do?
- Done when: You can explain the scalable oversight problem and three proposed approaches (debate, amplification, process reward models).

**Weekend**
- Read: [Let's Verify Step by Step — Process Reward Models (Lightman et al., 2023)](https://arxiv.org/abs/2305.20050) — **Sections 1–4**. Outcome reward models vs. process reward models. This is directly relevant to Anthropic's reasoning work.
- Write: 500-word technical note — "Outcome rewards vs. process rewards: the tradeoffs"

---

### Week 7 — PPO at Scale + Training Stability

**Goal:** Understand why PPO is hard to scale and what the failure modes are.

---

**Monday**
- Read: [Proximal Policy Optimization Algorithms (Schulman et al., 2017)](https://arxiv.org/abs/1707.06347) — re-read Section 5 (experiments) now that you have context
- Read: [What Matters for On-Policy Deep Actor-Critic Methods (Andrychowicz et al., 2021)](https://arxiv.org/abs/2006.05990) — **Sections 1–4**. 50+ ablations of what actually matters in PPO training.
- Done when: You can list the top 5 most impactful hyperparameters/choices from the Andrychowicz paper.

**Tuesday**
- Code: Systematic PPO ablations on your implementation:
  - Vary: clip epsilon (0.1, 0.2, 0.3), epochs per rollout (1, 4, 10, 20), GAE lambda (0.9, 0.95, 0.99, 1.0), entropy coefficient (0, 0.01, 0.05)
  - Use a fixed environment (HalfCheetah or similar)
  - Log everything to wandb
  - Build a heatmap: hyperparameter × metric

**Wednesday**
- Read: [Recurrent PPO for Partial Observability — Hausknecht & Stone, 2015](https://arxiv.org/abs/1507.06527) — **Sections 1–3**. What changes when the environment is partially observable?
- Code: Implement PPO with an LSTM policy. Test on a POMDP version of a simple environment (mask the velocity observations in CartPole).

**Thursday**
- Code: Vectorized environments — the standard way to speed up environment interaction:
  ```python
  env = gym.vector.make("CartPole-v1", num_envs=16)
  ```
  This collects 16x more experience per step. Measure the throughput improvement.
- Implement: proper handling of episode resets in vectorized envs (done flags, per-env return tracking)

**Friday**
- Read: [IMPALA (Espeholt et al., 2018)](https://arxiv.org/abs/1802.01561) — **Sections 1–3**. Distributed RL: many actors, one learner. The V-trace off-policy correction for importance sampling.
- Read: [Sample Factory (Petrenko et al., 2020)](https://arxiv.org/abs/2006.11751) — **Sections 1–2**. High-throughput PPO implementation — how to get 100k steps/sec on a single machine.
- Done when: You understand the importance sampling correction and why you need it when actors and learner run asynchronously.

**Weekend**
- Code: Add distributed collection to your PPO — even a simple version where multiple Python processes collect experience and send it to a central buffer.
- Write: Blog post 3 — "PPO implementation details: what the papers don't tell you"
  This is your most technical post so far. Reference the Engstrom paper, your ablations, and the 37 details post.

---

### Week 8 — RLHF Pipeline End to End

**Goal:** Run a complete SFT → reward model → PPO fine-tuning pipeline on a real language model.

---

**Monday–Tuesday**
- Code: Full RLHF pipeline on a small language model (GPT-2 or a 125M model):
  **Step 1: SFT**
  - Fine-tune GPT-2 on a subset of [Alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca) or similar instruction-following dataset
  - Standard cross-entropy loss on completions
  - Train for a few epochs — you just want a reasonable SFT baseline

  **Step 2: Reward model**
  - Use [Anthropic/hh-rlhf](https://huggingface.co/datasets/Anthropic/hh-rlhf) preference data
  - Load your SFT model, replace the language model head with a scalar head
  - Train with Bradley-Terry loss
  - Evaluate: preference accuracy should be >60% on test set

**Wednesday–Thursday**
- Code: **Step 3: PPO fine-tuning**
  - Policy: your SFT model
  - Reference model: frozen copy of SFT model (for KL penalty)
  - Reward: `r_θ(x,y) - β * KL(π_RL || π_SFT)`
  - PPO update on the combined reward
  
  This is the hardest part. Key implementation issues:
  - Handling variable-length sequences (pack or pad?)
  - Computing per-token logprobs for KL penalty
  - Value function for a language model (share backbone or separate?)
  - The "reward at EOS only" vs. "reward at every token" choice

**Friday**
- Evaluation: compare SFT baseline vs. RLHF fine-tuned model
  - Generate completions on a held-out set of prompts
  - Have your reward model score them
  - Manual inspection: do the RLHF completions seem better?
- Plot: the KL-reward frontier as you vary β

**Weekend**
- Read: [TRL library by HuggingFace](https://github.com/huggingface/trl) — read the PPOTrainer source code. How does their implementation differ from yours? What did you miss?
- Write: 500-word post-mortem — "What I got wrong building my first RLHF pipeline"
  This kind of honest reflection is what researchers respect. Publish this.

---

## PHASE 3: RESEARCH TOPICS (Weeks 9–11)

### Week 9 — Generalization and Distribution Shift

**Goal:** Understand when RL policies fail out-of-distribution — a key research question at Anthropic.

---

**Monday**
- Read: [Quantifying Generalization in Reinforcement Learning (Cobbe et al., 2019)](https://arxiv.org/abs/1812.02341) — **Sections 1–4**. Introduced the CoinRun benchmark specifically to measure generalization. Key finding: standard RL training overfits to the training environment.
- Read: [Leveraging Procedural Generation to Benchmark Reinforcement Learning (Cobbe et al., 2020)](https://arxiv.org/abs/1912.01588) — **Sections 1–3**. ProcGen benchmark.

**Tuesday**
- Code: Install and run ProcGen:
  ```bash
  pip install procgen
  ```
  Train PPO on CoinRun with 200 training levels. Test on test levels. Measure the generalization gap.
  - Vary: number of training levels (16, 200, 500, unlimited). Plot train vs. test return.
  - This is the generalization curve. Understand what it means.

**Wednesday**
- Read: [Dropout as Data Augmentation (Srivastava et al., 2014)](https://arxiv.org/abs/1207.0580) — just Section 2 (the technique itself). Then:
- Read: [Regularization for RL (Cobbe et al., 2019)](https://arxiv.org/abs/1912.02975) — **Sections 1–3**. Does L2 reg, dropout, data augmentation help RL generalization?
- Code: Apply dropout and data augmentation (random cropping via [DrAC](https://github.com/rraileanu/auto-drac)) to your PPO on ProcGen. Does it help?

**Thursday**
- Read: [DRIVE: One-step Sound Extrapolation by Diversifying Inputs (Farebrother et al., 2023)](https://arxiv.org/abs/2306.09205) — just the abstract and intro. This is recent work on what makes RL generalize.
- Experiment: Train a policy on CoinRun. Then change: (1) background colors, (2) character appearance, (3) coin position distribution. Which changes hurt performance most? Why?

**Friday**
- Read: [Reward Tampering Problems and Solutions (Everitt et al., 2021)](https://arxiv.org/abs/1908.04734) — **Sections 1–3**. A different generalization failure: agents that modify their own reward function.
- Write: 400-word journal — "What is the difference between overfitting in supervised learning and generalization failure in RL?"

**Weekend**
- Read: [Goal Misgeneralization in Deep RL (Langosco et al., 2022)](https://arxiv.org/abs/2105.14111) — **Sections 1–3**. The agent pursues a goal that correlates with the reward during training but diverges at test time. This is directly connected to alignment.
- Done when: You can give an example of goal misgeneralization from the paper and explain why it's an alignment concern.

---

### Week 10 — Reward Hacking and Specification Gaming

**Goal:** Go deep on the ways reward optimization goes wrong. This is Anthropic's core research concern.

---

**Monday**
- Read: [Specification Gaming: the Flip Side of AI Ingenuity (Krakovna et al., 2020)](https://deepmind.google/discover/blog/specification-gaming-the-flip-side-of-ai-ingenuity/) — the full DeepMind post with all examples (~45 min). Build intuition for the many ways agents exploit reward functions.
- Build: your own "specification gaming" example. Design a simple environment where a naive reward is gameable, train an agent on it, and document the exploit.

**Tuesday**
- Read: [Risks from Learned Optimization (Hubinger et al., 2019)](https://arxiv.org/abs/1906.01820) — **Sections 1–3**. The mesa-optimization problem: an agent might learn an internal optimizer that pursues a different goal than the training reward. This is one of the core concerns in AI safety.
- Read: [Towards Guaranteed Safe AI (Dalrymple et al., 2024)](https://arxiv.org/abs/2405.06624) — **just the abstract and introduction** (~20 min). Recent Anthropic-adjacent work on formal safety guarantees.

**Wednesday**
- Code: Reward hacking experiment:
  - Train an RLHF reward model on a limited dataset
  - Use PPO to maximize reward model score for many gradient steps
  - Track: reward model score vs. actual quality (use a separate oracle or human eval)
  - Find the point where they diverge
  - Plot the Gao et al. curve from your own data
- This experiment is publishable if you do it carefully.

**Thursday–Friday**
- Read: [Avoiding Side Effects in Complex Environments (Turner et al., 2020)](https://arxiv.org/abs/2006.06547) — **Sections 1–3**. Penalizing agents for changing things they shouldn't.
- Read: [Conservative Agency (Krakovna et al., 2023)](https://arxiv.org/abs/2210.13871) — **Sections 1–3**. How to make agents avoid unintended side effects.
- Write: Blog post 4 — "Reward hacking: a taxonomy with examples from my own experiments"
  Use your specification gaming example and your RLHF overoptimization experiment. This is original, technically grounded, and directly Anthropic-relevant.

**Weekend**
- Read: [Agent Foundations for Aligning Machine Intelligence with Human Interests — Paul Christiano](https://paulfchristiano.com/ea/) — Paul's informal writeup on alignment. (~1 hr) Paul Christiano is a key figure in the research Anthropic's RL work builds on.
- Deep-read the reward hacking section of the Concrete Problems paper (Amodei et al.) again. You now have the technical background to understand every example.

---

### Week 11 — Scalable Oversight and Process Reward Models

**Goal:** Understand the research direction Anthropic is most actively pursuing — making human oversight scale.

---

**Monday**
- Read: [Scalable Agent Alignment via Reward Modeling (Leike et al., 2018)](https://arxiv.org/abs/1811.07871) — **Sections 1–4**. The iterated amplification framework.
- Read: [AI Safety via Debate (Irving et al., 2018)](https://arxiv.org/abs/1805.00899) — **full paper** (~1 hr). Understand the game-theoretic argument for why debate works.
- Done when: You can explain the difference between debate, amplification, and process reward models as approaches to scalable oversight.

**Tuesday**
- Read: [Let's Reward Step by Step (Lightman et al., 2023)](https://arxiv.org/abs/2305.20050) — **full paper**. Process reward models (PRMs) score each reasoning step, not just the final answer. Key result: PRMs dramatically outperform outcome reward models on math.
- Code: Implement a toy PRM:
  - Task: multi-step arithmetic (e.g., "compute 17 × 23 + 45 step by step")
  - Outcome reward model: correct final answer = 1, else 0
  - Process reward model: each correct intermediate step = +0.2
  - Compare: which one produces better reasoning chains when used as RL reward?

**Wednesday**
- Read: [Improving Alignment of Dialogue Agents via Targeted Human Judgements (Glaese et al., 2022)](https://arxiv.org/abs/2209.14375) — **Sections 1–4**. Sparrow from DeepMind — a competing approach to InstructGPT with rule-based reward models.
- Synthesize: write a comparison document (for your own notes):
  - InstructGPT, Constitutional AI, Sparrow, and DPO — what are the key differences?

**Thursday–Friday**
- Code: Implement a toy scalable oversight setup:
  - Task: summarization with a long document that the "human" can't fully read
  - Weak supervisor: can only see a 100-token summary
  - Strong model: can see the full document
  - Use debate: two models write arguments pro/con a proposed summary, weak supervisor picks winner
  - Measure: does debate help the weak supervisor make better judgments than without it?
- This is pedagogical but teaches you the key experimental paradigm.

**Weekend**
- Review weeks 9–11. You should now be fluent in: generalization failure, reward hacking/specification gaming, scalable oversight approaches.
- Start thinking about your flagship project. Write a half-page of rough ideas.

---

## PHASE 4: FLAGSHIP PROJECT (Weeks 12–16)

### Week 12 — Choose + Scope

**Goal:** Pick a project with a tight scope, real experiments, and a non-obvious finding.

---

**Monday**
Decide on your flagship. Options:

**Option A: RLHF Overoptimization Study**
Reproduce and extend Gao et al. (2022). The KL-reward tradeoff under different conditions:
- How does overoptimization scale with reward model size?
- Does the overoptimization curve change with dataset quality?
- Does DPO exhibit the same overoptimization as PPO-based RLHF?

This is the strongest choice for Anthropic. It's directly relevant to their training methodology, you already have the infrastructure from week 8, and there are clear open questions.

**Option B: Goal Misgeneralization Empirical Study**
Reproduce Langosco et al. (2022) and run your own variations. When does goal misgeneralization appear, and what training interventions prevent it?

**Option C: Process Reward Model Ablation**
Replicate the Lightman et al. PRM result at smaller scale, then ablate: how does PRM quality scale with labeled step data? Is there a minimum viable PRM?

**Option D: DPO vs. PPO Detailed Comparison**
Both methods claim to do RLHF but work very differently. Careful head-to-head on the same task, same data. What are the actual tradeoffs? When does one dominate?

---

**Tuesday–Wednesday**
- Write a 1-page project spec (same format as ML Systems plan):
  - Problem statement (1 sentence)
  - What you'll build
  - 4+ planned experiments with expected outcomes
  - How you'll evaluate
  - What the writeup will claim
  
- Post the spec publicly (Twitter/X, or a blog preview). Getting feedback before you build saves weeks.

**Thursday–Friday**
- Set up project infrastructure:
  - New GitHub repo: `rlhf-study` (or your project name)
  - wandb project set up, logging skeleton in place
  - Baseline run working end-to-end before any ablations
  - README with project goals written
- Done when: end-to-end pipeline runs without errors, baseline result is logged.

---

### Weeks 13–14 — Build

Below is the spec for Option A (RLHF Overoptimization Study). Adapt to your choice.

**Week 13: Reproduce the baseline**

Monday–Wednesday:
- Reproduce the Gao et al. KL-reward curve on your setup:
  - Train a reward model on Anthropic HH-RLHF (or another preference dataset)
  - Use PPO to optimize against it with varying KL penalties
  - Track: reward model score, KL from reference, "ground truth" quality (use a held-out reward model as proxy oracle)
  - Produce Figure 1 analog from the Gao paper

Thursday–Friday:
- First extension: compare DPO overoptimization:
  - Train the same base model with DPO instead of PPO
  - For DPO, "optimization intensity" = training steps or learning rate
  - Does the same KL-quality curve appear?
  - Key question: does DPO overoptimize in the same way?

**Week 14: Ablations**

- Reward model size effect: train reward models of different sizes (50M, 125M, 350M params). How does overoptimization scale with RM size?
- Dataset quality effect: subsample your preference data randomly vs. by uncertainty. Does higher-quality preference data shift the curve?
- Ensemble effect: use an ensemble of 3 reward models. Does the overoptimization curve shift?
- Document everything carefully — uncertainty matters here.

---

### Week 15 — Write It Up

Same structure as the ML Systems plan. Write a short paper, not a blog post.

**Structure:**
1. **Introduction** (~300 words): Why overoptimization matters, what you studied, what you found
2. **Background** (~400 words): RLHF pipeline, KL penalty, prior work on overoptimization
3. **Experimental Setup** (~300 words): Model, dataset, hardware, metrics
4. **Results** (~600 words + figures): Your curves. Be precise about the numbers.
5. **Discussion** (~300 words): What this means for practitioners, open questions
6. **Conclusion** (~150 words)

**Figures:**
- KL-reward curve (main result)
- DPO vs. PPO comparison
- Reward model size scaling
- Dataset quality effect

**Monday–Wednesday:** Write the full draft  
**Thursday:** Revise  
**Friday:** Polish and publish — as a blog post that mirrors paper structure

---

### Week 16 — Application

Same as ML Systems plan. Review public outputs, write the application essay, submit.

**Application essay framing for RL path:**

> "I came from software engineering — I knew how to build systems, but I didn't know how to build systems that *align with human intent at scale*. Over the past four months I went deep on RL and RLHF: I implemented PPO from scratch, built a full RLHF pipeline, and ran an empirical study on reward model overoptimization that [specific finding]. What I found is that [non-obvious insight from your flagship]. I want to work at Anthropic because the RLHF training process has failure modes we don't fully understand yet, and I want to be the person figuring them out."

---

## Checkpoints

| Week | You should be able to... |
|------|--------------------------|
| 1 | Implement Q-learning and explain the Bellman equation without notes |
| 2 | Derive the policy gradient theorem on paper |
| 3 | Implement PPO from scratch and name the 5 most important implementation details |
| 5 | Explain the InstructGPT training pipeline and the KL penalty formula |
| 6 | Train a reward model on real preference data and evaluate its calibration |
| 8 | Run a full SFT → RM → PPO pipeline on a language model |
| 10 | Give 5 distinct examples of reward hacking / specification gaming from your own experiments |
| 11 | Explain debate, amplification, and PRMs as scalable oversight approaches |
| 14 | State your flagship project's key finding in 2 sentences with numbers |

---

## Resource Index

**Textbook**
- Sutton & Barto (free online): http://incompleteideas.net/book/the-book-2nd.html

**Core RL papers (in order)**
- DQN (2013): https://arxiv.org/abs/1312.5602
- DQN Nature (2015): https://www.nature.com/articles/nature14236
- TRPO: https://arxiv.org/abs/1502.05477
- PPO: https://arxiv.org/abs/1707.06347
- GAE: https://arxiv.org/abs/1506.02438
- TD3: https://arxiv.org/abs/1802.09477
- IMPALA: https://arxiv.org/abs/1802.01561

**RLHF papers (in order)**
- Learning to Summarize from HF: https://arxiv.org/abs/2009.01325
- InstructGPT: https://arxiv.org/abs/2203.02155
- Constitutional AI: https://arxiv.org/abs/2212.08073
- DPO: https://arxiv.org/abs/2305.18290
- Reward Model Overoptimization: https://arxiv.org/abs/2210.10760
- Process Reward Models: https://arxiv.org/abs/2305.20050
- RLAIF: https://arxiv.org/abs/2309.00267

**Safety/alignment papers**
- Concrete Problems in AI Safety: https://arxiv.org/abs/1606.06565
- Specification Gaming (blog): https://deepmind.google/discover/blog/specification-gaming-the-flip-side-of-ai-ingenuity/
- Risks from Learned Optimization: https://arxiv.org/abs/1906.01820
- Goal Misgeneralization: https://arxiv.org/abs/2105.14111

**Practical resources**
- 37 PPO Implementation Details: https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/
- Lilian Weng Policy Gradients: https://lilianweng.github.io/posts/2018-04-08-policy-gradient/
- TRL library: https://github.com/huggingface/trl
- Anthropic HH-RLHF dataset: https://huggingface.co/datasets/Anthropic/hh-rlhf
- Gymnasium: https://gymnasium.farama.org/
- ProcGen: https://github.com/openai/procgen

---

*Total core papers: ~18 + ~10 supplementary*  
*Total projects: 6 (Q-learning/DQN, REINFORCE/PPO, reward modeling, RLHF pipeline, generalization study, flagship)*  
*Total blog posts: 4 + flagship writeup*  
*Timeline: 16 weeks*
