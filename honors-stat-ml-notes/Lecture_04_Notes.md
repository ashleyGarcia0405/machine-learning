# Lecture 4: Regularization, Kernel Methods & Nonparametric Regression

---

## 1. Regularization (Continued)

### Ridge and Lasso

We want to find a weight vector ω that minimizes the empirical risk plus a penalty term. The penalized objective is:

$$\hat{\omega}_\lambda = \underset{\omega}{\text{argmin}} \; \hat{R}(\omega) + \lambda \cdot \text{reg}(\omega)$$

Concretely, this expands to:

$$\hat{\omega}_\lambda = \underset{\omega}{\text{argmin}} \; \frac{1}{n}\sum_{i=1}^{n}(y_i - \omega^T x_i)^2 + \lambda \|\omega\|^2$$

- The **first term** is the training loss (mean squared error).
- The **second term** is a regularization penalty, for some λ > 0:
  - `+ λ||ω||²` → **Ridge regression** (L2 regularization)
  - `+ λ||ω||₁` → **Lasso** (L1 regularization)

### Why Regularize?

Without regularization, a model can fit the training data perfectly by using a wildly oscillating function (orange curve in lecture sketch). This is called **overfitting** — the model memorizes noise and fails to generalize to future data.

Regularization constrains the **set of "possible" functions** f the model can learn. This is equivalent to limiting the model's **capacity**. By penalizing large weights (large ‖ω‖), we prefer smoother, simpler functions (green curve).

> **NOTE:** The right choice of λ depends on the unknown underlying data distribution. Too small → overfitting. Too large → underfitting (too simple a model).

---

## 2. Choosing λ: Cross-Validation

### The Problem

The population risk R(ω_λ) as a function of λ has a **U-shape**:

- λ too small → model overfits → high risk
- λ too large → model is too constrained → high risk
- There is an optimal λ* in the middle

We can't observe R(ω_λ) directly (it requires knowing the true data distribution), so we estimate it.

### Cross-Validation Procedure

1. **Hold out a validation dataset** {(x'_i, y'_i)}_{i=1}^m that is **independent of training**.

2. **Compute the validation risk** for each candidate λ:

$$\hat{R}'(\hat{\omega}_\lambda) = \frac{1}{m}\sum_{i=1}^{m}(y'_i - \hat{\omega}_\lambda \cdot x'_i)^2$$

   This is an estimate of the true future risk R(ω̂_λ).

3. **Pick the best λ:**

$$\hat{\lambda} = \underset{\lambda}{\text{argmin}} \; \hat{R}'(\hat{\omega}_\lambda)$$

4. **Retrain on all data.** After selecting λ̂, often combine training + validation data and retrain to get a final estimator ω̂_λ̂:

$$\hat{\omega}_{\hat{\lambda}} = \underset{\omega}{\text{argmin}} \; \frac{1}{n+m}\sum_{i=1}^{n+m}(y_i - x_i \cdot \omega)^2 + \hat{\lambda}\|\omega\|^2$$

> **Note:** There are multiple variants of cross-validation, e.g., **K-Fold Cross-Validation**, where the data is split into K groups and the validation is repeated K times.

### Why Does This Work?

By the **Law of Large Numbers (LLN)**, as m → ∞:

$$\hat{R}'(\omega) \xrightarrow{\text{w.h.p.}} R(\omega)$$

The rate of convergence is of order 1/√m, since the variance of a sample mean satisfies:

$$\text{Var}(\bar{X}) = \mathbb{E}(\bar{X} - \mathbb{E}\bar{X})^2 \approx \frac{\sigma^2}{n}$$

---

## 3. Kernel Ridge Regression

### Motivation

If regularization works well, we can use very high-degree polynomial features. Map each input x to its **monomial features of degree p**:

$$x \longrightarrow \tilde{x} \equiv \text{Monomials of } x \text{ of degree } p \quad \in \mathbb{R}^D$$

Then solve the standard ridge regression in the feature space ℝ^D:

$$\underset{\tilde{\omega} \in \mathbb{R}^D}{\min} \; \frac{1}{n}\sum_{i}(y_i - \tilde{\omega} \cdot \tilde{x})^2 + \lambda\|\tilde{\omega}\|^2$$

**Problem:** As p → ∞, D = d^p grows exponentially. Computing in ℝ^D is intractable.

### The Kernel Trick

**Key insight (Vapnik et al.):** We never need the explicit feature vectors x̃. We only ever need their **dot products** x̃ · x̃'. Can we compute these dot products cheaply using the original x?

**Example:** Let k(x,y) = (x·y + 1)² for x, y ∈ ℝ². Expanding:

$$(x_1 y_1 + x_2 y_2 + 1)^2 = x_1^2 y_1^2 + x_2^2 y_2^2 + 2x_1 x_2 y_1 y_2 + 2x_1 y_1 + 2x_2 y_2 + 1$$

This equals the dot product of two 6-dimensional feature vectors:

$$(x_1^2, x_2^2, \sqrt{2}\, x_1 x_2, \sqrt{2}\, x_1, \sqrt{2}\, x_2, 1) \cdot (y_1^2, y_2^2, \sqrt{2}\, y_1 y_2, \sqrt{2}\, y_1, \sqrt{2}\, y_2, 1)$$

So **k(x, y) = (x·y + 1)^p corresponds to x̃ · ỹ**, where x̃, ỹ are the degree-p monomial feature vectors (in D = d^p dimensions). We compute this dot product in O(d) time — completely bypassing the D-dimensional space!

Such a function k(x,y) is called a **kernel**. Models that use this trick are called **Kernel Machines**.

---

## 4. The Representer Theorem

We want to minimize:

$$\hat{R}_\lambda(\tilde{\omega}) = \frac{1}{n}\sum_{i=1}^{n}(y_i - \tilde{\omega} \cdot \tilde{x}_i)^2 + \lambda \,\tilde{\omega} \cdot \tilde{\omega}$$

over ω̃ ∈ ℝ^D.

**Key idea:** We only need to search over ω̃ that are **linear combinations of the training feature vectors**:

$$\tilde{\omega} = \sum_{j=1}^{n} d_j \tilde{x}_j \quad \text{for some unknown } d = (d_1, \ldots, d_n)^T \in \mathbb{R}^n$$

**Why is this W.L.O.G.?** (Without Loss of Generality)

For any ω̃ ∈ ℝ^D, consider its projection ω̃' onto the span of {x̃_i}:

- **The loss doesn't change:** ω̃ · x̃_i = ω̃' · x̃_i for all i, so R̂(ω̃) = R̂(ω̃')
- **The penalty only gets smaller:** ‖ω̃‖ ≥ ‖ω̃'‖ (projection is shorter), so λ‖ω̃‖ ≥ λ‖ω̃'‖
- **Therefore:** R̂_λ(ω̃) ≥ R̂_λ(ω̃')

So the minimizer always lies in the span of the training points:

$$\underset{\tilde{\omega}}{\min} \; \hat{R}_\lambda(\tilde{\omega}) = \underset{\tilde{\omega} \in \text{Span}\{\tilde{x}_i\}}{\min} \; \hat{R}_\lambda(\tilde{\omega})$$

This result is known as the **Representer Theorem**.

---

## 5. Solving Kernel Ridge Regression

### Rewriting in Terms of α

Let ω̃ = Σ_j α_j x̃_j. Then:

$$\hat{R}_\lambda(\alpha) = \frac{1}{n}\sum_{i=1}^{n}(y_i - \tilde{\omega} \cdot \tilde{x}_i)^2 + \lambda\, \tilde{\omega} \cdot \tilde{\omega}$$

Using the Representer Theorem substitution:

$$\tilde{\omega} \cdot \tilde{x}_i = \sum_j \alpha_j (\tilde{x}_j \cdot \tilde{x}_i) = \sum_j \alpha_j K_{ij}$$

$$\tilde{\omega} \cdot \tilde{\omega} = \sum_{i,j} \alpha_i \alpha_j (\tilde{x}_i \cdot \tilde{x}_j) = \alpha^T K \alpha$$

### The Gram (Kernel) Matrix

Define the **Gram matrix** K (also called the kernel matrix):

$$K = \begin{bmatrix} \tilde{x}_1 \cdot \tilde{x}_1 & \tilde{x}_1 \cdot \tilde{x}_2 & \cdots & \tilde{x}_1 \cdot \tilde{x}_n \\ \tilde{x}_2 \cdot \tilde{x}_1 & \tilde{x}_2 \cdot \tilde{x}_2 & \cdots & \vdots \\ \vdots & & \ddots & \vdots \\ \tilde{x}_n \cdot \tilde{x}_1 & \cdots & & \tilde{x}_n \cdot \tilde{x}_n \end{bmatrix}$$

So K_{ij} = x̃_i · x̃_j = **k(x_i, x_j)** using a kernel such as k(x,y) = (x·y + 1)^p.

This is the key: each entry of K is computed using the kernel function on the *original* inputs — no high-dimensional vectors needed.

### The Objective in Matrix Form

$$\hat{R}_\lambda(\alpha) = \frac{1}{n}\|Y - K\alpha\|^2 + \lambda\, \alpha^T K \alpha$$

where Y = (y_1, ..., y_n)^T is the vector of labels.

### Finding the Optimal α̂

**Goal:** Minimize over α ∈ ℝ^n. Set the gradient to zero:

$$\nabla \hat{R}_\lambda(\alpha) = 2(K^2 + n\lambda K)\alpha - 2KY = 0$$

$$(K^2 + n\lambda K)\alpha = KY$$

Assuming K is invertible (factor out K):

$$(K + n\lambda I_n)\alpha = Y$$

$$\boxed{\hat{\alpha} = (K + n\lambda I_n)^{-1} Y}$$

This is an n×n linear system — much better than a D×D system when p is large!

### Making Predictions

For any future input x ∈ ℝ^d:

$$\hat{f}(x) = \tilde{\omega} \cdot \tilde{x} = \sum_{i=1}^{n} \hat{\alpha}_i \,(\tilde{x}_i \cdot \tilde{x}) = \sum_{i=1}^{n} \hat{\alpha}_i \, k(x_i, x)$$

The prediction is a **weighted sum of kernel evaluations** between the training points and x. No feature vectors are ever explicitly computed.

---

## 6. Common Kernels

| Kernel | Formula | Notes |
|--------|---------|-------|
| **Polynomial** | k(x,y) = (x·y + 1)^p | Corresponds to degree-p monomials; D = d^p |
| **Gaussian (RBF)** | k(x,y) = exp(−‖x−y‖²/2σ²) | σ is a bandwidth parameter; embeds in ℝ^∞ |
| **Laplace** | k(x,y) = exp(−‖x−y‖·c₀), c₀ > 0 | Heavier tails than Gaussian |
| **Exponential** | k(x,y) = exp(x·y) | |

> **Note:** The Gaussian and Laplace kernels "embed" x → x̃ ∈ ℝ^∞ — they implicitly use an infinite-dimensional feature space!

### Non-Parametric Methods

Procedures that effectively operate in ℝ^∞ are called **non-parametric**. The defining property: the number of parameters we learn grows with n → ∞. The model class F is "infinite-dimensional."

---

## 7. Bayesian Interpretation of Regularization

### Background: MLE vs MAP

**Maximum Likelihood Estimation (MLE):** Find the parameter that maximizes the likelihood of the observed data:

$$\hat{\theta} = \underset{\theta}{\text{argmax}} \; \log\left(\prod_i f_\theta(z_i)\right) = \underset{\theta}{\text{argmax}} \; \text{Likelihood(Data)}$$

**Maximum A Posteriori (MAP):** Instead of maximizing likelihood, maximize the posterior probability of the parameter given the data:

$$\hat{\theta} = \underset{\theta}{\text{argmax}} \; P(\theta \mid \text{Data}) \quad \leftarrow \text{Posterior "Belief"}$$

By Bayes' theorem:

$$P(\theta \mid \text{Data}) = \frac{P(\text{Data} \mid \theta)\, P(\theta)}{P(\text{Data})}$$

So MAP maximizes: **log P(Data|θ) + log P(θ)** = log likelihood + log prior.

P(θ) is the **prior belief** about θ before seeing data.

### Ridge Regression as MAP

Ridge regression can be derived as MAP estimation:

$$\underset{\omega}{\text{argmax}} \; P(\omega \mid \text{Data}) = \underset{\omega}{\text{argmax}} \; \log P(\text{Data} \mid \omega) + \log P(\omega)$$

**Likelihood model:** Assume Y = ω^T X + ε, where ε ~ N(0,1) is noise. Then:

$$Y_i \sim \mathcal{N}(\omega^T x_i, 1) \propto \exp\!\left(-\tfrac{1}{2}\|y_i - \omega^T x_i\|^2\right)$$

$$\log P(\text{Data} \mid \omega) \propto -\sum_i (y_i - \omega^T x_i)^2$$

**Prior on ω:** Suppose ω ~ N(0, σ²). Then:

$$\log P(\omega) \propto -\frac{1}{2\sigma^2}\|\omega\|^2$$

Putting it together, MAP becomes:

$$\underset{\omega}{\text{argmin}} \; \sum_i (y_i - \omega^T x_i)^2 + \frac{1}{2\sigma^2}\|\omega\|^2$$

This is exactly **Ridge Regression** with λ = 1/(2σ²)!

**Takeaway:** Ridge regularization is equivalent to placing a Gaussian prior on the weights. Larger σ² → weaker prior → smaller λ → less regularization.

---

## 8. Classical Nonparametric Approaches

### Setup

We observe data Y = f(X) + noise, where f is an unknown function. Rather than fitting a single global model, the idea is to **approximate f locally with simple functions**.

Three main approaches (from simplest to most complex):

1. **Local Constant Regression** ← covered in this lecture
2. Local Linear Regression
3. Local Polynomial Regression

### Local Constant Regression

**Intuition:** If Y = f(x) + ε with E[ε] = 0, then E[Y|x] = f(x). So we can estimate f(x) by averaging nearby Y values.

**Procedure:** Let B(x) be a neighborhood of x. Then:

$$\hat{f}(x) = \frac{1}{n_x}\sum_{x_i \in B(x)} y_i$$

where n_x = number of training points in B(x).

### How to Define the Neighborhood B(x)

**Option 1: ε-Nearest Neighbor (ε-NN)**

$$B(x) = \{x' : \|x - x'\| \leq \varepsilon\}$$

- Fixed radius ε; the number of points n_x varies with x.

**Option 2: k-Nearest Neighbor (k-NN)**

- Pick the k closest training points to x.
- Fixed number of neighbors k; the radius of B(x) varies with x.

**Option 3: Partition-Based (Tree-Based)**

- Divide the feature space X into cells.
- B(x) = the cell containing x.
- **Advantage:** Pre-compute the average Y in each cell at training time.
- At test time, finding the cell containing x takes O(height of tree), typically **O(log n)**.
- f̂(x) = f̂(cell containing x).

> **Note:** Partition/tree-based methods are often easier to compute at test time than NN methods, which require searching over all training points.

---

## 9. Bias-Variance Tradeoff

### Computing the Estimator's Error

For a cell B, the estimator is:

$$\hat{f}(B) = \frac{1}{n_B}\sum_{x_i \in B} y_i$$

Treating the X_i as fixed and Y_i as random:

$$\mathbb{E}[\hat{f}(B)] = \frac{1}{n_B}\sum_{x_i \in B} \mathbb{E}[y_i] = \frac{1}{n_B}\sum_{x_i \in B} f(x_i)$$

This is the average of f over the training points in B — not f(x) exactly.

### Bias

$$\text{Bias} = |f(x) - \mathbb{E}[\hat{f}(x)]|$$

The bias measures how far the expected prediction is from the truth. If B is large, the average of f over B may be far from f(x) — **larger cells → larger bias**.

### Variance

$$\text{Var}(\hat{f}(B)) = \frac{\sigma^2}{n_B} \quad \text{(if Var}(Y_i | x_i) = \sigma^2\text{)}$$

If B is small, n_B is small, and the average of a few noisy Y values is highly variable — **smaller cells → larger variance**.

### The Tradeoff

| Cell Size | Bias | Variance | Total Error |
|-----------|------|----------|-------------|
| Large | High (over-smoothed) | Low (many points) | High |
| Small | Low (local) | High (few points) | High |
| Optimal (λ*) | Balanced | Balanced | Minimum |

The total error `(f̂(x) - f(x))²` is a U-shaped function of cell size. This is the **Bias-Variance Tradeoff** — it is a fundamental principle across all of machine learning.

**How to pick the right cell size (or tree height):** Use **cross-validation** on the validation set, exactly as with λ in ridge regression.

---

## Summary of Key Ideas

| Topic | Key Formula / Concept |
|-------|-----------------------|
| Ridge Regression | min (1/n)Σ(y_i − ω^T x_i)² + λ‖ω‖² |
| Cross-Validation | Pick λ̂ = argmin_λ R̂'(ω̂_λ) on held-out data |
| Kernel Trick | k(x,y) = x̃·ỹ computed in O(d) time without computing x̃ |
| Representer Theorem | Optimal ω̃ is a linear combination of training x̃_i's |
| Kernel Solution | α̂ = (K + nλI)^{-1}Y, then f̂(x) = Σ α̂_i k(x_i, x) |
| MAP Interpretation | Ridge = Gaussian prior on ω (λ = 1/2σ²) |
| Local Regression | f̂(x) = average Y in a neighborhood B(x) |
| Bias-Variance Tradeoff | Large B → high bias; small B → high variance |
