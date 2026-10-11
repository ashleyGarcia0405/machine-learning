# Lecture 06: Nonlinear SVMs & Multiclass Classification

---

## 1. Nonlinear SVMs: The Kernel Trick

### 1.1 Motivation

Recall the soft-margin SVM objective (from Lecture 05):

$$\min_{\omega, c} \; \lambda \|\omega\|^2 + \frac{1}{n} \sum_{i=1}^n \left(1 - y_i(\omega \cdot x_i - c)\right)_+$$

This finds a **linear** decision boundary $\omega \cdot x - c = 0$. But many real datasets are **not linearly separable** in the original feature space.

**Goal:** Consider nonlinear boundaries. For example:

$$x_{(2)} = a x_{(1)}^2 + c \quad \Rightarrow \quad f(x) = x_{(2)} - a x_{(1)}^2 - c$$

The decision boundary $f(x) = 0$ is a **parabola** in input space, and the classifier is:

$$h(x) = \operatorname{sign}(f(x))$$

**Intuition:** Higher-order polynomials in $x$ yield richer decision boundaries — quadratics give parabolas/ellipses/hyperbolas, cubics give even more complex shapes.

### 1.2 Feature Map Approach

**Idea (from Lecture 04):** Map $x \in \mathbb{R}^d$ to a higher-dimensional feature vector:

$$\tilde{x} \leftarrow \text{all monomials of degree } p \text{ of } x$$

For example, with $d = 2$ and $p = 2$: $\tilde{x} = (x_1^2, x_1 x_2, x_2^2, x_1, x_2, 1)$, giving $\tilde{x} \in \mathbb{R}^D$ where $D \gg d$.

Then we solve the SVM in $\tilde{x}$ space:

$$\min_{\tilde{\omega}, \tilde{c}} \; \lambda \|\tilde{\omega}\|^2 + \frac{1}{n} \sum_{i=1}^n \left(1 - y_i(\tilde{\omega} \cdot \tilde{x}_i - \tilde{c})\right)_+$$

**Problem:** If $p$ is large, $D$ can be enormous (exponential in $p$), making this computationally prohibitive.

### 1.3 The Kernel Trick (Better Yet!)

**Key observation:** We never need to explicitly compute $\tilde{x}$. A kernel function:

$$k(x, x') = \tilde{x} \cdot \tilde{x}'$$

computes the dot product in the high-dimensional space without ever forming $\tilde{x}$.

**Recall common kernels:**

| Kernel | Formula |
|--------|---------|
| Polynomial | $k(x, x') = (x \cdot x' + 1)^p$ |
| Gaussian (RBF) | $k(x, x') = \exp\!\left(-\frac{\|x - x'\|^2}{2h^2}\right)$ |

The kernel $k(x, x')$ implicitly encodes $\tilde{x} \cdot \tilde{x}'$ **without actually mapping into $\mathbb{R}^D$**, where $D \gg d$ (and $x \in \mathbb{R}^d$). This is the **kernel trick**.

### 1.4 Kernel SVM Derivation

By the **Representer Theorem** (proved in Lecture 04), the optimal $\tilde{\omega}^*$ lies in the span of the training feature vectors:

$$\tilde{\omega} = \sum_{j=1}^n \alpha_j \tilde{x}_j \quad (*)$$

Substituting $(*)$ into the SVM objective:

$$\lambda \sum_{i,j} \alpha_i \alpha_j \underbrace{\tilde{x}_i \cdot \tilde{x}_j}_{K(x_i, x_j)} + \frac{1}{n} \sum_{i=1}^n \left(1 - y_i\!\left(\sum_{j=1}^n \alpha_j \underbrace{\tilde{x}_j \cdot \tilde{x}_i}_{K(x_j, x_i)} - \tilde{c}\right)\right)_+$$

Everything is expressed in terms of **kernel evaluations** $K(x_i, x_j)$, so **we never need to compute $\tilde{x}$**.

### 1.5 Gram Matrix Formulation

Define the **Gram matrix** $K \in \mathbb{R}^{n \times n}$ with entries:

$$K_{ij} = k(x_i, x_j)$$

The kernel SVM becomes:

$$\min_{\alpha, \tilde{c}} \; \lambda \, \alpha^T K \alpha + \frac{1}{n} \sum_{i=1}^n \left(1 - y_i(K_{i,\cdot} \cdot \alpha - \tilde{c})\right)_+$$

We minimize over $\alpha \in \mathbb{R}^n$ (the dual variables) instead of $\tilde{\omega} \in \mathbb{R}^D$.

**Prediction:** For a new point $x$, compute:

$$f(x) = \tilde{\omega} \cdot \tilde{x} = \sum_{j=1}^n \alpha_j k(x_j, x), \quad h(x) = \operatorname{sign}(f(x))$$

> **Historical Remark:** This kernel SVM framework (Vapnik et al.) is considered the **First Revolution** in machine learning — it enabled powerful nonlinear classifiers without explicit high-dimensional computation.

---

## 2. Multiclass Classification: $|Y| > 2$

So far we have handled binary classification ($Y \in \{-1, +1\}$). What if there are $k > 2$ classes?

Three main approaches:

| Approach | Idea |
|----------|------|
| A: Reductions to Binary | Decompose into binary sub-problems |
| B: Multiclass Losses | Generalize logistic loss directly |
| C: Neighborhood-Based | Extend k-NN / trees to classification |

---

## 3. Approach A: Reductions to Binary Classification

### 3.1 One vs All (OvA)

Train $k$ binary classifiers. Classifier $j$ separates class $j$ from all other classes. To predict, run all $k$ classifiers and pick the most confident.

$$h(x) = \operatorname{argmax}_j f_j(x)$$

where $f_j(x)$ is the score from the "class $j$ vs rest" classifier.

**Pros:** Simple — only $k$ classifiers needed.

**Cons:** Each binary problem is imbalanced (one class vs all others). It **may miss the margin** between individual classes since it doesn't directly optimize boundaries between pairs.

### 3.2 All Pairs (One vs One)

Train a binary classifier for **every pair** of classes. With $k$ classes, this gives $\binom{k}{2}$ classifiers. To predict, use a voting scheme — each classifier votes for one class, and the class with the most votes wins.

**Example with 3 classes (1, 2, 3):**
- Train: 1v2, 1v3, 2v3
- At test time: if 2v3 → class 2, and 1v2 → class 2, then vote → class 2

**Pros:** Each classifier sees only two classes, **better margins** on each sub-problem.

**Cons:** Requires $O(k^2)$ classifiers — expensive for large $k$. The right combination order is also **a priori unclear**.

---

## 4. Approach B: Multiclass Logistic Loss (Softmax)

### 4.1 Binary Logistic Recall

In binary classification ($Y \in \{-1, +1\}$), we modeled:

$$\Pr(Y \mid x) = \sigma(y \cdot f(x)), \quad \sigma(u) = \frac{1}{1+e^{-u}}$$

where $f(x)$ is a real-valued score: $f(x) > 0 \Rightarrow$ positive class, $f(x) < 0 \Rightarrow$ negative class.

For linear $f$: $f(x) = \omega^T x$.

### 4.2 Extension to $k$ Classes

For $k$ classes, replace the scalar $f(x)$ with a **vector**:

$$f(x) \in \mathbb{R}^k, \quad f(x) = \begin{bmatrix} f_1(x) \\ \vdots \\ f_k(x) \end{bmatrix}$$

The interpretation is: if $f_i(x) > f_j(x)$, then class $i$ is more likely than class $j$.

**Linear case:** $f_i(x) = \omega_i \cdot x$ for unknown weight vectors $\omega_1, \ldots, \omega_k$.

### 4.3 The Softmax Function

The **sigmoid** generalizes to a vector-valued function. For any $a \in \mathbb{R}^k$, define $\sigma(a) \in \mathbb{R}^k$ component-wise:

$$\sigma_i(a) = \frac{e^{a_i}}{\sum_{j=1}^k e^{a_j}}$$

This is called the **softmax** function (also: multiclass logistic function). Note:
- $\sigma_i(a) \geq 0$ for all $i$
- $\sum_{i=1}^k \sigma_i(a) = 1$ (it's a valid probability distribution)
- The largest $a_i$ gets the highest probability

### 4.4 Multiclass Probabilistic Model

$$\boxed{\Pr(Y = i \mid x) = \sigma_i(f(x)) = \frac{e^{f_i(x)}}{\sum_{j=1}^k e^{f_j(x)}}}$$

The classifier predicts the class with highest probability:

$$h(x) = \operatorname{argmax}_{i \in \{1,\ldots,k\}} \Pr(Y = i \mid x) = \operatorname{argmax}_{i} f_i(x)$$

### 4.5 MLE → Multiclass Logistic Loss

Given training data $\{(x_i, y_i)\}_{i=1}^n$, let $\tilde{y}_i \in \mathbb{R}^k$ be the **one-hot encoding** of $y_i$:

$$\tilde{y}_i = [0, \ldots, 0, \underbrace{1}_{j\text{-th position}}, 0, \ldots, 0] \quad \text{when } y_i = j$$

Then: $\Pr(Y = j \mid x_i) = \sigma_j(f(x_i))$, and since $Y_i = j$:

$$\Pr(Y_i \mid x_i) = \tilde{y}_i \cdot \sigma(f(x_i)) = \sigma_j(f(x_i))$$

**Maximum Likelihood Estimation** maximizes the log-likelihood:

$$\mathcal{L}(f) = \log \prod_{i=1}^n \Pr_f(Y_i \mid x_i) = \sum_{i=1}^n \log\!\left(\tilde{y}_i \cdot \sigma(f(x_i))\right)$$

Equivalently, **minimize** the negative log-likelihood:

$$\boxed{\min_f \sum_{i=1}^n -\log\!\left(\tilde{y}_i \cdot \sigma(f(x_i))\right) \quad \leftarrow \text{ Multiclass Logistic Loss (Cross-Entropy Loss)}}$$

### 4.6 Relation to Binary Case

**Claim:** The multiclass logistic loss reduces to the binary logistic loss when $k = 2$.

**Proof:** Suppose $f \in \mathbb{R}^1$ (scalar), and let $F(x) = [f(x), -f(x)]^T \in \mathbb{R}^2$ (class 1 score = $f(x)$, class 2 score = $-f(x)$).

Then the softmax probability for class 1 is:

$$\sigma_1(F(x)) = \frac{e^{f(x)}}{e^{f(x)} + e^{-f(x)}}$$

Now notice:

$$\sigma(2f(x)) = \frac{1}{1+e^{-2f(x)}} = \frac{e^{f(x)}}{e^{f(x)}(1+e^{-2f(x)})} = \frac{e^{f(x)}}{e^{f(x)} + e^{-f(x)}} = \sigma_1(F(x))$$

So the binary logistic model is a special case of softmax with $F(x) = [f(x), -f(x)]^T$. For linear $f(x) = \omega^T x$, we get $2f(x) = 2\omega^T x$ — just a rescaling of the weight vector.

### 4.7 Application: Next Token Prediction

The multiclass logistic framework is the foundation of **language models** (like ChatGPT).

**Setup:**
- A **token** is an individual unit of a sequence (e.g., a word, subword, or character in text)
- **Example:** "The blue cat is [beautiful]"
  - $x$ = "The blue cat is" (context)
  - $y$ = "beautiful" (next token to predict)
- The label space $Y$ has $k$ possible tokens (vocabulary size, often $k \approx 50{,}000$)
- $f(x) \in \mathbb{R}^k$, $\Pr(Y = \text{token}_i \mid x) = \sigma_i(f(x))$

**Representation of $x$:** A key challenge is how to represent the context $x$ as a vector. One approach is a **one-hot** representation (a vector with a 1 at the position of each word), but this ignores word order and context. Modern transformers use learned embeddings — this connects to neural networks (Lecture 07+).

**Loss:** The model is trained with multiclass logistic loss (cross-entropy) over a huge corpus of text, which is exactly next-token prediction.

---

## 5. Approach C: Neighborhood-Based Methods for Classification

### 5.1 $\varepsilon$-Nearest Neighbor Classifier

This is the classification analogue of $\varepsilon$-ball regression from Lecture 04. Define $B(x, \varepsilon) = \{x' : \|x' - x\| \leq \varepsilon\}$ (the ball of radius $\varepsilon$ around $x$).

**Regression (recall):** $\hat{f}(x) = \text{average } Y \text{ value in } B(x, \varepsilon)$

**Classification:** 

$$\hat{h}(x) = \text{Majority class label in } B(x, \varepsilon)$$

Among all training points $x_i$ within distance $\varepsilon$ of $x$, predict the most common class.

### 5.2 Relation to Regression

For $Y \in \{-1, +1\}$ (binary), the regression estimate $\hat{f}(x)$ in the neighborhood is:

$$\hat{f}(x) = \hat{\Pr}(Y = 1 \mid B(x,\varepsilon)) - \hat{\Pr}(Y = -1 \mid B(x,\varepsilon))$$

(since the average of $\pm 1$ labels gives the probability difference). Then:

$$\hat{h}(x) = \operatorname{sign}(\hat{f}(x))$$

This means: the classification rule from $\varepsilon$-NN regression is exactly the neighborhood majority vote. This connection applies to **any** neighborhood-based regression approach:
- $k$-NN
- Partition / Tree-based methods

### 5.3 Tree/Partition Methods for Classification

In Lecture 04, we saw partition-based regression: divide the input space into cells, predict the average $Y$ in each cell.

For classification: predict the **majority class** in each cell.

**Richness:** A partition of $\mathbb{R}^d$ with cells corresponding to alternating $+/-$ labels can express complex, non-convex class boundaries — far more expressive than any linear separator.

**Example:** Documents as $x = (x_{(1)}, \ldots, x_{(d)})$ where $x_{(j)}$ = frequency of word $j$. A tree partition of document space can capture complex topic boundaries.

With enough data ($n \to \infty$), partition classifiers are **universally consistent** — they can approximate any true decision boundary.

### 5.4 The Curse of Dimensionality

**Problem:** In high dimensions, partition methods require exponentially many samples.

**Setup:** Suppose $P_x \sim \text{Uniform}([0,1]^d)$ (uniform over the $d$-dimensional unit cube). Use a regular grid partition — in $d$ dimensions with 2 cells per axis, this gives $2^d$ cells.

Each cell has probability:

$$P_x(B_x) \approx \frac{1}{2^d} = 2^{-d}$$

**Expected points per cell:**

$$n \cdot 2^{-d} \geq 1 \quad \Rightarrow \quad n \gg 2^d$$

To have even one training point per cell, we need **exponentially many samples** in $d$. This is the **Curse of Dimensionality**.

**Implications:**
- In $d = 10$: need $n \gg 1024$
- In $d = 20$: need $n \gg 10^6$
- In $d = 100$: infeasible

The number of samples required grows **exponentially with dimension** $d$. This fundamental obstacle affects all neighborhood-based methods.

### 5.5 General k-Class Neighborhood Classifier

For $k$ classes ($Y \in \{1, 2, \ldots, k\}$), the neighborhood classifier generalizes directly:

$$\hat{h}(x) = \operatorname{argmax}_{l \in \{1, \ldots, k\}} \hat{P}_r(Y = l \mid B(x))$$

Predict the class with the highest empirical probability in the neighborhood $B(x)$.

**Bayes Risk Recap:** Recall from Lecture 05, the Bayes risk is:

$$R(h) = \mathbb{E}\,\mathbf{1}\{h(x) \neq Y\} = \Pr(h(x) \neq Y) = \int_{\mathcal{X}} \Pr(h(x) \neq Y \mid x)\, d\mu(x)$$

The neighborhood classifier approximates the Bayes optimal rule $h^*(x) = \operatorname{argmax}_l \Pr(Y=l \mid x)$ using empirical estimates from the local neighborhood. As $n \to \infty$ (and $\varepsilon \to 0$ appropriately), it converges to the Bayes classifier.

---

## 6. Summary

| Topic | Key Formula / Concept |
|-------|----------------------|
| Kernel SVM | $\min_\alpha \lambda \alpha^T K \alpha + \frac{1}{n}\sum_i (1 - y_i(K_{i,\cdot}\alpha - c))_+$ |
| Kernel trick | $k(x,x') = \tilde{x} \cdot \tilde{x}'$, avoids computing $\tilde{x} \in \mathbb{R}^D$ |
| One vs All | $k$ binary classifiers; simple but may miss pairwise margins |
| All Pairs | $\binom{k}{2}$ classifiers; better margins, expensive |
| Softmax | $\sigma_i(a) = e^{a_i} / \sum_j e^{a_j}$, generalizes logistic to $k$ classes |
| Multiclass loss | $\min_f \sum_i -\log(\tilde{y}_i \cdot \sigma(f(x_i)))$ (cross-entropy) |
| Binary ↔ Multiclass | Binary logistic = softmax with $F = [f, -f]^T$ |
| Next Token | Language modeling = multiclass classification over vocabulary |
| $\varepsilon$-NN classifier | Majority vote in $B(x, \varepsilon)$; $\hat{h}(x) = \text{argmax}_l \hat{P}(Y=l \mid B(x))$ |
| Curse of Dimensionality | Need $n \gg 2^d$ for partition methods in $d$ dimensions |

---

## 7. Key Connections

- **Kernel SVM → Kernel Ridge Regression (Lec 04):** The same Representer Theorem applies. The dual variable $\alpha$ plays the same role; the kernel matrix $K$ is the same Gram matrix. The only difference is the loss: hinge loss for SVM vs squared loss for KRR.
- **Softmax → Binary Logistic (Lec 05):** Softmax with $k=2$ and $F(x) = [f(x), -f(x)]^T$ exactly recovers binary logistic regression.
- **Neighborhood methods → Bias-Variance (Lec 04):** Small $\varepsilon$ (or large $k$) gives low bias but high variance; large $\varepsilon$ (or small $k$) gives high bias but low variance. The curse of dimensionality makes it hard to have both good bias AND enough samples in high dimensions.
- **Multiclass Softmax → Neural Networks (Lec 07+):** The output layer of nearly every neural network classifier is a softmax. The function $f(x) = \sum_i \alpha_i \phi(\omega_i \cdot x)$ (seen at end of notes) previews the neural network architecture where $\phi$ is a nonlinear activation.
