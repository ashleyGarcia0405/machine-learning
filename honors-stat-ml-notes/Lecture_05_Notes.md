# Lecture 05: Classification

## Overview

**Classification**: Learn a mapping $X \to Y$ where $Y$ is **discrete** — a finite set of values.

**Examples:**
- $Y$ = object in image $X$
- $Y$ = word corresponding to sound clip $X$
- $Y$ = document topic, $X$ = document
- "Next token" prediction: $Y$ = next word, $X$ = text so far

In contrast to unsupervised clustering (where we only observe $\{X_i\}$), classification is **supervised** — we observe labeled pairs $\{(X_i, Y_i)\}_1^n$.

---

## Lecture Outline

- **I)** Key Concepts: Generative vs. Discriminative, Bayes Classifier, Bayes Decision Boundary, Model Class $\mathcal{H}$, 0-1 Loss, Surrogate Losses
- **II a)** Basic Generative Approach — Gaussian Mixture, Decision Boundaries
- **III)** Logistic Regression (both generative and discriminative aspects)
- **IV)** Major Discriminative Approaches — Perceptron (first neural network), Support Vector Machines (kernel approach)
- **V)** Nonparametrics

---

## I. Key Concepts

We observe training data $\{(X_i, Y_i)\}_1^n$.

### a) Generative Approach

A **generative model** specifies a **joint distribution** of $X, Y$.

For simplicity, focus on **binary classification**: $Y \in \{0, 1\}$ or $Y \in \{-1, 1\}$ (e.g., SPAM vs. NOT SPAM).

**Usual model**: Nature picks class $j$ with probability $\pi_j$, then generates $X$ according to distribution $\mu_j$ with density $f_j$. So:

$$X, Y \sim \pi_1 \cdot f_1(x) \quad \text{or} \quad \pi_0 \cdot f_0(x)$$

The **marginal density** of $X$ is:

$$f(x) = \pi_1 f_1(x) + \pi_0 f_0(x)$$

**Usual approach**: Estimate $\pi_j$'s and $f_j$'s from the data.

**Estimator**: The estimated classifier is:

$$\hat{h}(x) = \underset{j}{\operatorname{argmax}} \; \hat{\Pr}(Y = j \mid x)$$

Using Bayes' rule, $\Pr(Y=j \mid x) \propto f_j(x) \cdot \pi_j$, so we compute $\hat{f}_j(x) \cdot \hat{\pi}_j$ for each class.

---

**Definition (Bayes Classifier)**: At the population level, suppose nature truly generated data according to $\{\pi_j, f_j\}$. Then:

$$h^*(x) = \underset{j}{\operatorname{argmax}} \; \Pr(Y = j \mid x)$$

is called the **Bayes Classifier**. It **minimizes** $\Pr(h(x) \neq Y)$ over all possible classifiers $h$.

*Proof sketch*: Fix any $x$. We want to choose $h(x) \in \{-1, 1\}$ to minimize $\Pr(h(x) \neq Y)$. This is minimized by predicting the most likely class, i.e., $h^*(x) = \operatorname{argmax}_{y} \Pr(Y=y \mid x)$.

---

**Definition (Bayes Decision Boundary)**: For $Y \in \{-1, 1\}$:

$$\{x : \Pr(Y = 0 \mid x) = \Pr(Y = 1 \mid x)\}$$

More generally (multiclass): $\{x : \exists j,k,\; \Pr(Y=j \mid x) = \Pr(Y=k \mid x)\}$

This is the set of points where the classifier is "uncertain" — equivalently, where $\Pr(Y=1 \mid x) - \Pr(Y=-1 \mid x) = 0$.

---

### b) Discriminative Approach

For $Y \in \{0, 1\}$ or $\{-1, 1\}$, a discriminative method directly models the classifier:

- $h(x) = -1$ or $1$
- Consider a **model class** (family) $\mathcal{H}$ of classifiers $h$
- Define a **loss** $\ell(h(x), y)$ and **risk**:

$$R(h) = \mathbb{E}_{X,Y}[\ell(h(x), y)]$$

**Example — 0-1 Loss**:

$$\ell(h(x), y) \triangleq \mathbf{1}\{h(x) \neq y\} \qquad \Rightarrow \qquad R(h) = \Pr(h(x) \neq y)$$

**Empirical Risk**:

$$\hat{R}(h) = \frac{1}{n} \sum_{i=1}^n \ell(h(x_i), y_i)$$

**Estimated classifier**:

$$\hat{h} = \underset{h \in \mathcal{H}}{\operatorname{argmin}} \; \hat{R}(h)$$

**Example of model class**: $\mathcal{H} = \{h_t(x) = \operatorname{sign}(x - t)\}$ — threshold classifiers.

---

**Remark (Why the Bayes Classifier is Optimal)**: Fix any $x$, and ask: what is the choice of $h(x) \in \{-1, 1\}$ that minimizes $\Pr(h(x) \neq Y)$?

$$h^*(x) = \underset{\text{values of } Y}{\operatorname{argmax}} \; \Pr(Y \mid x) \quad \leftarrow \text{Bayes Classifier}$$

This minimizes misclassification probability over all "possible" $h$.

---

### Optimization Issues with 0-1 Loss

To use gradient descent (GD or SGD), we need $\hat{R}(\theta) = \frac{1}{n}\sum \ell(h_\theta(x_i), y_i)$ to be differentiable w.r.t. $\theta$ (where $h_\theta(x) = \operatorname{sign}(\theta^T x)$).

**Problem 1**: The 0-1 loss is **not differentiable** → need a **Surrogate Loss**.

**Problem 2**: Need to **parametrize** $h$ to be able to differentiate.

**Solution**: View $h = \operatorname{sign}(f_\theta(x))$ and treat the loss as $\ell(f_\theta(x), y)$ or $\ell(\theta, x, y)$.

**Remark**: $f_\theta$ may be viewed as modeling either:
- The **regression function** $f^*(x) = P(Y=1 \mid x) - P(Y=-1 \mid x)$
- The **decision boundary** (where $f_\theta(x) = 0$)

For the threshold example on $\mathbb{R}$: $f_\theta(x) = x - \theta$ — the decision boundary is at $x = \theta$.

---

## II a. A Basic Generative Model (Gaussian Mixture)

**Model**: $X$ has a mixture density, $Y \in \{0, 1\}$:

$$f(x) = \pi_0 \mathcal{N}(\mu_0, \Sigma_0; x) + \pi_1 \mathcal{N}(\mu_1, \Sigma_1; x)$$

where the Gaussian density is:

$$\mathcal{N}(\mu, \Sigma; x) \propto \frac{1}{|\Sigma|^{1/2}} \exp\!\left(-\tfrac{1}{2}(x-\mu)^T \Sigma^{-1}(x-\mu)\right)$$

**From data** $\{(X_i, Y_i)\}_1^n$: estimate $\hat{\pi}_j$, $\hat{\mu}_j$, $\hat{\Sigma}_j$ using the $X_i$'s with $Y_i = j$, $j \in \{0, 1\}$.

$$\hat{h}(x) = \underset{j \in \{0,1\}}{\operatorname{argmax}} \; \hat{\Pr}(Y=j \mid x)$$

---

### Bayes Decision Boundary for Gaussians

If we had the true distribution, the Bayes Decision is:

$$h^*(x) = \mathbf{1}\{\Pr(1 \mid x) \geq \Pr(0 \mid x)\} = \mathbf{1}\{\log \Pr(1 \mid x) \geq \log \Pr(0 \mid x)\} \tag{*}$$

Computing the log:

$$\log(\mathcal{N}(\mu, \Sigma; x) \cdot \pi) = -\tfrac{1}{2}(x-\mu)^T \Sigma^{-1}(x-\mu) + \underbrace{\log\frac{\pi}{|\Sigma|^{1/2}}}_{C(\pi, \Sigma)}$$

So $(*)$ becomes:

$$-\tfrac{1}{2}(x-\mu_1)^T \Sigma_1^{-1}(x-\mu_1) \geq -\tfrac{1}{2}(x-\mu_0)^T \Sigma_0^{-1}(x-\mu_0) + C(\pi_1, \Sigma_1, \pi_0, \Sigma_0)$$

**Expanding** $(x-\mu)^T \Sigma^{-1}(x-\mu) = x^T \Sigma^{-1} x + \underbrace{\mu^T \Sigma^{-1} \mu}_{\text{no } x} - 2x^T \Sigma^{-1}\mu$, the decision boundary becomes:

$$x^T(\Sigma_0^{-1} - \Sigma_1^{-1})x + 2x^T(\Sigma_1^{-1}\mu_1 - \Sigma_0^{-1}\mu_0) \geq C(\mu_0, \mu_1, \pi_0, \pi_1, \Sigma_0, \Sigma_1)$$

Written as $x^T A x + x^T b \geq C$, so the **decision boundary** is:

$$x^T A x + x^T b - C = 0 \quad \leftarrow \text{Conic Section}$$

This yields **ellipsoids** (including circles), **parabolas**, or **hyperbolas** depending on $A$.

---

### Special Cases

**Basic Case: $\Sigma_1 = \Sigma_0$** (equal covariances):
- $A = \Sigma_0^{-1} - \Sigma_1^{-1} = 0$, so the decision collapses to $x^T b \geq c$ — a **linear boundary**.
- A model that assumes $\Sigma_1 = \Sigma_0$ is called **Fisher's Linear Discriminant**.

**Naive Bayes**: Assumes $\Sigma_j$'s are **diagonal** — i.e., all coordinates of $X$ are independent within each class. This greatly simplifies computation.

**Parabola example**: If $\Sigma_1^{-1} - \Sigma_0^{-1} = \begin{bmatrix} c & 0 \\ 0 & 0 \end{bmatrix}$ (only first coordinate differs), then:

$$x_{(1)}^2 + b_1 x_{(1)} + b_2 x_{(2)} \geq c_0 \quad \leftarrow \text{parabola in } x_{(1)}$$

**Hyperbola example**: $\Sigma_1^{-1} - \Sigma_0^{-1} = \begin{bmatrix} c_1 & 0 \\ 0 & -c_0 \end{bmatrix}$ with $c_1, c_0 > 0$:

$$c_1 x_{(1)}^2 - c_2 x_{(2)}^2 + b_1 x_{(1)} + b_2 x_{(2)} \geq c_0 \quad \leftarrow \text{hyperbola}$$

---

**Remark**: Neural Networks allow for **rich, arbitrary decision boundaries**.

**Remark**: The decision boundary **induced by a classifier** $h$ is just the boundary between the classes it returns — it doesn't have to be a nice geometric shape.

---

## III. Mixed Generative/Discriminative Models: Logistic Regression

$Y \in \{-1, 1\}$.

**Intuition**: In generative models, we modeled $P_{XY}$, i.e., both $P_X$ and $P_{Y|X}$. Here, **ignore $P_X$** and model only $P_{Y|X}$ directly.

---

**Definition (Logistic / Sigmoid Function)**:

$$\sigma(u) = \frac{1}{1 + e^{-u}}$$

This is called the **sigmoid**. It maps $\mathbb{R} \to (0,1)$, so it can be used to model a probability. As $u \to \infty$, $\sigma(u) \to 1$; as $u \to -\infty$, $\sigma(u) \to 0$.

---

Fix a class of functions $f \in \mathcal{H}$ such that very large $f$ → class $+1$, very small $f$ → class $-1$. For example, $f(x) = \omega^T x$.

**Logistic Model**:

$$\Pr(Y=1 \mid x) = \sigma(f(x))$$
$$\Pr(Y=-1 \mid x) = 1 - \Pr(Y=1 \mid x) = \sigma(-f(x))$$

**Why**: $1 - \sigma(u) = \sigma(-u)$.

*Verify*: $1 - \frac{1}{1+e^{-u}} = \frac{e^{-u}}{1+e^{-u}} = \frac{1}{1+e^u} = \sigma(-u)$ ✓

Therefore, for $y \in \{-1, +1\}$:

$$P_f(Y = y \mid x) = \sigma(y \cdot f(x)) \quad \leftarrow \text{Logistic Model with class } \mathcal{H}$$

The decision boundary is $\{x : f(x) = 0\}$, where $\Pr(Y=1) = \Pr(Y=-1) = 1/2$.

---

### Induced Classifier via Maximum Likelihood

Observe $\{x_i, y_i\}_1^n$. Choose $f \in \mathcal{H}$ via maximum likelihood:

$$\mathcal{L}(f) = \prod_{i=1}^n P_f(y_i \mid x_i) \cdot P(x_i)$$

**Log-likelihood** (dropping the constant $\sum \log P(x_i)$, since we're ignoring $P_X$):

$$\hat{f} = \underset{f \in \mathcal{H}}{\operatorname{argmax}} \sum_{i=1}^n \log P_f(y_i \mid x_i) = \underset{f \in \mathcal{H}}{\operatorname{argmax}} \sum_{i=1}^n \log \sigma(y_i f(x_i))$$

Since $\log \sigma(y_i f(x_i)) = -\log(1 + \exp(-y_i f(x_i)))$:

$$\hat{f} = \underset{f \in \mathcal{H}}{\operatorname{argmin}} \; \frac{1}{n} \sum_{i=1}^n \log(1 + \exp(-y_i f(x_i)))$$

This is the **Logistic Loss**:

$$\ell(f(x), y) = \log(1 + e^{-y f(x)})$$

This is a first example of a **surrogate loss** — a smooth, differentiable proxy for the 0-1 loss.

> **Remark**: The logistic loss is one of the most used losses today, especially with neural networks.

---

### Surrogate Loss Notation

We often express losses in terms of $y \cdot f(x)$:

$$\ell(f(x), y) \equiv g(y \cdot f(x)), \quad y \in \{-1, +1\}$$

For logistic loss: $g(u) = \log(1 + e^{-u})$.

Note: $g(u)$ is a **convex upper bound** on the 0-1 loss $\mathbf{1}\{u \leq 0\}$:

$$\hat{R}_{0\text{-}1}(f) = \frac{1}{n}\sum_i \mathbf{1}\{y_i f(x_i) \leq 0\} \leq \frac{1}{n}\sum_i \ell(y_i f(x_i))$$

---

## IV a. Perceptron Algorithm (Rosenblatt)

The Perceptron is a choice of **surrogate loss + SGD**. It is historically the **first neural network**.

For simplicity: $\mathcal{H} = \{f(x) = \omega^T x,\; \omega, x \in \mathbb{R}^d\}$.

**Perceptron Loss**: $\ell(\omega^T x, y) \equiv g(y \cdot \omega^T x)$ where:

$$g(u) = (-u)_+ = \max(-u, 0)$$

This penalizes only when $y \cdot \omega^T x \leq 0$ (i.e., when the classifier makes a mistake — the point is on the wrong side).

**Empirical risk**:

$$\hat{R}(\omega) = \frac{1}{n}\sum_{i=1}^n (-y_i \cdot \omega^T x_i)_+$$

**Optimization**: $\hat{\omega} = \operatorname{argmin}_{\omega \in \mathbb{R}^d} \hat{R}(\omega)$ via SGD:

$$\omega_{t+1} = \omega_t - \eta_t \cdot \tilde{\nabla}\hat{R}(\omega_t)$$

where we pick a random $(y_t, x_t)$ at each step.

Since the loss is **not differentiable** at 0, we use a **sub-gradient**:

$$\partial g(u) = \begin{cases} -1 & \text{if } u \leq 0 \\ 0 & \text{otherwise} \end{cases}$$

Therefore:

$$\partial\bigl(g(y \cdot \omega^T x)\bigr) = \begin{cases} -yx & \text{if } y \cdot \omega^T x \leq 0 \quad (\text{mistake!}) \\ 0 & \text{otherwise} \end{cases}$$

---

### Perceptron Update Rule

Start with $\omega_0 = 0$. At each time $t$, pick random $(x_t, y_t)$ (let $\eta_t = 1$):

$$\omega_{t+1} = \begin{cases} \omega_t + y_t \cdot x_t & \text{if } y_t \cdot \omega_t^T x_t \leq 0 \quad (\text{mistake}) \\ \omega_t & \text{otherwise} \end{cases}$$

**Interpretation**: Only update when the current $\omega_t$ misclassifies $(x_t, y_t)$. The update pushes $\omega$ in the direction of $y_t x_t$, which makes $y_t \cdot \omega^T x_t$ larger (more positive) for that point.

---

### Margin

**Definition**: The **margin** of $\omega$ (with $\|\omega\|=1$) is:

$$\gamma(\omega) = \min_i \; y_i \cdot \omega^T x_i$$

This measures how far the **closest** point is from the separating hyperplane $\{\omega^T x = 0\}$.

**Terminology**: We say $\{(x_i, y_i)\}$ is **separable with margin $\gamma$** if $\exists \omega$ with $\|\omega\|=1$ such that $\gamma(\omega) \geq \gamma > 0$.

Equivalently: $\forall i,\; y_i \cdot \omega^T x_i \geq \gamma$.

---

### Theorem: Perceptron Convergence

**Theorem**: Suppose data $\{(x_i, y_i)\}_1^n$ is separable with margin $\gamma > 0$, and let $\max_i \|x_i\| \leq r$. Then the Perceptron converges within at most:

$$T \leq \frac{r^2}{\gamma^2} \text{ iterations}$$

**Proof**:

Suppose $\exists\, \omega^*$, $\|\omega^*\|=1$, achieving margin $\gamma$, i.e., $\forall i: y_i(\omega^* \cdot x_i) \geq \gamma$.

Note: $\omega_T = \sum_{t=1}^T y_t \cdot x_t$ (sum over all mistaken points, since we only update on mistakes).

**Lower bound on $\omega_T \cdot \omega^*$**:

$$\omega_T \cdot \omega^* = \sum_{t=1}^T y_t(x_t \cdot \omega^*) \geq T \cdot \gamma$$

(Each term $y_t(x_t \cdot \omega^*) \geq \gamma$ by assumption.)

**Upper bound on $\|\omega_T\|^2$**:

$$\|\omega_T\|^2 = \|\omega_{T-1} + y_T x_T\|^2 = \|\omega_{T-1}\|^2 + \|y_T x_T\|^2 + 2y_T x_T \cdot \omega_{T-1}$$

The last term is $\leq 0$ because we only update on mistakes ($y_T \omega_{T-1}^T x_T \leq 0$). So:

$$\|\omega_T\|^2 \leq \|\omega_{T-1}\|^2 + \|x_T\|^2 \leq \|\omega_{T-1}\|^2 + r^2$$

By recursion (base case $\|\omega_1\| = \|x_1\| \leq r$):

$$\|\omega_T\|^2 \leq T \cdot r^2$$

**Combining** with Cauchy-Schwarz ($\omega_T \cdot \omega^* \leq \|\omega_T\| \cdot \|\omega^*\| = \|\omega_T\|$):

$$T \cdot \gamma \leq \omega_T \cdot \omega^* \leq \|\omega_T\| \leq \sqrt{T} \cdot r$$

$$\Rightarrow \sqrt{T} \leq \frac{r}{\gamma} \Rightarrow T \leq \frac{r^2}{\gamma^2} \quad \square$$

**Remark**: This shows the effect of **margin on computation** — a larger margin guarantees fewer mistakes before convergence.

**Remark (Statistical quality)**: The Perceptron finds a $\hat{\omega}^*$ that achieves zero training error, but $\hat{\omega}^*$ which achieves a **large margin** likely has better **future error** than $\omega_T$ which just has zero mistakes on the sample. → This motivates SVMs.

---

## IV b. Large Margin SVMs (Support Vector Machines)

**Goal**: Find the **largest margin** classifier in the data.

**Model class**: $f(x) = \omega^T x - c$, $Y \in \{-1, 1\}$.

- Margin at $(x_i, y_i)$: $y_i(\omega^T x_i - c)$
- Margin of $(\omega, c)$: $\gamma(\omega, c) = \min_{i \in [n]} y_i(\omega^T x_i - c)$

---

### Hard-Margin SVM

**Algo 0** (naive): $\underset{\|\omega\|=1,\, c \in \mathbb{R}}{\operatorname{argmax}} \; \gamma(\omega, c)$

This is not easy to optimize. Find an equivalent, easier formulation.

**Working Assumption**: $\exists\, \omega^*, c^*$ with $\|\omega^*\|=1$ such that $\forall i,\; y_i(\omega^* \cdot x_i - c) \geq \gamma^* > 0$.

**Key insight**: If we rescale by $1/\gamma^*$, letting $\tilde{\omega} = \omega^*/\gamma^*$ and $\tilde{c} = c^*/\gamma^*$, then $y_i(\tilde{\omega} \cdot x_i - \tilde{c}) \geq 1$ with $\|\tilde{\omega}\| = 1/\gamma^*$.

So **minimizing $\|\omega\|$** (with margin $\geq 1$) is equivalent to **maximizing** $\gamma^*$.

**Algo 1 — Hard-Margin SVM** (boxed):

$$\min_{\omega, c} \|\omega\|^2 \quad \text{s.t.} \quad \forall i,\; y_i(\omega \cdot x_i - c) \geq 1$$

**Remark**: This is a **convex quadratic program** → can be optimized efficiently (e.g., via the Lagrangian).

---

### KKT Conditions and Support Vectors

**Lagrangian**:

$$\mathcal{L}(\lambda, \omega) = \|\omega\|^2 + \sum_i \lambda_i(1 - y_i(\omega \cdot x_i - c))$$

At the optimum $(\lambda^*, \omega^*)$: $\mathcal{L}(\lambda^*, \omega^*) = \|\omega^*\|^2$, which requires:

$$\sum_i \lambda_i^*(1 - y_i(\omega^* \cdot x_i - c)) = 0$$

Since each $\lambda_i^* \geq 0$ and each $(1 - y_i(\ldots)) \leq 0$ at the optimum, we get (KKT complementary slackness):

$$\forall i: \quad \lambda_i^* = 0 \quad \text{OR} \quad y_i(\omega^* \cdot x_i - c) = 1$$

From $\nabla_\omega \mathcal{L}|_{\lambda^*, \omega^*} = 0$:

$$2\omega^* - \sum_i \lambda_i^*(y_i \cdot x_i) = 0 \quad \Rightarrow \quad \omega^* = \sum_i d_i \cdot x_i$$

where $d_i = \lambda_i^* y_i / 2$, and $d_i \neq 0$ only when $\lambda_i^* \neq 0$, i.e., when $y_i(\omega^* \cdot x_i - c) = 1$.

The points $x_i$ where $\lambda_i^* \neq 0$ are called **Support Vectors** — they lie exactly on the margin boundary and define $\omega^*$. All other points have no influence on the solution.

---

### General (Soft-Margin) SVM

**What if the data is not linearly separable (no margin)?**

There is a tradeoff between:
1. Choosing a large margin $\gamma$
2. How many points **violate** the margin

**Margin violation** at $(x_i, y_i)$: $1 - y_i(\tilde{\omega} \cdot x_i - \tilde{c}) > 0$

**Hinge Loss**:

$$\ell(\tilde{\omega}, \tilde{c}, x_i, y_i) = (1 - y_i(\tilde{\omega} \cdot x_i - \tilde{c}))_+ = g(y_i(\tilde{\omega} \cdot x_i - \tilde{c}))$$

where $g(u) = (1-u)_+$ — this is the **hinge loss**, a convex upper bound on the 0-1 loss.

**Goal**: Optimize margin on most points and penalize margin violations.

**Algo 2 — General (Soft-Margin) SVM**: Pick $\lambda \geq 0$:

$$\min_{\omega, c} \; \lambda \|\omega\|^2 + \frac{1}{n}\sum_i (1 - y_i(\omega \cdot x_i - c))_+$$

This is equivalent to:

$$\min_{\omega,c} \; \hat{R}(\omega, c) + \lambda \|\omega\|^2 \quad \leftarrow \text{Regularized Hinge Empirical Risk}$$

The parameter $\lambda$ is selected by **cross-validation**. It controls the tradeoff: large $\lambda$ → force large margin (penalize $\|\omega\|$); small $\lambda$ → allow more margin violations.

**Remark**: Even when there is a margin in the data, the General SVM might give a **better tradeoff** by allowing a few margin violations in exchange for a classifier that generalizes better.

---

### Slack Variables Formulation

The soft-margin SVM is commonly presented with **slack variables** $\delta_i \geq 0$:

$$\min_{\omega, c, \delta \geq 0} \; \lambda \|\omega\|^2 + \frac{1}{n}\sum_{i=1}^n \delta_i$$

$$\text{s.t.} \quad y_i(\omega \cdot x_i - c) \geq 1 - \delta_i$$

**Equivalence**: The optimal $\delta_i$ for any fixed $(\omega, c)$ is:
- $\delta_i = 0$ if $1 - y_i(\omega \cdot x_i - c) \leq 0$ (no violation)
- $\delta_i = 1 - y_i(\omega \cdot x_i - c)$ if $1 - y_i(\omega \cdot x_i - c) > 0$ (violation)

So $\delta_i \triangleq (1 - y_i(\omega \cdot x_i - c))_+$, recovering Algo 2.

---

## Summary Table

| Method | Type | Loss | Decision Boundary |
|---|---|---|---|
| Gaussian Mixture (equal $\Sigma$) | Generative | — | Linear (Fisher's LDA) |
| Gaussian Mixture (unequal $\Sigma$) | Generative | — | Conic section |
| Logistic Regression | Mixed Gen/Disc | Log loss | $\{f(x)=0\}$ |
| Perceptron | Discriminative | $(-u)_+$ | Hyperplane |
| Hard-Margin SVM | Discriminative | Hinge $(1-u)_+$ | Max-margin hyperplane |
| Soft-Margin SVM | Discriminative | Regularized hinge | Approx. max-margin |

**Key relationships**:
- The **Bayes Classifier** $h^*(x) = \operatorname{argmax}_j \Pr(Y=j \mid x)$ is the optimal classifier at the population level.
- **Surrogate losses** (logistic, hinge, perceptron) are convex upper bounds on 0-1 loss that enable gradient-based optimization.
- **Large margin** → better generalization (Perceptron convergence theorem shows $T \leq r^2/\gamma^2$).
- **Support Vectors** are the critical points that define the SVM solution; all other training points are irrelevant.
