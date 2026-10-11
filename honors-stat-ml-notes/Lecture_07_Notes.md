# Lecture 07: Performance Analysis & Generalization Theory

---

## 1. Setup: True Risk vs. Empirical Risk

### 1.1 The Classification Setting

Consider binary classification with $Y \in \{+1, -1\}$ and a hypothesis class:

$$\mathcal{H} = \{h : \mathcal{X} \to \mathcal{Y}\}, \quad h(x) = \operatorname{sign}(\omega \cdot x + c)$$

Given a loss function (e.g. 0-1 loss):

$$\ell(h(x), y) = \mathbf{1}\{h(x) \neq y\}$$

**True Risk** (what we actually care about — performance on future data):

$$R(h) = \mathbb{E}_{X,Y \sim P_{XY}} \ell(h(X), Y)$$

**Empirical Risk** (what we can compute from the training data $\{(x_i, y_i)\}_{i=1}^n$ i.i.d. $\sim P_{XY}$):

$$\hat{R}(h) = \frac{1}{n} \sum_{i=1}^n \ell(h(x_i), y_i)$$

**Empirical Risk Minimizer (ERM):**

$$\hat{h} = \operatorname{argmin}_{h \in \mathcal{H}} \hat{R}(h) \quad \leftarrow \text{the ML solution}$$

**Central question:** What is $R(\hat{h})$? How well does $\hat{h}$ perform on *future* data it hasn't seen?

---

## 2. The Generalization Problem

### 2.1 What We Want

Let $h^* = \operatorname{argmin}_{h \in \mathcal{H}} R(h)$ be the **best classifier in $\mathcal{H}$** (best possible true risk).

We want to know: is $R(\hat{h}) \approx R(h^*)$?

In other words: does the ERM $\hat{h}$ (which minimizes the *empirical* risk) also achieve close to the *true* best risk?

This is the **generalization** question. If $\hat{R}(h) \approx R(h)$ for all $h$, then:
- The minimizer $\hat{h}$ of $\hat{R}$ will be close to $h^*$
- At minimum, $R(\hat{h}) \approx R(h^*)$

### 2.2 Key Lemma: Uniform Convergence Implies Generalization

**Theorem:** Suppose $|\hat{R}(h) - R(h)| \leq \varepsilon$ for **all** $h \in \mathcal{H}$. Then:

$$R(\hat{h}) \leq R(h^*) + 2\varepsilon$$

**Proof (three lines):**

$$R(\hat{h}) \leq \hat{R}(\hat{h}) + \varepsilon \quad \text{(by uniform bound, applied to } \hat{h}\text{)}$$
$$\leq \hat{R}(h^*) + \varepsilon \quad \text{(since } \hat{h} \text{ minimizes } \hat{R}\text{, so } \hat{R}(\hat{h}) \leq \hat{R}(h^*)\text{)}$$
$$\leq R(h^*) + \varepsilon + \varepsilon \quad \text{(by uniform bound, applied to } h^*\text{)}$$

So: if the empirical risk is within $\varepsilon$ of the true risk **for every classifier** in $\mathcal{H}$, then the ERM $\hat{h}$ is within $2\varepsilon$ of the Bayes-optimal classifier in $\mathcal{H}$.

**Goal:** Establish this uniform bound $|\hat{R}(h) - R(h)| \leq \varepsilon$ for some $\varepsilon = \varepsilon(n, \mathcal{H})$ that depends on the sample size and complexity of $\mathcal{H}$.

---

## 3. Concentration Inequalities

### 3.1 Intuition: Law of Large Numbers

The empirical risk is an average:

$$\hat{R}(h) = \frac{1}{n} \sum_{i=1}^n \ell(h(x_i), y_i)$$

Since the samples are i.i.d., by the **Law of Large Numbers**:

$$\mathbb{E}\hat{R}(h) = R(h), \quad \text{and} \quad \hat{R}(h) \xrightarrow{n \to \infty} R(h) \text{ (in probability)}$$

But we need *finite $n$ guarantees* with explicit rates.

### 3.2 The Central Limit Theorem Route

For a scalar average $\bar{X} = \frac{1}{n}\sum_{i=1}^n X_i$ with $\mathbb{E} X_i = \mu$ and $\text{Var}(X_i) = \sigma^2$:

$$\sigma_{\bar{X}}^2 = \frac{\sigma^2}{n}$$

By CLT (as $n \to \infty$):

$$\frac{\bar{X} - \mu}{\sigma_{\bar{X}}} \xrightarrow{d} Z \sim \mathcal{N}(0,1)$$

For a Gaussian, the tail decays like:

$$\Pr(|Z| \geq t) \leq c_0 e^{-c_1 t^2} \quad (\text{since the pdf} \propto e^{-z^2/2})$$

Substituting $\varepsilon = t \cdot \sigma_{\bar{X}}$ so that $t^2 = \varepsilon^2/\sigma_{\bar{X}}^2 = \varepsilon^2 n / \sigma^2$:

$$\Pr\!\left(|\bar{X} - \mu| \geq \varepsilon\right) \leq c_0 \, e^{-c_1 \varepsilon^2 n / \sigma^2}$$

This is an approximation (valid as $n \to \infty$). For finite $n$ we need stronger tools.

### 3.3 Hoeffding / Chernoff Bounds (Exact, Finite $n$)

**Proposition:** For "nice" (bounded) random variables $X_i \in [0,1]$, for **all** $n$:

$$\Pr\!\left(|\bar{X} - \mu| \geq \varepsilon\right) \leq c_0 \, e^{-c_1 \varepsilon^2 n}$$

This is the **Hoeffding bound** (also related to Chernoff bounds). It holds exactly for all $n$, not just asymptotically. Key features:
- The probability of deviation $\geq \varepsilon$ decays **exponentially** in $n$
- The rate is $\varepsilon^2 n$ — so to halve the error, you need 4x the data
- No $\sigma^2$ dependence when variables are in $[0,1]$ (subsumed into constants)

**Application to risk:** Fix a single $h$. Then $\ell(h(x_i), y_i) \in \{0,1\}$ (bounded), so by Hoeffding:

$$\Pr\!\left(|\hat{R}(h) - R(h)| \geq \varepsilon\right) \leq c_0 \, e^{-c_1 \varepsilon^2 n}$$

For any fixed $h$, the empirical risk concentrates around the true risk at an exponential rate.

---

## 4. Generalization for Finite Hypothesis Classes

### 4.1 The Problem with a Single Bound

The single-$h$ bound above is not enough. We need:

$$\Pr\!\left(\forall h \in \mathcal{H},\; |\hat{R}(h) - R(h)| < \varepsilon\right) \geq 1 - \delta$$

It's easier to bound the complement:

$$\Pr\!\left(\exists h \in \mathcal{H},\; |\hat{R}(h) - R(h)| \geq \varepsilon\right) \leq \delta$$

### 4.2 Union Bound

Assume $|\mathcal{H}| < \infty$. By the **Union Bound** ($\Pr(A \cup B) \leq \Pr(A) + \Pr(B)$):

$$\Pr\!\left(\exists h,\; |\hat{R}(h) - R(h)| \geq \varepsilon\right) \leq \sum_{h \in \mathcal{H}} \Pr\!\left(|\hat{R}(h) - R(h)| \geq \varepsilon\right)$$

Applying the Hoeffding bound to each term:

$$\leq |\mathcal{H}| \cdot c_0 \, e^{-c_1 \varepsilon^2 n}$$

We want this to be at most $\delta$ (e.g. $\delta = 0.01$ or $0.05$):

$$c_0 |\mathcal{H}| \cdot e^{-c_1 \varepsilon^2 n} \leq \delta$$

### 4.3 Solving for $\varepsilon$

Inverting:

$$e^{-c_1 \varepsilon^2 n} \leq \frac{\delta}{c_0 |\mathcal{H}|}$$

$$-c_1 \varepsilon^2 n \leq \log\frac{\delta}{c_0 |\mathcal{H}|}$$

$$c_1 \varepsilon^2 n \geq \log\frac{c_0 |\mathcal{H}|}{\delta} = \log|\mathcal{H}| + \log\frac{c_0}{\delta}$$

$$\boxed{\varepsilon \geq \sqrt{\frac{\log|\mathcal{H}| + \log(c_0/\delta)}{c_1 \cdot n}}}$$

### 4.4 The Finite-H Generalization Theorem

**Theorem:** For $\varepsilon \geq \sqrt{(\log|\mathcal{H}| + \log(c_0/\delta)) / (c_1 n)}$:

$$\Pr\!\left(\forall h \in \mathcal{H},\; |\hat{R}(h) - R(h)| < \varepsilon\right) \geq 1 - \delta$$

**Corollary:** With probability at least $1 - \delta$:

$$R(\hat{h}) - R(h^*) \leq 2\varepsilon \propto \sqrt{\frac{\log|\mathcal{H}|}{n}}$$

**Interpretation:** The generalization gap shrinks as:
- $n$ increases (more data → better generalization)
- $\log|\mathcal{H}|$ decreases (simpler class → better generalization)
- The dependence is on the **log** of the class size — so even exponentially large $|\mathcal{H}|$ is manageable if $\log|\mathcal{H}|$ is small

**Example:** $\mathcal{H} = \{h_1, h_2, h_3\}$. The event $\exists h,\; |\hat{R}(h)-R(h)| \geq \varepsilon$ equals:

$$\{|\hat{R}(h_1)-R(h_1)| \geq \varepsilon\} \cup \{|\hat{R}(h_2)-R(h_2)| \geq \varepsilon\} \cup \{|\hat{R}(h_3)-R(h_3)| \geq \varepsilon\}$$

The union bound says the probability of any one being bad is at most the sum. If each is $\leq \delta/3$, the union is $\leq \delta$.

---

## 5. Generalization for Infinite Hypothesis Classes

### 5.1 The Problem

Linear classifiers $\mathcal{H} = \{h(x) = \operatorname{sign}(\omega \cdot x - c) : \omega \in \mathbb{R}^d, c \in \mathbb{R}\}$ have $|\mathcal{H}| = \infty$, so the union bound above doesn't directly apply.

### 5.2 Projection onto the Sample

**Key insight:** On a finite training set $S_X = \{x_1, \ldots, x_n\}$, many distinct $h$'s in $\mathcal{H}$ produce the **same predictions**. We only care about behavior on the training data.

Define the **projection** of $\mathcal{H}$ onto $S_X$:

$$\mathcal{H}|_{S_X} = \{\text{distinct labelings } (h(x_1), \ldots, h(x_n)) : h \in \mathcal{H}\}$$

Then $|\mathcal{H}|_{S_X}|$ is finite even when $|\mathcal{H}| = \infty$.

**Example:** $\mathcal{H}$ = one-sided threshold classifiers on $\mathbb{R}$: $h_c(x) = \mathbf{1}\{x \geq c\}$. There are infinitely many thresholds $c$, but on $n$ points there are only $n+1$ distinct labelings (the threshold can fall in one of $n+1$ gaps between points). So $|\mathcal{H}|_{S_X}| = n+1$ even though $|\mathcal{H}| = \infty$.

### 5.3 Effective Complexity: VC Dimension

Define the **effective complexity** of $\mathcal{H}$ as:

$$|\mathcal{H}^*_S| = \max_{S_X} |\mathcal{H}|_{S_X}|$$

(the largest number of distinct labelings over any sample of size $n$). This is related to the **VC dimension**.

**Generalization bound for infinite $\mathcal{H}$:**

$$\Pr\!\left(|R(\hat{h}) - R(h)| \leq 2\varepsilon\right) \geq 1 - \delta \quad \text{for } \varepsilon \geq \sqrt{\frac{\log|\mathcal{H}^*_{S_X}| + \log(1/\delta)}{n}}$$

### 5.4 Bounds for Linear Classifiers

For linear classifiers in $\mathbb{R}^d$:

$$\mathcal{H} = \{h(x) = \operatorname{sign}(\omega \cdot x - c) : \omega \in \mathbb{R}^d, c \in \mathbb{R}\}$$

It can be shown that $|\mathcal{H}^*_S| \leq n^{O(d)}$ (polynomial in $n$, exponential in $d$). Substituting:

$$\varepsilon \propto \sqrt{\frac{d \cdot \log n + \log(1/\delta)}{n}}$$

The generalization error scales as $\sqrt{d/n}$ (up to log factors). To achieve $\varepsilon$ accuracy, you need roughly $n \sim d/\varepsilon^2$ samples.

### 5.5 Large Margin Improves the Bound

**Regularization = restricting to a smaller subclass of $\mathcal{H}$.** If we restrict to classifiers achieving margin $\geq \gamma > 0$ (recall from Lecture 05: $\gamma = \min_i y_i \omega^T x_i / \|\omega\|$), the effective complexity shrinks.

**Theorem:** For large-margin linear classifiers (margin $\geq \gamma$):

$$\varepsilon \propto \sqrt{\frac{d \cdot \log n + \log(1/\delta)}{n \cdot \gamma^2}}$$

The margin $\gamma$ appears in the denominator: a **larger margin** means better generalization. This is the theoretical justification for maximizing the margin in SVMs — it directly controls the generalization error, not just the training fit.

### 5.6 Comparison: Neighborhood Methods Suffer the Curse

**Note:** For $k$-NN, tree-based methods, and other neighborhood classifiers:

$$\varepsilon \propto n^{-1/O(d)}$$

This is **polynomial** in $n^{-1}$ with an exponent that decays with $d$ — much slower than $\sqrt{1/n}$ for linear classifiers. This is a formal manifestation of the **Curse of Dimensionality** introduced in Lecture 06: neighborhood methods need exponentially more data in high dimensions, while linear classifiers (with margin) need only polynomially more.

| Method | Generalization rate |
|--------|-------------------|
| Linear classifiers (general) | $\sim \sqrt{d \log n / n}$ |
| Linear classifiers (margin $\gamma$) | $\sim \sqrt{d \log n / (n\gamma^2)}$ |
| Nearest Neighbor / Trees | $\sim n^{-1/O(d)}$ |

---

## 6. Performance Analysis for Regression

### 6.1 OLS Setup

Consider the linear regression model:

$$Y = \omega^* \cdot X + \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, \sigma^2), \quad \varepsilon \perp\!\!\!\perp X$$

where $\omega^* \in \mathbb{R}^d$ is the true (unknown) parameter.

**OLS estimator:**

$$\hat{\omega} = \operatorname{argmin}_{\omega \in \mathbb{R}^d} \frac{1}{n} \sum_{i=1}^n (y_i - \omega \cdot x_i)^2$$

Closed form (from Lecture 04): $\hat{\omega} = (\mathbb{X}^T \mathbb{X})^{-1} \mathbb{X}^T \mathbb{Y}$

**True risk (squared loss):**

$$R(\omega) = \mathbb{E}(Y - \omega \cdot X)^2$$

### 6.2 The Excess Risk

The excess risk (how much worse $\hat{\omega}$ is than $\omega^*$) is:

$$R(\hat{\omega}) - R(\omega^*) = (\hat{\omega} - \omega^*)^T \Sigma (\hat{\omega} - \omega^*)$$

where $\Sigma = \mathbb{E}[XX^T]$ is the feature covariance matrix. This equals:

$$= \mathbb{E}\!\left[(f_{\hat{\omega}}(x) - f_{\omega^*}(x))^2\right], \quad f_\omega(x) = \omega \cdot x$$

It measures the average squared difference between the fitted linear function and the true one.

### 6.3 The OLS Generalization Bound

**Theorem:**

$$\boxed{\mathbb{E}_S\!\left[R(\hat{\omega}) - R(\omega^*)\right] \leq \frac{d \cdot \sigma^2}{n}}$$

The excess risk of OLS scales as $d/n$: dimension divided by sample size. To achieve excess risk $\leq \varepsilon$, you need $n \gtrsim d/\varepsilon$.

### 6.4 Intuition: Linear Regression as Averaging in $\mathbb{R}^d$

**Why $d\sigma^2/n$?** Consider the simpler problem of estimating a mean vector.

Let $X = (X_{(1)}, \ldots, X_{(d)})$ with $\mathbb{E} X = \mu$ and $\text{Cov}(X) = \sigma^2 I_d$ (variance $\sigma^2$ per coordinate). The sample mean is $\bar{X} = \frac{1}{n}\sum_{i=1}^n X_i$.

The mean squared error of $\bar{X}$ as an estimator of $\mu$:

$$\mathbb{E}\|\bar{X} - \mu\|^2 = \sum_{i=1}^d \mathbb{E}(\bar{X}_{(i)} - \mu_{(i)})^2 = \sum_{i=1}^d \frac{\sigma^2}{n} = \frac{d\sigma^2}{n}$$

Each of the $d$ coordinates contributes $\sigma^2/n$ variance, so the total is $d\sigma^2/n$.

**The connection:** Under $\Sigma = \mathbb{E}XX^T = I_d$ (for simplicity), the OLS estimator is:

$$\hat{\omega} = (\mathbb{X}^T \mathbb{X})^{-1} \mathbb{X}^T \mathbb{Y} \approx \frac{1}{n} \sum_{i=1}^n Y_i X_i$$

Note that:

$$\mathbb{E}[X \cdot Y] = \mathbb{E}[X(X^T \omega^* + \varepsilon)] = (\mathbb{E}XX^T)\omega^* + \mathbb{E}X \cdot \mathbb{E}\varepsilon = I_d \omega^* + 0 = \omega^*$$

So $\hat{\omega} \approx \frac{1}{n}\sum_i Y_i X_i$ is an **average in $\mathbb{R}^d$** that estimates $\mathbb{E}[XY] = \omega^*$. Its variance is exactly $d\sigma^2/n$ by the above calculation.

**∴ Linear regression is computing an average in $\mathbb{R}^d$**, and the bound $d\sigma^2/n$ is exactly the variance of that average.

---

## 7. Summary

### 7.1 The Generalization Pipeline

$$\underbrace{R(\hat{h}) - R(h^*)}_{{\text{generalization gap}}} \leq 2\varepsilon \quad \text{with prob. } \geq 1 - \delta$$

where $\varepsilon = \varepsilon(n, \mathcal{H})$ depends on sample size and class complexity.

**To get $\varepsilon$ small:**
1. More data ($n \uparrow$)
2. Simpler hypothesis class ($|\mathcal{H}|$ or effective complexity $\downarrow$)
3. Larger margin ($\gamma \uparrow$) for linear classifiers

### 7.2 Key Bounds Reference

| Setting | Generalization Bound |
|---------|---------------------|
| Finite $\mathcal{H}$ | $\varepsilon \propto \sqrt{\log|\mathcal{H}|/n}$ |
| Infinite $\mathcal{H}$ (general) | $\varepsilon \propto \sqrt{\log|\mathcal{H}^*_S|/n}$ |
| Linear classifiers in $\mathbb{R}^d$ | $\varepsilon \propto \sqrt{d \log n / n}$ |
| Large-margin linear (margin $\gamma$) | $\varepsilon \propto \sqrt{d \log n / (n\gamma^2)}$ |
| Nearest neighbor / Trees | $\varepsilon \propto n^{-1/O(d)}$ (Curse of Dim.) |
| OLS Regression (expected excess risk) | $\mathbb{E}[R(\hat{\omega}) - R(\omega^*)] \leq d\sigma^2/n$ |

### 7.3 Conceptual Connections

- **Regularization** (Lecture 04) restricts $\mathcal{H}$ to a smaller class → reduces $\log|\mathcal{H}^*_S|$ → better generalization bound
- **Large Margin / SVM** (Lecture 05) maximizes $\gamma$ → enters the denominator of the bound → better generalization
- **Curse of Dimensionality** (Lecture 06, neighborhood methods) shows up formally: $\varepsilon \propto n^{-1/O(d)}$, far worse than $\sqrt{d/n}$ for linear models
- **Bias-Variance tradeoff** (Lecture 04) is an instance of this theory: large $\mathcal{H}$ → low bias but harder to control $\varepsilon$; small $\mathcal{H}$ → easy to generalize but higher $R(h^*)$

The generalization gap $R(\hat{h}) - R(h^*)$ measures variance (estimation error); $R(h^*) - R(h_{\text{Bayes}})$ measures bias (approximation error). Good ML = controlling both.
