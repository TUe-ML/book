(regression_regularization_exercises)=
# Exercises

## Ridge vs. Lasso: True or False
Which statements are true regarding the solutions to the Ridge and Lasso Regression objectives,
using the same regularization weight $\lambda>0$:
$$\min_{\bm\beta}\lVert y-X\bm\beta\rVert^2+\lambda\lVert\bm\beta\rVert^2 \qquad \text{(Ridge Regression)}$$
$$\min_{\bm\beta}\lVert y-X\bm\beta\rVert^2+\lambda\lvert\bm\beta\rvert \qquad \text{(Lasso)}$$

Select all statements that are true.

- The Lasso regression vector $\bm\beta_{L_1}$ is likely more sparse than $\bm\beta_{L_2}$.
- The Lasso optimization iterates do not always converge.
- The solution of Ridge Regression is generally faster to compute than the one of Lasso.
- There might be infinitely many solutions to the Ridge Regression optimization problem.

````{dropdown} Solution
**True: The Lasso regression vector $\bm\beta_{L_1}$ is likely more sparse than $\bm\beta_{L_2}$.**
The $L_1$ penalty's unit ball has corners on the coordinate axes, so the optimum of the penalized
objective is likely to land exactly on one of those corners, setting some coordinates of $\beta$ to
exactly zero. The $L_2$ penalty's unit ball is a smooth sphere with no corners, so Ridge shrinks
coefficients toward zero without typically setting them to exactly zero.

**False: The Lasso optimization iterates do not always converge.**
Lasso is convex, and although the $L_1$ penalty is not differentiable at zero, it is optimized with
coordinate descent, which is a theoretically well-founded procedure: minimizing exactly along one
coordinate at a time is guaranteed to converge to the global optimum for this class of objectives
(a convex, differentiable data-fit term plus a coordinate-wise separable, convex penalty).

**True: The solution of Ridge Regression is generally faster to compute than the one of Lasso.**
Ridge Regression has a closed-form solution $\bm\beta_{L_2}=(X^\top X+\lambda I)^{-1}X^\top y$,
computable directly via a matrix inversion (or solving one linear system). Lasso has no such
closed form because of the non-differentiable $L_1$ penalty, and instead needs an iterative
procedure like coordinate descent, which is generally slower.

**False: There might be infinitely many solutions to the Ridge Regression optimization problem.**
For any $\lambda>0$, the matrix $X^\top X+\lambda I$ is invertible (it is positive definite, since
$X^\top X$ is positive semi-definite and $\lambda I$ is positive definite), regardless of whether
$X^\top X$ itself is invertible. Hence Ridge Regression always has a unique solution for
$\lambda>0$. Infinitely many solutions can only occur for the unregularized case $\lambda=0$, when
$X^\top X$ is singular (e.g. when $p>n$).
````

## Comparing the Ridge and Lasso Penalty
Which statements are true regarding the solutions to the Ridge and Lasso Regression objectives,
using the same regularization weight $\lambda>0$:
$$\min_{\bm\beta}\lVert y-X\bm\beta\rVert^2+\lambda\lVert\bm\beta\rVert^2 \qquad \text{(Ridge Regression)}$$
$$\min_{\bm\beta}\lVert y-X\bm\beta\rVert^2+\lambda\lvert\bm\beta\rvert \qquad \text{(Lasso)}$$

Of the two regression objectives, ___ penalizes large absolute values ($\lvert\beta_s\rvert>1$) in
$\beta$ more (or just as much).

Of the two regression objectives, ___ penalizes small absolute values ($\lvert\beta_s\rvert<1$) in
$\beta$ more (or just as much).

````{dropdown} Solution
The Ridge penalty term for a single coefficient is $\beta_s^2$, and the Lasso penalty term is
$\lvert\beta_s\rvert$. Comparing the two functions:
- For $\lvert\beta_s\rvert>1$ (e.g. $\beta_s=2$), we have $\beta_s^2>\lvert\beta_s\rvert$ (here
  $4>2$): **Ridge** penalizes large absolute values more.
- For $\lvert\beta_s\rvert<1$ (e.g. $\beta_s=0.5$), we have $\beta_s^2<\lvert\beta_s\rvert$ (here
  $0.25<0.5$): **Lasso** penalizes small absolute values more.
- At $\lvert\beta_s\rvert=1$ the two penalties agree, since $1^2=1$.

This is the intuitive explanation for why Lasso tends to produce sparse solutions: it penalizes
small, nonzero coefficients relatively harshly compared to Ridge, giving the optimizer a strong
incentive to push them all the way to exactly zero rather than leaving them small. Ridge, on the
other hand, penalizes large coefficients disproportionately more, so it strongly discourages any
single coefficient from growing large, but has comparatively little incentive to eliminate small
coefficients entirely.
````

## Effect of Increasing $\lambda$ in Ridge Regression
Consider the Ridge Regression objective:
$$\min_{\bm\beta\in\mathbb{R}^p} RSS_{L_2}(\bm\beta) = \lVert y-X\bm\beta\rVert^2+\lambda\lVert\bm\beta\rVert^2.$$

Starting from $\lambda=0$, as we increase $\lambda$:
- the training RSS will ___,
- the testing RSS will ___,
- the squared bias of the obtained model will ___, and
- the variance of the obtained model will ___.

````{dropdown} Solution
- **The training RSS will increase.** At $\lambda=0$, $\bm\beta$ is exactly the least-squares
  solution, which by definition minimizes the training RSS. Any $\lambda>0$ shrinks $\bm\beta$
  away from that minimizer to also reduce $\lVert\bm\beta\rVert^2$, which can only increase (or
  at best not decrease) the training RSS.
- **The testing RSS will first decrease, then increase.** For small $\lambda$, shrinking
  $\bm\beta$ reduces overfitting to the training data (reduces variance) faster than it introduces
  bias, so the testing RSS tends to improve. Once $\lambda$ grows large enough that the model is
  too constrained to capture the true signal (underfitting dominates), the testing RSS increases
  again. This gives the typical U-shaped test error curve as a function of $\lambda$.
- **The squared bias of the obtained model will increase.** Shrinking $\bm\beta$ toward zero
  moves the fitted model systematically away from the best possible fit to the true function,
  which is exactly what an increase in bias means.
- **The variance of the obtained model will decrease.** Shrinking $\bm\beta$ makes the fitted
  model less sensitive to the particular noise in the training sample: refitting on a different
  training set would yield a more similar $\bm\beta$ the more it is shrunk, i.e. lower variance.

This is the standard bias-variance trade-off induced by the regularization weight $\lambda$: it
trades a monotonic increase in bias for a monotonic decrease in variance, and the testing error
(which combines both) is minimized at some intermediate value of $\lambda$.
````



## Practical Exercise: The Overconfident Model at HomeQuant
Six months later, HomeQuant expands into the broader California housing market. To see what
happens when a model has far more features than data points, let's artificially blow up the
feature count: take HomeQuant's 8 real numeric features (income, house age, room and bedroom
counts, population, occupancy, latitude, longitude) and generate every polynomial and
interaction term up to degree 4, going from 8 features to 494. HomeQuant fits a plain linear
regression on the first 60 sales on record. A senior executive proudly reports in a meeting:
*"Training error is nearly zero! But somehow our real-world pricing errors have gotten worse,
and buyers keep complaining that prices are wildly off for houses slightly outside our usual
size range."*

**Question 1.** Using what you know about the bias-variance trade-off, explain to the executive
what is likely happening here. Is this model suffering from high bias or from high variance? How
would you check which one it is in practice, using only the data you already have?

The cell below sets this up using the real California housing dataset
(`sklearn.datasets.fetch_california_housing`): 494 polynomial and interaction features
engineered from 8 real numeric features, and only 60 training sales.


```python
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso

housing = fetch_california_housing()  # real data; downloads on first use
rng = np.random.default_rng(seed=0)
shuffled = rng.permutation(len(housing.data))
n_train, n_test = 60, 500
train_idx, test_idx = shuffled[:n_train], shuffled[n_train:n_train + n_test]

poly = PolynomialFeatures(degree=4, include_bias=False)
X_train_poly = poly.fit_transform(housing.data[train_idx])
X_test_poly = poly.transform(housing.data[test_idx])

scaler = StandardScaler().fit(X_train_poly)
X_train = scaler.transform(X_train_poly)
X_test = scaler.transform(X_test_poly)
y_train, y_test = housing.target[train_idx], housing.target[test_idx]  # $100,000s
n_features = X_train.shape[1]
```

**Task 1.** Implement `train_test_rss` below, then use it to check the training and test RSS of
the executive's model: an ordinary least-squares (unregularized) linear regression. With
$p=494$ features but only $n=60$ training sales, what do you expect to see?


```python
def train_test_rss(model, X_train, y_train, X_test, y_test):
    """Fit `model` on the training data and return (train_rss, test_rss)."""
    # Your implementation here: fit the model, then compute the RSS (sum of squared
    # residuals) on both the training and the test data.
    train_rss = 0.0
    test_rss = 0.0

    return train_rss, test_rss

unregularized = LinearRegression()
train_rss, test_rss = train_test_rss(unregularized, X_train, y_train, X_test, y_test)
print(f"Unregularized: train RSS = {train_rss:.4f}, test RSS = {test_rss:.2f}")

predicted_millions = unregularized.predict(X_test) * 0.1  # $100,000s to $ millions
actual_millions = y_test * 0.1
print(f"Predicted prices range from ${predicted_millions.min():.1f}M to "
      f"${predicted_millions.max():.1f}M; actual prices range from "
      f"${actual_millions.min():.2f}M to ${actual_millions.max():.2f}M")
```

**Question 2.** A colleague suggests: *"Let's just add an $L_2$ penalty, Ridge Regression,
problem solved."* Would you expect this to help?

**Task 2.** Starting from Task 1's unregularized baseline, sweep the regularization weight λ
over several orders of magnitude, recording the train and test RSS at each value with
`train_test_rss` from Task 1. Plot both curves against λ (log scale): this is how you'd
actually pick λ in practice, rather than "just knowing" it.


```python
import matplotlib.pyplot as plt

alphas = np.logspace(-2, 5, 20)

# Your implementation here: for every alpha in `alphas`, fit Ridge(alpha=alpha) on the
# training data and compute its train/test RSS with train_test_rss(...) from Task 1.
# Store the results so you can plot how train/test RSS change as lambda increases.
train_rss_by_alpha = [0.0 for _ in alphas]
test_rss_by_alpha = [0.0 for _ in alphas]

plt.plot(alphas, train_rss_by_alpha, marker="o", label="train RSS")
plt.plot(alphas, test_rss_by_alpha, marker="o", label="test RSS")
plt.xscale("log")
plt.yscale("log")
plt.xlabel(r"$\lambda$ (Ridge alpha)")
plt.ylabel("RSS (log scale)")
plt.legend()
plt.show()
```

**Question 3.** Your manager wants a single number for λ that you'll "just know is right"
from experience. How would you actually choose λ in practice, and why should you expect the
best λ to make your model less accurate on the training data it was fit on?

**Question 4.** What about Lasso instead? Would you recommend it here, and why? (Think about
what it means to have hundreds of correlated polynomial and interaction terms derived from just
8 base features.)

**Task 3.** Fit a Lasso model and count how many of its ~494 features end up exactly zero. How
does its test RSS compare to your best Ridge model from Task 2?


```python
lasso = Lasso(alpha=0.2)

# Your implementation here: fit `lasso` on the training data, then count how many of
# its coefficients are exactly zero vs. nonzero, and compute its test RSS.
num_zero_coefficients = 0
num_nonzero_coefficients = 0
lasso_test_rss = 0.0

print(f"Lasso set {num_zero_coefficients} of {n_features} coefficients to exactly zero, "
      f"keeping {num_nonzero_coefficients} nonzero (test RSS = {lasso_test_rss:.2f}).")
```

````{dropdown} Solution
This exercise will be solved together with the TA during the exercise session.
````