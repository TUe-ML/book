(reg_exercises)=
# Exercises

## Design Matrices and Regression Function Fitting
Imagine you have a dataset which consists of three datapoints (of course this is super unrealistic but if we want to go through regression step by step, then we need a really small example). The data is listed in the following table:

|$D$| $x$ | $y$ |
|---|-------|-----|
| 1 | 5 | 2 |
| 2 | 3 | 5 |
| 3 | 1 | 3 |

In this exercise, you are asked to fit a regression function for specified function classes.
That is, you will have to create the design matrix and compute the global optimizer(s) $\beta$ of the regression objective. You can and probably should use Python to solve the system of linear equations to get $\beta$, but it's a good exercise to compute the design matrix by hand. Plot the regression function.

### Affine Function ($k=1$)
Fit an affine function to the data
\begin{align*} f(x) = \beta_1 x + \beta_0.\end{align*}

````{dropdown} Solution
Affine functions are decomposed into an inner product of a basis function and the regression parameter vector $\bm\beta$ by

\begin{align*}
f(x)=\beta_1 x + \beta_0 =\phi_{aff}(x)^\top \bm\beta,
\end{align*}
where the feature transformation is defined as

$$\phi_{aff}(x)=\begin{pmatrix}1\\x\end{pmatrix}.$$
The design matrix is given by
\begin{align*} X = \begin{pmatrix} - & \phi_{aff}(5)^\top & -\\- & \phi_{aff}(3)^\top & -\\
- & \phi_{aff}(1)^\top & -
\end{pmatrix}
=
\begin{pmatrix}
1 & 5\\
1 & 3\\
1 & 1
\end{pmatrix}.
\end{align*}
We have seen in the lecture videos, that the global minimizer(s) $\beta$ which minimize the residual sum of squares are given by the solver(s) of the following system of linear equations:
```{math}
:label: eq_sysbeta
\{\bm\beta\mid X^\top X\bm\beta = X^\top y\}.
```
The minimizer $\beta$ is uniquely defined if $X^\top X$ is invertible. This is here the case.

You can either compute the inverse $(X^\top X)^{-1}$ (via Python or manually) and compute the regression vector $\beta$ by the formula
```{math}
:label: eq_betainv
\bm\beta = (X^\top X)^{-1}X^\top y,
```
or you solve the system of linear equations defined in Eq. {eq}`eq_sysbeta`. We have
$$ X^\top X = \begin{pmatrix}
3 & 9\\
9 & 35
\end{pmatrix},\quad X^\top y = \begin{pmatrix}
10\\28
\end{pmatrix}.$$
Hence, we have to solve the following system of linear equations according to Eq. {eq}`eq_sysbeta`:
\begin{align*}
    3\beta_0 + 9\beta_1 &= 10\\
    9\beta_0 + 35\beta_1 &=28.
\end{align*}
No matter which way you choose, the result should be
$$ \bm\beta \approx \begin{pmatrix}
4.08\\ -0.25
\end{pmatrix}.$$
Hence, the affine regression function is defined as
$$f(x) = -0.25x + 4.08.$$
```{tikz}
\begin{tikzpicture}
\begin{axis}[
width=.8\textwidth,
axis lines = center,
xlabel=$x$, % label x axis
ylabel=$y$, % label y axis
xmin=-5, xmax=12, % set the min and max values of the x-axis
domain=-19:12,
ymin= -1,ymax=6, % set the min and max values of the y-axis
]
\addplot [blue,only marks,  mark = *]
coordinates {
(5,2)
(3,5)
(1,3)
};
\addplot+[magenta,ultra thick,smooth, mark=none]
{-2/5*x+68/15};
\end{axis}
\end{tikzpicture}
```
````

### Polynomial of Degree $k=2$
Fit a polynomial regression function of degree $k=2$ to the data
\begin{align*}
    f(x) = \beta_2 x^2 + \beta_1x + \beta_0.
\end{align*}

````{dropdown} Solution
The feature transformation for a polynomial of degree 2 is here defined as
$$\phi_{p2}(x)=\begin{pmatrix}
1\\x \\\ x^2
\end{pmatrix}.$$
The design matrix is given by
\begin{align*}
X = \begin{pmatrix}
- & \phi_{p2}(5)^\top & -\\
- & \phi_{p2}(3)^\top & -\\
- & \phi_{p2}(1)^\top & -
\end{pmatrix}
=
\begin{pmatrix}
1 & 5 & 25\\
1 & 3 & 9\\
1 & 1 & 1
\end{pmatrix}.
\end{align*}
We calculate the matrices
$$X^\top X = \begin{pmatrix}
3 & 9 & 35\\
9 & 35& 153\\
35 & 153 & 707
\end{pmatrix},\quad X^\top y = \begin{pmatrix}
10\\28\\98
\end{pmatrix},$$
and solve the system of linear equation in Eq. {eq}`eq_sysbeta` or compute the inverse of $X^\top X$ and compute $\beta$ by Eq. {eq}`eq_betainv`. As a result we get
$$\bm\beta \approx \begin{pmatrix}
0.125\\ 3.5\\-0.625
\end{pmatrix}.$$
Hence, the polynomial regression function is
$$f(x)= -0.625x^2 +3.5x +0.125.$$
```{tikz}
\begin{tikzpicture}
\begin{axis}[
width=.8\textwidth,
xlabel=$x$, % label x axis
ylabel=$y$, % label y axis
axis lines=center, %set the position of the axes
xmin=-5, xmax=12, % set the min and max values of the x-axis
domain=-19:12,
ymin= -1,ymax=6, % set the min and max values of the y-axis
]
\addplot [blue,only marks,  mark = *]
coordinates {
(5,2)
(3,5)
(1,3)
};
\addplot+[magenta,ultra thick,smooth, mark=none]
{-0.625*x^2+3.5*x+0.125};
\end{axis}
\end{tikzpicture}
```
````

### Sum of Three Gaussians
Fit a sum of three Gaussians to the data:
\begin{align*}
    f(x)= \beta_1\exp(-(x-5)^2)+\beta_2\exp(-(x-3)^2)+\beta_3\exp(-(x-1)^2).
\end{align*}
The mean values $\mu$, which have to be specified when we choose a Gaussian basis function, are here equal to the three given feature values in the data. This strategy is also often used in practice.

````{dropdown} Solution
The feature transformation when using Gaussian basis functions as stated above is given by
$$\phi_{G3}(x)=\begin{pmatrix}
\exp(-(x-5)^2)\\\exp(-(x-3)^2) \\\ \exp(-(x-1)^2)
\end{pmatrix}.$$
The design matrix is defined as
\begin{align*}
X = \begin{pmatrix}
- & \phi_{G3}(5)^\top & -\\
- & \phi_{G3}(3)^\top & -\\
- & \phi_{G3}(1)^\top & -
\end{pmatrix}
=
\begin{pmatrix}
1 & \exp(-4) &\exp(-16)\\
\exp(-4) & 1 & \exp(-4)\\
\exp(-16) & \exp(-4) & 1
\end{pmatrix}.
\end{align*}
As in previous exercises, we compute the regression parameter vector. As a result we get
$$\bm\beta \approx \begin{pmatrix}
1.91\\ 4.91\\2.91
\end{pmatrix}.$$
Hence, the polynomial regression function is
$$f(x)= 1.91\exp(-(x-5)^2)+4.91\exp(-(x-3)^2)+2.91\exp(-(x-1)^2).$$

```{tikz}
\begin{tikzpicture}
\begin{axis}[
width=.8\textwidth,
xlabel=$x$, % label x axis
ylabel=$y$, % label y axis
axis lines=center, %set the position of the axes
xmin=-5, xmax=12, % set the min and max values of the x-axis
domain=-19:12,samples=200,
ymin= -1,ymax=6, % set the min and max values of the y-axis
]
\addplot [blue,only marks,  mark = *]
coordinates {
(5,2)
(3,5)
(1,3)
};
\addplot+[magenta,ultra thick,smooth, mark=none]
{1.91*exp(-(x-5)^2)+4.912*exp(-(x-3)^2)+2.91*exp(-(x-1)^2)};
\end{axis}
\end{tikzpicture}
```
````

### Underdetermined Polynomial of Degree $k=3$
Fit a polynomial of degree $k=3$:
$$f(x) = \beta_0 +\beta_1 x+ \beta_2x^2 + \beta_3x^3.$$
Note that this results in an underdetermined system, you will hence get a set of regression solvers.

````{dropdown} Solution
The feature transformation for a polynomial of degree 3 is defined as
$$\phi_{p3}(x)=\begin{pmatrix}
1\\x \\\ x^2 \\ x^3
\end{pmatrix}.$$
The design matrix is given by
\begin{align*}
X = \begin{pmatrix}
- & \phi_{p3}(5)^\top & -\\
- & \phi_{p3}(3)^\top & -\\
- & \phi_{p3}(1)^\top & -
\end{pmatrix}
=
\begin{pmatrix}
1 & 5 & 25 & 125\\
1 & 3 & 9 & 27\\
1 & 1 & 1 & 1
\end{pmatrix}.
\end{align*}

We have to solve the system of equations given by $X^\top X \bm\beta = X^\top \vvec{y}$:
\begin{align*}
\begin{pmatrix}
3 &     9 &    35 &   153\\
9 &    35 &   153 &   707\\
35&   153 &   707 &  3369\\
153&  707 &  3369 & 16355
\end{pmatrix} \bm\beta =
\begin{pmatrix}
10\\  28\\  98\\ 388
\end{pmatrix}
\end{align*}
We can solve this system of equations by hand. There are multiple ways to do this. One is to transform the equations above into an upper triangle form. To do this, we divide the first equation by three and subtract the first equation, multiplied accordingly, from the other equations such that the first coefficient is equal to zero.
\begin{align*}
\begin{pmatrix}
1 &    3 &   11.67 &   51\\
0 &    8 &   48 &  248\\
0 &   48 &  298.67& 1584\\
0 &  248 & 1584 & 8552
\end{pmatrix}\bm\beta =
\begin{pmatrix}
3.33\\-2\\-18.67\\-122
\end{pmatrix}
\end{align*}
Now we do the same with the second equation, we divide by eight and subtract the second equation from the ones below, multiplied accordingly.
\begin{align*}
\begin{pmatrix}
1 &    3 &   11.67 &   51\\
0  &   1  &   6 &   31\\
0  &   0  &  10.67 & 96\\
0  &   0  &  96 &  864
\end{pmatrix}\bm\beta =
\begin{pmatrix}
3.33\\ -0.25\\ -6.67\\ -60
\end{pmatrix}
\end{align*}
Now we do the same with the third equation and we get
\begin{align*}
\begin{pmatrix}
1 &    3 &   11.67 &   51\\
0  &   1  &   6 &   31\\
0  &   0  &   1 & 9\\
0  &   0  &   0 &  0
\end{pmatrix}\bm\beta =
\begin{pmatrix}
3.33\\ -0.25\\ -\frac{5}{8}\\ 0
\end{pmatrix}
\end{align*}
We observe that the last equation is always true. We can now solve this system in dependence of $\beta_3$. The last equation states that
$$\beta_2 + 9\beta_3 = -\frac{5}{8} \Leftrightarrow \beta_2 = -\frac{5}{8} -9\beta_3.$$
Substituting $\beta_2$ into the second equation yields $\beta_1 = 3.5 +23\beta_3 $ and substituting $\beta_1$ and $\beta_2$ into the first equation yields $\beta_0 = \frac{1}{8}-15\beta_3$. Hence, the set of all regression solvers is given by
$$\left\{\beta = \begin{pmatrix}\frac{1}{8}-15\beta_3\\3.5 +23\beta_3\\-\frac{5}{8} -9\beta_3\\\beta_3 \end{pmatrix} \mid \beta_3\in\mathbb{R} \right\}$$
The plot below indicates the regression models for three values of $\beta_3$.
```{tikz}
\begin{tikzpicture}
\begin{axis}[
width=.8\textwidth,
xlabel=$x_1$, % label x axis
ylabel=$y$, % label y axis
axis lines=left, %set the position of the axes
xmin=0, xmax=7, % set the min and max values of the x-axis
domain=0:6,
ymax=12, % set the min and max values of the y-axis
legend pos=outer north east]
\addplot+[only marks, black, mark = *] 
coordinates {
(5,2)
(3,5)
(1,3)
};
\addlegendentry{Data Points}
\addplot+[magenta,thick,smooth, mark=none]
{x^3-(5/8+9)*x^2+(7/2+23)*x+1/8-15};
\addlegendentry{$\beta_3=1$}
\addplot+[blue,thick,smooth, mark=none]
{x^3/2-(5/8+9/2)*x^2+(7/2+23/2)*x+1/8-15/2};
\addlegendentry{$\beta_3=0.5$}
\addplot+[green,thick,smooth, mark=none]
{x^3/4-(5/8+9/4)*x^2+(7/2+23/4)*x+1/8-15/4};
\addlegendentry{$\beta_3=0.25$}
\end{axis}
\end{tikzpicture}
```
````

## Practical Exercise: Pricing Houses at HomeQuant
You've just joined a real-estate startup, HomeQuant, as their first data scientist. Your manager shows you a spreadsheet of 20 past home
sales (square footage vs. sale price) and says: *"A colleague of yours suggested we fit a
degree-19 polynomial to it: with exactly 20 data points, that should let it match every single
historical sale exactly. Zero error! Can we ship this to production so we can start pricing new
listings today?"*

**Question 1.** What would you tell your manager? What is actually going on with a "zero error"
model like this, and why might it be a bad idea to use it to price a brand new listing?

The cell below generates HomeQuant's 20 historical sales (`train_sizes`, `train_prices`).


```python
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(seed=0)
train_sizes = np.sort(rng.uniform(50, 250, size=20))  # square meters
train_prices = 50 + 3.0 * train_sizes + rng.normal(scale=40, size=20)  # $1000s

plt.scatter(train_sizes, train_prices)
plt.xlabel("size (m$^2$)")
plt.ylabel("price ($1000s)")
plt.show()
```

**Task 1.** Implement `fit_polynomial_and_predict` below: it should fit a polynomial of the
given `degree` to `(sizes, prices)`, and return its coefficients, its training RSS, and its
predicted price at `query_size`. (`np.polyfit` and `np.polyval` are your friends here.) You'll
reuse this function for the rest of the exercise.


```python
def fit_polynomial_and_predict(sizes, prices, degree, query_size):
    """Fit a degree-`degree` polynomial to (sizes, prices) and return
    (coefficients, training_rss, prediction_at_query_size).
    """
    # Your implementation here
    coefficients = np.zeros(degree + 1)
    training_rss = 0.0
    prediction_at_query_size = 0.0

    return coefficients, training_rss, prediction_at_query_size
```

**Task 2.** Put your colleague's claim to the test. Fit a degree-19 polynomial to
`train_sizes`/`train_prices` with your function, then use the same fit to predict the price of a
900 m² mansion, more than triple the size of anything in the historical data. Print the training
RSS and the mansion prediction.


```python
mansion_size = 900  # square meters, the new listing from the discussion above

# Your implementation here: call `fit_polynomial_and_predict` with degree=19 on
# (train_sizes, train_prices), querying `mansion_size`.
degree19_coefficients, degree19_train_rss, degree19_mansion_prediction = None, 0.0, 0.0

print(f"Degree 19: training RSS = {degree19_train_rss:.2f}, "
      f"predicted price for a {mansion_size} m^2 mansion = {degree19_mansion_prediction:.2f}")
```

**Question 2.** Look at the price your degree-19 model just predicted for the 900 m² mansion.
Should you trust this number? Why or why not?

**Question 3.** What tradeoff are you weighing between a straight line and a higher-degree
polynomial? What would you actually do, in practice, to choose a good degree instead of picking
whichever one looks best on the historical data?

**Task 3.** A near-zero training RSS doesn't tell you how the model behaves on unseen houses.
The cell below (given) draws a second, independent batch of 20 sales from the same process: a
held-out test set. It also plots your fitted degree-19 curve against the train and test points.
Using `degree19_coefficients` from Task 2 (no need to refit), compute the degree-19 model's test
RSS with `np.polyval` and compare it to the training RSS from Task 2.


```python
test_rng = np.random.default_rng(seed=1)
test_sizes = np.sort(test_rng.uniform(50, 250, size=20))
test_prices = 50 + 3.0 * test_sizes + test_rng.normal(scale=40, size=20)

curve_sizes = np.linspace(50, 250, 500)
curve_prices = np.polyval(degree19_coefficients, curve_sizes)

plt.scatter(train_sizes, train_prices, label="train")
plt.scatter(test_sizes, test_prices, label="test", marker="x")
plt.plot(curve_sizes, curve_prices, color="C2", label="degree-19 fit")
plt.ylim(0, 1200)  # zoom in on the data; the curve swings much higher near the edges
plt.xlabel("size (m$^2$)")
plt.ylabel("price ($1000s)")
plt.legend()
plt.show()

# Your implementation here: use `np.polyval` with `degree19_coefficients` (from
# Task 2) to predict prices for `test_sizes`, then compute the degree-19 model's
# RSS on this held-out test set.
degree19_test_rss = 0.0

print(f"Degree 19: training RSS = {degree19_train_rss:.2f}, test RSS = {degree19_test_rss:.2f}")
```

**Task 4.** Instead of eyeballing which degree "looks best," write a function
`evaluate_degrees(train_sizes, train_prices, test_sizes, test_prices, degrees)` that fits each
candidate degree (including `degree=1`, your manager's straight line) on the training data and
returns its training and test RSS. Use it to compare degrees $1,\ldots,15$, plot both curves,
and pick and justify a good degree for HomeQuant.


```python
degrees_to_try = list(range(1, 16))

def evaluate_degrees(train_sizes, train_prices, test_sizes, test_prices, degrees):
    """For every degree in `degrees`, fit on (train_sizes, train_prices) and return
    (train_rss_by_degree, test_rss_by_degree), each evaluated on `test_sizes`/`test_prices`.
    """
    # Your implementation here
    train_rss_by_degree = [0.0 for _ in degrees]
    test_rss_by_degree = [0.0 for _ in degrees]

    return train_rss_by_degree, test_rss_by_degree

train_rss_by_degree, test_rss_by_degree = evaluate_degrees(
    train_sizes, train_prices, test_sizes, test_prices, degrees_to_try
)

plt.plot(degrees_to_try, train_rss_by_degree, marker="o", label="train RSS")
plt.plot(degrees_to_try, test_rss_by_degree, marker="o", label="test RSS")
plt.xlabel("polynomial degree")
plt.ylabel("RSS")
plt.legend()
plt.show()
```

````{dropdown} Solution
This exercise will be solved together with the TA during the exercise session.
````

## Bias and Variance Computation
Consider the following true regression function: $$f^*(x) = \tan(\pi x).$$ Imagine you fit three regression models on i.i.d. data samples $\mathcal{D}_1, \mathcal{D}_2, \mathcal{D}_3$ and obtain the following models:
\begin{align*}
    f_{\mathcal{D}_1}(x) &= x -0.1\\
    f_{\mathcal{D}_2}(x) &= 3x + 0.1\\
    f_{\mathcal{D}_3}(x) &= 5x + 0.2
\end{align*}
Compute for $x_0 = 0.1$ the sample
* bias$^2$ and
* variance.

````{dropdown} Solution
We have $f^*(x_0) = f^*(0) = \tan(0) = 0$. The regression models return
\begin{align*}
f_{\mathcal{D}_1}(x_0) &= f_{\mathcal{D}_1}(0) = 0.2 \\
f_{\mathcal{D}_2}(x_0) &= f_{\mathcal{D}_2}(0) = 0.3\\
f_{\mathcal{D}_3}(x_0) &= f_{\mathcal{D}_3}(0) = 0.1
\end{align*}
We estimate the mean model prediction as
$$\mathbb{E}_\mathcal{D}[f_{\mathcal{D}}(x_0)] \approx \frac{1}{3}(f_{\mathcal{D}_1}(x_0) + f_{\mathcal{D}_2}(x_0) + f_{\mathcal{D}_3}(x_0)) = 0.2.$$
From this we compute the bias and variance estimates:
\begin{align*}
\text{Bias}^2 &= (f^*(x_0) - \mathbb{E}_\mathcal{D}[f_{\mathcal{D}}(x_0)])^2\\
&\approx (0 - 0.2)^2 = (0.2)^2 = 0.04\\
\text{Variance} &= \mathbb{E}_\mathcal{D}[(\mathbb{E}_\mathcal{D}[f_{\mathcal{D}}(x_0)] - f_{\mathcal{D}}(x_0))^2]\\
&\approx \frac{1}{3}((0.2 - 0.2)^2 + (0.2-0.3)^2 + (0.2-0.1)^2)\\
&= \frac{2}{3}(0.1)^2 = 0.0067
\end{align*}
````

## Model Complexity: RSS, Bias, and Cross-Validation
Below you see the plots of three regression models fit to the same training data (blue dots):

```{figure} images/regression/exercises_model_fit_comparison.png
---
width: 450px
name: fig-regression-model-comparison
align: center
---
Three regression models (A, B, C) fit to the same training data.
```

Match the statements to the corresponding model (A, B, or C).

- The highest RSS on the training data can be expected for ___.
- If I evaluate my model with ten-fold cross-validation, then I would expect that I get the lowest cross-validated test error with ___.
- The highest bias is expected for ___.

````{dropdown} Solution
**Model A** is a straight line that clearly does not follow the U-shaped trend of the data (it
undershoots the low-$x$ and high-$x$ points and overshoots the points in the middle). A model this
rigid cannot represent the true shape of the data no matter which training set it sees: that
inflexibility is exactly what **bias** measures, so Model A has the **highest bias**. Because it
systematically misses the training points by a wide margin, it also has the **highest RSS on the
training data** among the three models.

**Model C** is the opposite extreme: an extremely flexible curve that wiggles through (almost)
every training point, including points like $(7, 25)$ that look like noise. This gives Model C the
*lowest* training RSS and the *lowest* bias, not the highest, so Model C is never the correct match
for the first two statements. Fitting the noise this closely also means the fitted curve would
look very different if we resampled the training data (high **variance**), so under 10-fold
cross-validation this overfitting hurts generalization to the held-out folds — Model C would **not**
give the lowest cross-validated test error either. Model C is therefore the correct match for none
of the three statements; it is the distractor.

**Model B** follows the overall U-shaped trend without chasing every individual point, striking a
balance between the underfitting of Model A and the overfitting of Model C. This balance between
bias and variance is exactly what tends to generalize best to unseen folds, so Model B is expected
to give the **lowest cross-validated test error**.
````

## Computing RSS and MSE by Hand
The plot below shows a regression line (green) together with 3 train data points (blue circles)
and 3 test data points (orange squares). You can read their coordinates off the plot.

```{figure} images/regression/exercises_train_test_rss_mse.png
---
width: 450px
name: fig-regression-rss-mse
align: center
---
A linear model together with 3 train and 3 test data points.
```

Fill in the blanks with integers:

- The RSS of this regression model on the train data is ___
- The MSE of this regression model on the train data is ___
- The RSS of this regression model on the test data is ___
- The MSE of this regression model on the test data is ___

````{dropdown} Solution
From the plot, the linear model passes through $(0,1)$ and $(7,8)$, so it has slope $1$ and
intercept $1$:
$$f(x) = x+1.$$
The train points are $(1,3), (3,6), (5,5)$ and the test points are $(2,2), (4,7), (6,5)$.

**Train data.** The predictions and squared residuals $(y-f(x))^2$ are

| $x$ | $y$ | $f(x)$ | $(y-f(x))^2$ |
|---|---|---|---|
| 1 | 3 | 2 | $1^2=1$ |
| 3 | 6 | 4 | $2^2=4$ |
| 5 | 5 | 6 | $(-1)^2=1$ |

$$\text{RSS}_{\text{train}} = 1+4+1 = 6,\qquad \text{MSE}_{\text{train}} = \frac{6}{3} = 2.$$

**Test data.** Analogously,

| $x$ | $y$ | $f(x)$ | $(y-f(x))^2$ |
|---|---|---|---|
| 2 | 2 | 3 | $(-1)^2=1$ |
| 4 | 7 | 5 | $2^2=4$ |
| 6 | 5 | 7 | $(-2)^2=4$ |

$$\text{RSS}_{\text{test}} = 1+4+4 = 9,\qquad \text{MSE}_{\text{test}} = \frac{9}{3} = 3.$$

Note that the test RSS/MSE is higher than the train RSS/MSE here, which is the typical (though not
guaranteed) pattern: the model was fit to minimize error on the train data specifically, so it
tends to fit that data at least as well as unseen test data.
````

## Fitting an Affine Regression Function by Hand
Let's say we want to fit a polynomial of degree $k=1$
$$f(x) = \beta_1 x + \beta_0$$
to the following data points (represented in the form $(x,y)$):

| $x$ | $y$ |
|---|---|
| 1 | 8 |
| 2 | 20 |
| 4 | 2 |

The design matrix $X$ given these data points is
$$X = \begin{pmatrix}1 & 1\\1 & 2\\1 & 4\end{pmatrix}$$
for the feature transformation $\phi(x)$, such that
$$f(x) = \phi(x)^\top\begin{pmatrix}\beta_0\\\beta_1\end{pmatrix}.$$

In the resulting affine regression function $f(x)$, what is the value of $\beta_0$?

What is the value of $\beta_1$?

(Hint: it is a whole integer)

````{dropdown} Solution
From $X = \begin{pmatrix}1 & 1\\1 & 2\\1 & 4\end{pmatrix}$, the feature transformation is
$\phi(x) = \begin{pmatrix}1\\x\end{pmatrix}$, i.e. the design matrix's rows are $\phi(x_i)^\top$
for the three data points $x_1=1, x_2=2, x_3=4$. With $y = \begin{pmatrix}8\\20\\2\end{pmatrix}$,
we compute
$$X^\top X = \begin{pmatrix}3 & 7\\7 & 21\end{pmatrix}, \qquad X^\top y = \begin{pmatrix}30\\56\end{pmatrix},$$
since $\sum_i x_i = 1+2+4=7$, $\sum_i x_i^2 = 1+4+16=21$, $\sum_i y_i=8+20+2=30$, and
$\sum_i x_iy_i = 1\cdot 8+2\cdot 20+4\cdot 2=56$.

Solving $X^\top X\bm\beta = X^\top y$:
\begin{align*}
3\beta_0 + 7\beta_1 &= 30\\
7\beta_0 + 21\beta_1 &= 56
\end{align*}
From the first equation, $\beta_0 = (30-7\beta_1)/3$. Substituting into the second equation:
$$7\cdot\frac{30-7\beta_1}{3} + 21\beta_1 = 56 \;\Longrightarrow\; 210 - 49\beta_1 + 63\beta_1 = 168 \;\Longrightarrow\; 14\beta_1 = -42 \;\Longrightarrow\; \beta_1 = -3.$$
Then $\beta_0 = (30 - 7\cdot(-3))/3 = 51/3 = 17$.

So $\bm\beta = \begin{pmatrix}17\\-3\end{pmatrix}$, giving $\beta_0 = 17$ and $\beta_1 = -3$.
````


## Bias-Variance Trade-off: True or False
Select all correct statements on the bias-variance trade-off for regression.

Group of answer choices
- Irreducible errors can always be reduced by improving the features used in the model, while
  reducible errors are caused by fluctuations in the data and cannot be minimized.
- Expected prediction error can be decomposed into reducible errors and irreducible errors.
- Even if you have a perfect model, obtaining a low bias and variance, the expected prediction
  error (i.e. EPE) can still be high.
- The expected prediction error can be high, even if the chosen regression model returns in
  expectation (over all training data sets and test samples) the true regression function.

````{dropdown} Solution
**False: Irreducible errors can always be reduced by improving the features used in the model,
while reducible errors are caused by fluctuations in the data and cannot be minimized.** This
statement has the two error types swapped. The irreducible error is caused by inherent randomness
in the data-generating process (noise) and, true to its name, cannot be reduced no matter how good
the model or features are. The reducible error (bias$^2$ + variance) is the part that *can* be
minimized, by choosing a better model class, more informative features, or more training data.

**True: Expected prediction error can be decomposed into reducible errors and irreducible
errors.** This is exactly the standard decomposition
$$\text{EPE}(x_0) = \underbrace{\text{Bias}^2(\hat f(x_0)) + \text{Var}(\hat f(x_0))}_{\text{reducible error}} + \underbrace{\sigma^2}_{\text{irreducible error}}.$$

**True: Even if you have a perfect model, obtaining a low bias and variance, the expected
prediction error (i.e. EPE) can still be high.** Even if both terms of the reducible error are
driven close to zero, the irreducible error $\sigma^2$ in the decomposition above remains: it does
not depend on the model at all. If the data is inherently very noisy (large $\sigma^2$), the EPE
has a high floor that no model, however good, can get below.

**True: The expected prediction error can be high, even if the chosen regression model returns in
expectation (over all training data sets and test samples) the true regression function.** "Returns
in expectation the true regression function" means the model is unbiased, i.e.
$\mathbb{E}_\mathcal{D}[\hat f_\mathcal{D}(x_0)] = f^*(x_0)$, so $\text{Bias}^2=0$. But the EPE
still contains the variance and irreducible error terms: an unbiased model can still have high
variance (e.g. if it is very flexible, so its fit changes a lot from one training set to another),
or the data itself can be very noisy, either of which alone is enough to make the EPE high.
````

## Diagnosing Model Fits: Model 1 vs. Model 2
The graph below shows the actual targets and predictions of Model 1 and Model 2. Select all
correct statements.

```{figure} images/regression/exercises_model1_model2_fit.png
---
width: 600px
name: fig-regression-model1-model2
align: center
---
Model 1 (left) and Model 2 (right) fit to the same underlying data.
```

Group of answer choices
- Model 1 has a low variance and Model 2 has a high variance.
- A high noise level in the dataset will always lead to high bias.
- Model 1 has a high bias and Model 2 has a low bias.
- Increasing the complexity of Model 2 might improve its performance.
- Model 1 is overfitting, while Model 2 is underfitting.

````{dropdown} Solution
**False: Model 1 has a low variance and Model 2 has a high variance.** This has the two models
swapped. Model 1's prediction curve is extremely flexible and wiggles to pass through (almost)
every single training point, including the noisy fluctuations between them; refit on a different
noisy sample from the same process, and this wiggly curve would look substantially different, so
Model 1 has **high** variance. Model 2's prediction is close to a flat line, which barely changes
shape regardless of which particular noisy sample it is fit to, so Model 2 has **low** variance.

**False: A high noise level in the dataset will always lead to high bias.** The noise level in the
data corresponds to the irreducible error $\sigma^2$, which is a separate term in the bias-variance
decomposition from the bias itself. A model's bias is determined by how well its function class can
represent the true underlying signal, regardless of how noisy the observations of that signal are;
a simple, well-specified model can still have low bias on very noisy data (though the noise will
inflate the EPE regardless, and it can make variance harder to control on finite samples).

**False: Model 1 has a high bias and Model 2 has a low bias.** Again swapped: Model 1 fits the
training points almost exactly, so its average prediction is very close to the true signal
(**low** bias), at the cost of chasing noise (high variance). Model 2's near-flat curve
systematically misses the true U-shaped trend everywhere except near the middle, which is exactly
what **high** bias looks like.

**True: Increasing the complexity of Model 2 might improve its performance.** Model 2 is clearly
underfitting: its function class (essentially a flat line) is too rigid to capture the U-shaped
trend that is visibly present in the data. Allowing it more flexibility (e.g. fitting a higher
degree polynomial) would let it reduce its bias and better track the true signal, which should
improve performance, at least up to the point where it starts overfitting.

**True: Model 1 is overfitting, while Model 2 is underfitting.** Model 1 chases every fluctuation
in the training data, including noise, which is the hallmark of overfitting (low training error,
but poor generalization to new data since the noise it fit is not present in the same way in
unseen samples). Model 2 fails to capture even the overall trend that is clearly present in the
data, which is the hallmark of underfitting.
````
