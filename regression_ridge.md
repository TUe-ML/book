# Ridge Regression
Ridge Regression describes an $L_2$-penalized regression task.

`````{admonition} Task (Ridge Regression)
:class: tip
:name: ridge_task
**Given** a dataset of $n$ observations
\begin{equation*}\mathcal{D}=\left\{(\vvec{x}_i,y_i)\vert \vvec{x}_i\in\mathbb{R}^{d}, y_i\in\mathbb{R}, 1\leq i \leq n\right\},\end{equation*}  
the design matrix $X\in\mathbb{R}^{n\times p}$, where $X_{i\cdot}=\bm\phi(\vvec{x}_i)^\top$ and a regularization weight $\lambda>0$.     
**Find** the regression vector $\bm\beta$, solving the following objective
\begin{align}
    \min_{\bm\beta\in\mathbb{R}^p} RSS_{L_2}(\bm{\beta})&= \lVert \vvec{y}-X\bm\beta\rVert^2 +\lambda \lVert\bm{\beta}\rVert^2. 
\end{align}      
**Return** the predictor function $f:\mathbb{R}^d\rightarrow\mathbb{R}$, $f(\vvec{x})=\bm\phi(\vvec{x})^\top\bm\beta$ 
`````
## Optimization
First of all, we observe that the ridge regression objective is convex. The objective function $RSS_{L_2}$ is convex as it is the nonnegatively weighted sum of convex functions. The feasible set is the set $\mathbb{R}^p$, which is convex as well. Hence, the whole objective is a convex, unconstrained optimization problem. That means that every stationary point of the objective function is a minimizer. We can compute the stationary points as follows:
:::{math}
:label: eq:ridge
\begin{align*}
    &\nabla_{\bm{\beta}} RSS_{L_2}(\bm{\beta})= -2X^\top(\vvec{y}-X\bm{\beta}) +2\lambda\bm{\beta} =0 \\
    \Leftrightarrow\quad & (X^\top X+\lambda I){\bm{\beta}} = X^\top \vvec{y}
\end{align*}
:::
How does this change the standard set of regression solutions?
## Effect of the Regularization Weight
The following lemma indicates that the solutions of Ridge Regression computed above are always unique (exactly one regression vector $\beta$ solves the ridge regression objective).
`````{prf:lemma}
For any matrix $X\in\mathbb{R}^{n\times d}$, the matrix $X^\top X+\lambda I$ is invertible for all $\lambda>0$
`````
```{prf:proof}
Let $X=U\Sigma V^\top$ be the singular value decomposition of $X$, then
\begin{align}
    X^\top X+\lambda I 
    &= V(\Sigma^\top\Sigma +\lambda I )V^\top
\end{align}
The matrix $\Sigma^\top\Sigma +\lambda I$ is invertible, as it is a diagonal matrix where each value on the diagonal is at least as large as $\lambda>0$. Hence, the matrix $X^\top X+\lambda I$ is invertible and the inverse is $V(\Sigma^\top\Sigma +\lambda I )^{-1}V^\top$.  
```
As a result, we can multiply with $(X^\top X+\lambda I)^{-1}$ from the left in Eq. {eq}`eq:ridge` and obtain the ridge regression solver.
```{prf:corollary}
The solution to the {ref}`Ridge Regression<ridge_task>` task, is given by
$$\bm\beta_{L_2} = (X^\top X+\lambda I)^{-1}X^\top \vvec{y}$$
```
So, for any regularization weight $\lambda>0$ we have a unique regression solver, but for $\lambda=0$ we can get infinitely many solutions. Yet, what kind of solution do we get for really small $\lambda$? Is there one solution of the infinitely many that is somewhat better than the others? We can answer this question indeed with "yes", as the following theorem shows.
```{prf:theorem}
Let $X=U\Sigma V^\top\in\mathbb{R}^{n\times p}$ be the SVD of the design matrix of the {ref}`Ridge Regression<ridge_task>` task. If only $r<p$ singular values of $X$ are nonzero ($X$ has a rank of $r$), then the global minimizer $\bm{\beta}_{L_2}$ converges for decreasing regularization weights $\lambda\rightarrow 0$ to 
\begin{align*}
    \bm\beta_{L_2} &= (X^\top X+\lambda I)^{-1}X^\top \vvec{y} 
    &\rightarrow V\begin{pmatrix}\Sigma_r^{-1} U_r^\top \vvec{y}\\ \mathbf{0}\end{pmatrix} 
\end{align*}
$\Sigma_r$ denotes here the matrix containing only the first $r$ rows and columns of the singular values matrix $\Sigma$ and $U_r$ denotes the matrix containing the first $r$ left singular vectors (the first $r$ columns of $U$).
```
````{toggle}
```{prf:proof}
We substitute $X$ with its SVD in Eq. {eq}`eq:ridge` yielding the ridge regression solutions.
\begin{align*}
(X^\top X+\lambda I){\bm{\beta}} &= X^\top \vvec{y}\\
\Leftrightarrow (V\Sigma^\top\Sigma V^\top+\lambda I){\bm{\beta}} &= V\Sigma^\top U^\top \vvec{y}\\
\Leftrightarrow V(\Sigma^\top\Sigma +\lambda I)V^\top{\bm{\beta}} &= V\Sigma^\top U^\top \vvec{y}\\
    \Leftrightarrow {\bm{\beta}}&= V(\Sigma^\top\Sigma +\lambda I )^{-1}\Sigma^\top U^\top \vvec{y}
\end{align*}
We use the notation of Observation {ref}`obs:sigma_r` to compute
\begin{align*}
    (\Sigma^\top\Sigma +\lambda I )^{-1}\Sigma^\top &= 
    \left(
    \begin{array}{c;{2pt/2pt}c}
    \begin{matrix}
    \frac{1}{\sigma^2_1+\lambda} & \ldots & 0  \\
    \vdots  & \ddots  & \vdots \\
    0 & \ldots   & \frac{1}{\sigma^2_r+\lambda} 
    \end{matrix} & \vvec{0} \\
    \vvec{0} &
    \begin{matrix}
     \frac1\lambda &&\\
     &  \ddots & \\
     &  & \frac1\lambda
    \end{matrix}
    \end{array}
    \right)
    \left(
    \begin{array}{c;{2pt/2pt}c}
    \begin{matrix}
        \sigma_1 & \ldots & 0  \\
        \vdots  & \ddots  & \vdots \\
        0 & \ldots   & \sigma_r\\
        \\
        & \mathbf{0} &\\
        \\
    \end{matrix}
    & \mathbf{0}
    \end{array}
    \right)\\
    &= \left(
    \begin{array}{c;{2pt/2pt}c}
    \begin{matrix}
    \frac{\sigma_1}{\sigma^2_1+\lambda} & \ldots & 0  \\
    \vdots  & \ddots  & \vdots \\
    0 & \ldots   & \frac{\sigma_r}{\sigma^2_r+\lambda} 
    \end{matrix} & \vvec{0} \\
    \vvec{0} & \vvec{0}
    \end{array}
    \right).
\end{align*}
Hence, the lower $p-r$ rows of $(\Sigma^\top\Sigma +\lambda I )^{-1}\Sigma^\top U^\top \vvec{y}$ are equal to zero, returning
$$\bm\beta=V \begin{pmatrix}\diag(\frac{\sigma_1}{sigma_1^2 + \lambda},\ldots, \frac{\sigma_r}{sigma_r^2 + \lambda}) U_r^\top \vvec{y}\\ \mathbf{0}\end{pmatrix}. $$
For $\lambda\rightarrow 0$, the diagonal matrix above converges to $\Sigma_r^{-1}$.
```
````
Let's look again at the plot of {prf:ref}`example_reg_p_larger_n`. The regression function that we learn has four parameters, but we have only three data points. Hence, we have $p-r=1$ and our regression solution vectors are computed as $\beta=V\mathbf{w}$ where $\vvec{w}=\begin{pmatrix}\Sigma_r^{-1} U_r^\top \vvec{y}\\ w_4\end{pmatrix}$ and $w_4\in\mathbb{R}$. If $w_4=0$, then the resulting $\beta$ is the one that ridge regression converges to when $\lambda\rightarrow 0$. We plot now the resulting regression functions, depending on the value of $w_4$ and get the following:  