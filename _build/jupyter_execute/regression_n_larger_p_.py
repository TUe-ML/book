#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas
import scipy
import scipy.linalg as linalg
np.set_printoptions(precision=2,suppress=True)


# # Regression When $p>n$
# The global minimizers of the regression problem are given by
# $$\{\bm{\beta}\in\mathbb{R}^p\mid X^\top X\bm{\beta} =X^\top\vvec{y} \}.$$
# 
# If the matrix $X^\top X$ is invertible, then there is only one minimizer:
# $$\bm{\beta}= (X^\top X)^{-1}X^\top\vvec{y} $$
# 
# However, there also might be _infinitely many_ local and global minimizers of $RSS(\bm{\beta})$. 
# 
# ````{prf:example} Regression with $p>n$
# We consider a toy regression task, where the data is given by following three data points (observations) of one feature.
# 
#     D = np.array([5,3,1])
#     y = np.array([2,5,3])
#     pandas.DataFrame({"F1":D,"y":y})
# 
# That is, our given data looks as follows:
#  
#  $\mathtt{F}_1$ | $y$ 
# ----------------|----------------
# 5 | 2 
# 3 | 5 
# 1 | 3 
# 
# We fit a polynomial of degree $k=3$. For polynomial regression functions $f:\mathbb{R}\rightarrow\mathbb{R}$ we have
# $$f(x) = \beta_0 +\beta_1 x+ \beta_2x^2 + \beta_3x^3 = \phi(x)^\top\bm\beta,$$
# where 
# $$\bm\phi(x)^\top=\begin{pmatrix}1& x& x^2& x^3\end{pmatrix}.$$
# We implement $\phi$ and create the design matrix. Note that the following definition of $\phi$ for polynomials is only correct if the dimensionality of the feature space is equal to one, as it is in this example.
# 
#     def ϕ(x):
#         return np.row_stack((np.ones(x.shape[0]),x, x**2, x**3))
#     X=ϕ(D).T
# The design matrix gathers the transposed feature vectors of the data matrix:
# $$X = \begin{pmatrix} \phi^\top(5)\\ \phi^\top(3)\\ \phi^\top(1)\end{pmatrix} = \begin{pmatrix} 1 & 5 & 25 & 125\\ 1 & 3 & 9 & 27\\ 1 & 1 & 1 & 1\end{pmatrix}$$
# We have $n=3$ observations of $p=4$ features in the design matrix, and hence, $p>n$.
# 
# In this case ($p>n$), the matrix $X^\top X$ is not invertible. When you try to compute the inverse of $X^\top X$ by hand, then you will get a contradiction. If we compute the inverse numerically with `np.linalg.inv(X.T@X)`, then you will see that the computed "inverse" has extremely large values (in the scope of $10^13$ to $10^14$). This corresponds to the fact that we are trying to divide by zero (in a matrix way). Generally, you can assume that such extreme values (either being very close to zero like `1e-16` or very big like `1e16`) indicates values that are actually equal to infinity or equal to zero.
# 
# Anyways, we can easily check how good the computed inverse is, by multiplying the "inverse" with the matrix itself. The multiplication of the computed inverse and the matrix itself should return an approximate identity matrix. However, if we check `np.linalg.inv(X.T@X)@(X.T@X)`, then we see that this matrix is nowhere near the identity matrix.
# 
# In turn, what mostly works is to use a numerical solver for the system of linear equations which returns the global minimizers $\beta$ of the regression objective
# $$\{\beta\in\mathbb{R}^p\mid X^\top X\beta = X^\top y\}.$$
# 
# With `β = linalg.solve(X.T@X,X.T@y)` yields a solution for $\beta$, but returns a warning that the result might not be accurate. We observe that this result is accurate and by checking if $X^\top X\beta = X^\top y$. You can do this  by inspecting `X.T@X@β, X.T@y`, which indeed returns the same vector.   
# ````

# The example above shows that we can compute a solution for the regression problem in the case $p>n$, but how do we do that in general? Is there maybe a way to determine all the solvers $\beta$?

# ## Characterizing the Set of Regression Solvers with SVD
# A $(n\times n)$ matrix $A=U\Sigma V^\top$ is invertible if all singular values are larger than zero. The inverse is given by
# $$ A^{-1} = V \Sigma^{-1} U^\top,\ \text{ where }$$
# \begin{align*}
#     \Sigma=\begin{pmatrix}\sigma_1 & 0 & \ldots & 0 \\
#     0 & \sigma_2 &\ldots & 0\\
#       &     & \ddots &\\
#     0 & 0   & \ldots & \sigma_n\end{pmatrix},\qquad 
#     \Sigma^{-1}=
#     \begin{pmatrix}\frac{1}{\sigma_1} & 0 & \ldots & 0 \\
#     0 & \frac{1}{\sigma_2} &\ldots & 0\\
#       &     & \ddots &\\
#     0 & 0   & \ldots & \frac{1}{\sigma_n}\end{pmatrix}
# \end{align*}

# The $(p\times p)$ matrix $X^\top X$ is not invertible if this matrix has $r<p$ nonzero singular values. The singular values of $X^\top X$ are specified by the SVD of $X=U\Sigma V^\top$, we have
# \begin{align*}
#     X^\top X = V\Sigma^\top \underbrace{U^\top U}_{=I} \Sigma V^\top = V\Sigma^\top \Sigma V^\top.
# \end{align*}
# The singular value decomposition is uniquely defined and the decomposition $V\Sigma^\top \Sigma V^\top$ satisfies the requirements for the singular value decomposition of $X^\top X$. Hence, the singular values of $X^\top X$ are given by the diagonal elements of the matrix $\Sigma^\top \Sigma$, which can be decomposed into an invertible part $\Sigma_r^2$ and a non-invertible part: 
# \begin{align}
# \Sigma^\top \Sigma= 
# \left(
# \begin{array}{c:r}
# \begin{matrix}
# \sigma^2_1 & \ldots & 0  \\
# \vdots  & \ddots  & \vdots \\
# 0 & \ldots   & \sigma^2_r 
# \end{matrix} & \vvec{0} \\ \hdashline
# \vvec{0} & \vvec{0}
# \end{array}
# \right) 
# = \left(
# \begin{array}{c:r}
# \Sigma_r^2 & \vvec{0} \\ \hdashline
# \vvec{0} & \vvec{0}
# \end{array}
# \right).
# \label{eq:STS}
# \end{align} Given the singular value decomposition of $X$ and $X^\top X$, we can try to solve Eq.~\eqref{eq:minimizers} for $\bm\beta$:
# \begin{align}
#    X^\top X\bm{\beta} &= X^\top \vvec{y} \quad 
#    \Leftrightarrow \quad V\Sigma^\top\Sigma V^\top \bm{\beta}= V\Sigma^\top U^\top\vvec{y}
#    \quad 
#    \Leftrightarrow \quad \Sigma^\top\Sigma V^\top \bm{\beta}= \Sigma^\top U^\top\vvec{y}, \label{eq:beta1}
# \end{align}
# where the last equality follows from multiplying with $V^\top$ from the left.

# ````{prf:observation} Characterization of Regression solvers by SVD
# The global minimizers $\bm{\beta}$ to the linear regression problem with design matrix $X$, having the SVD $X=U\Sigma V^\top$, are given by
# $$\{\bm{\beta}\in\mathbb{R}^p\mid \Sigma^\top\Sigma V^\top \bm{\beta}= \Sigma^\top U^\top\vvec{y} \}.$$
# ````
# From this characterization of regression solvers follows that $\Sigma^\top\Sigma$ does **not** have an inverse if only $r<p$ singular values are nonzero. This is definitely the case if $n<p$, since we have at most $\min\{n,p\}$ nonzero singular values of a $n\times p$ matrix. 

# In[2]:


D = np.array([5,3,1])
y = np.array([2,5,3])
def ϕ(x):
    return np.row_stack((np.ones(x.shape[0]),x, x**2, x**3))
X=ϕ(D).T
U,σs,Vt = linalg.svd(X, full_matrices=True)
print(U.shape, σs.shape, Vt.shape)
V=Vt.T


# When we look at the singular values, then we have $r=3<4=p$.

# In[3]:


σs


# Correspondingly, the matrix $\Sigma^\top\Sigma$ (computed below) has not an inverse.

# In[4]:


Σ = np.column_stack((np.diag(σs),np.zeros(3)))
Σ


# In[5]:


Σ.T@Σ


# How is that reflected in the system of linear equations that we have to solve?

# In[6]:


print(Σ.T@Σ@V.T,"β=",Σ.T@U@y)


# \begin{align*}
# \Sigma^\top \Sigma V^\top \beta &= \Sigma^\top U \vvec{y}\\ \\
# \begin{pmatrix}
#  157.18 &   724.22 &  3445.93 & 16714.05\\
#  -4.36  &  -8.61  & -14.36   &  3.38\\
#  0.58   & 0.25   & -0.32    & 0.05\\
#  0      & 0      & 0        & 0
# \end{pmatrix}
# \beta &= 
# \begin{pmatrix}
# 415.45 \\ -21.56 \\ 1.04 \\   0
# \end{pmatrix}
# \end{align*}

# We have an underdetermined system. There are only 3 equations to determine 4 parameters of $\beta$. If you are going to solve this system by hand, then you will see that one parameter is always left over (it can't be determined by the given equations). Setting this parameter to any number yields then one of the infinite solutions to the regression problem. For example, we can set $\beta_4=2$.  

# ## The Pseudo-Inverse

# If only $r<p$ singular values are nonzero, we employ the pseudoinverse $(\Sigma^\top\Sigma)^+$ defined by
# \begin{align*}
# \Sigma^\top \Sigma= 
# \left(
# \begin{array}{c:r}
# \begin{matrix}
# \sigma^2_1 & \ldots & 0  \\
# \vdots  & \ddots  & \vdots \\
# 0 & \ldots   & \sigma^2_r 
# \end{matrix} & \vvec{0} \\ \hdashline
# \vvec{0} & \vvec{0}
# \end{array}
# \right)
# ,\quad 
# (\Sigma^\top \Sigma)^+=
# \left(
# \begin{array}{c:r}
# \begin{matrix}
# \frac{1}{\sigma^2_1} & \ldots & 0  \\
# \vdots  & \ddots  & \vdots \\
# 0 & \ldots   & \frac{1}{\sigma^2_r} 
# \end{matrix} & \vvec{0} \\ \hdashline
# \vvec{0} & \vvec{0}
# \end{array}
# \right)
# \end{align*}
# 
# If we have $r<p$ nonzero singular values, then we have infinitely many global optimizers 
# $$\bm\beta = VA\Sigma^\top U^\top\vvec{y}$$
# where 
# \begin{align*}
# A = \left(
# \begin{array}{c:c}
# \begin{matrix}
# \frac{1}{\sigma^2_1} & \ldots & 0  \\
# \vdots  & \ddots  & \vdots \\
# 0 & \ldots   & \frac{1}{\sigma^2_r} 
# \end{matrix} & \vvec{0} \\ \hdashline
# A_{r+1, 1}\ldots &  A_{r+1,p}\\
# \vdots &  \vdots\\
# A_{p,\ 1}\ldots &  A_{p,p}
# \end{array}
# \right)\in\mathbb{R}^{p\times p}
# \end{align*}
# 
# 
# We define the **regression solution** derived by the pseudo inverse as
# $$\bm\beta_+ = V(\Sigma^\top \Sigma)^+\Sigma^\top U^\top\vvec{y}$$
# where 
# \begin{align*}
# (\Sigma^\top \Sigma)^+=
# \left(
# \begin{array}{c:r}
# \begin{matrix}
# \frac{1}{\sigma^2_1} & \ldots & 0  \\
# \vdots  & \ddots  & \vdots \\
# 0 & \ldots   & \frac{1}{\sigma^2_r} 
# \end{matrix} & \vvec{0} \\ \hdashline
# \vvec{0} & \vvec{0}
# \end{array}
# \right)\in\mathbb{R}^{p\times p}
# \end{align*}

# We can now calculate a $\beta$ in the set of global minimizers. If ```random=True``` then a random matrix replaces the zero rows in the pseudo inverse of $\Sigma^\top\Sigma$.

# In[7]:


def get_beta(U,Σ,V,random =True):
    p=V.shape[1]
    ΣtΣ_p = Σ.T@Σ
    ΣtΣ_p[ΣtΣ_p>0] = 1/ΣtΣ_p[ΣtΣ_p>0]
    A=ΣtΣ_p
    if random:
        r=(Σ>0).sum() # the number of nonzero singular values
        A[r:p,:]=np.random.rand(p-r,p)
    return V@A@Σ.T@U.T@y


# Plot it! The function resulting from setting ```random=False``` is the one in blue.

# In[8]:


plt.figure(figsize=(10, 7))
x = np.linspace(0, 6, 100)
β = get_beta(U,Σ,V,random = False)
f_x = ϕ(x).T@β
plt.plot(x, f_x, label="f_0")
for i in range(1,5):
    β = get_beta(U,Σ,V)
    f_x = ϕ(x).T@β
    plt.plot(x, f_x, label="f_"+str(i))
plt.scatter(D, y, edgecolor='b', s=50)
plt.xlabel("x")
plt.ylabel("y")
plt.ylim((-5, 15))
plt.legend(loc="best")
plt.show()


# In[ ]:




