#!/usr/bin/env python
# coding: utf-8

# ## Matrix Derivatives
# In principle, we can compute matrix derivatives using partial derivatives and familiar rules like the chain rule. However, this process can be tedious and complex. Fortunately, most of the rules you know for one-dimensional derivatives also apply to higher-order derivatives. However, unlike in one-dimensional calculus, the order of multiplication matters for matrix derivatives because matrix multiplication is generally not commutative.
# 
# We’ll start by examining the dimensionalities involved in matrix derivatives. For a function $f:\mathbb{R}^{n\times d}\rightarrow \mathbb{R}$ that maps matrices to real values, we can define its derivative in two ways: as the gradient or the Jacobian.
# \begin{align*}
#     \frac{\partial f(X)}{\partial X} &=
#     \begin{pmatrix}
#     \frac{\partial f(X)}{\partial X_{11}} & \ldots & \frac{\partial f(X)}{\partial X_{n1}}\\
#     \vdots & \ddots & \vdots\\ 
#     \frac{\partial f(X)}{\partial X_{1d}} & \ldots & \frac{\partial f(X)}{\partial X_{nd}}
#     \end{pmatrix}\in\mathbb{R}^{d\times n} &\text{(Jacobian)}\\
#     \nabla{f(X)} &=
#     \begin{pmatrix}
#     \frac{\partial f(X)}{\partial X_{11}} & \ldots & \frac{\partial f(X)}{\partial X_{1d}}\\
#     \vdots & \ddots & \vdots\\ 
#     \frac{\partial f(X)}{\partial X_{n1}} & \ldots & \frac{\partial f(X)}{\partial X_{nd}}
#     \end{pmatrix}\in\mathbb{R}^{n\times d} &\text{(Gradient)}
# \end{align*}
# You might notice that the Jacobian is the transposed of the gradient, and vice versa.
# $$\nabla_\vvec{x} \vvec{f}(\vvec{x}) = \left(\frac{\partial \vvec{f}(\vvec{x})}{\partial\vvec{x}}\right)^\top$$
# Be aware that the notation above is not used by all authors. Some authors define the Jacobian as we define the gradient here. To remember what is what, consider the gradient descent update rule, where we subtract the scaled gradient from the iterates. For functions mapping to real values (such as loss or objective functions), the gradient must have the same dimensionality as the function's arguments; otherwise, subtraction would not be possible.      
# 
# From the definition of matrix derivatives, we can also infer the definition of a vector derivative for a function $f:\mathbb{R}^d\rightarrow \mathbb{R}$:
# \begin{align*}
#     \frac{\partial f(\vvec{x})}{\partial \vvec{x}} &=
#     \begin{pmatrix}
#     \frac{\partial f(\vvec{x})}{\partial x_1} & \ldots & \frac{\partial f(\vvec{x})}{\partial x_d}
#     \end{pmatrix}\in\mathbb{R}^{1\times d} & \text{(Jacobian)}\\
#     \nabla_\vvec{x} f(\vvec{x}) &=
#     \begin{pmatrix}
#     \frac{\partial f(\vvec{x})}{\partial x_1} \\ \vdots \\ \frac{\partial f(\vvec{x})}{\partial x_d}
#     \end{pmatrix}\in\mathbb{R}^{d} &\text{(Gradient)}
# \end{align*}
# If we have a function that maps to a vector space, then we can compute the partial derivatives for each coordimate of the function value. For example, if we have a function mapping from real values to the $c$-dimensional real-valued vector space $$\vvec{f}:\mathbb{R}\rightarrow \mathbb{R}^{c},\ \vvec{f}(x)=\begin{pmatrix}f_1(x)\\\vdots\\f_c(x)\end{pmatrix},$$ then the Jacobian and gradient are defined as 
# \begin{align*}
#     \frac{\partial \vvec{f}(x)}{\partial x} &=
#     \begin{pmatrix}
#     \frac{\partial f_1(x)}{\partial x} \\ \vdots \\ \frac{\partial f_c(x)}{\partial x}
#     \end{pmatrix} \in\mathbb{R}^{c} & \text{(Jacobian)}\\
#     \nabla_x \vvec{f}(x) &=
#     \begin{pmatrix}
#     \frac{\partial f_1(x)}{\partial x} & \ldots & \frac{\partial f_c(x)}{\partial x}
#     \end{pmatrix} \in\mathbb{R}^{1\times c} &\text{(Gradient)}
# \end{align*}
# Note that the Jacobian preserves now the dimensionality of the output of the function: function vales are in $\mathbb{R}^c$ and so is the Jacobian. Likewise, we can define the derivatives for a function $\vvec{f}:\mathbb{R}^d\rightarrow \mathbb{R}^{c}$ from a vector space to a vector space:
# \begin{align*}
#     \frac{\partial \vvec{f}(\vvec{x})}{\partial \vvec{x}} &=
#     \begin{pmatrix}
#     \frac{\partial f_1(x)}{\partial x_1}& \ldots &  \frac{\partial f_1(x)}{\partial x_d}\\ 
#     \vdots & \ddots & \vdots \\ 
#     \frac{\partial f_c(x)}{\partial x_1} &\ldots & \frac{\partial f_c(x)}{\partial x_d}
#     \end{pmatrix} \in\mathbb{R}^{c\times d} & \text{(Jacobian)}\\
#     \nabla_\vvec{x}\vvec{f}(\vvec{x}) &=
#     \begin{pmatrix}
#     \frac{\partial f_1(x)}{\partial x_1}& \ldots &  \frac{\partial f_c(x)}{\partial x_1}\\ 
#     \vdots & \ddots & \vdots \\ 
#     \frac{\partial f_1(x)}{\partial x_d} &\ldots & \frac{\partial f_c(x)}{\partial x_d}
#     \end{pmatrix} \in\mathbb{R}^{d\times c} &\text{(Gradient)}
# \end{align*}
# Of course we could now consider more cases, like a function mapping a matrix to a matrix. Unfortunately, from this point on, it gets really complicated. There are multiple ways to define such derivatives - as tensors or as specifically structured matrices. We're going to keep it comparatively simple and circumvent these cases in this course.   
# We can now concatenate these derivatives according to linearity and the chain rule for matrix derivatives.
# ```{prf:theorem} The Jacobian is linear
# For any function whose Jacobian is defined as a matrix of partial derivatives $\frac{\partial \vvec{f}(\vvec{x})}{\partial \vvec{x}} = \begin{pmatrix}\frac{\partial f_j(\vvec{x})}{\partial x_i}\end{pmatrix}_{i,j}$ for some indexes $i,j$, the Jacobian is linear:
# $$\frac{\partial\alpha\vvec{f}(\vvec{x})+\vvec{g}(\vvec{x})}{\partial\vvec{x}}
#         =\alpha\frac{\partial\vvec{f}(\vvec{x})}{\partial\vvec{x}}+\frac{\partial\vvec{g}(\vvec{x})}{\partial\vvec{x}}$$
# ```
# ```{prf:proof}
# The proof follows from the linearity of the partial derivatives: 
# \begin{align*}
# \frac{\partial \alpha\vvec{f}(\vvec{x})+\vvec{g}(\vvec{x})}{\partial \vvec{x}} &= \begin{pmatrix}\frac{\partial \alpha f_j(\vvec{x}) +g_j(\vvec{x})}{\partial x_i}\end{pmatrix}_{i,j}\\
# &= \begin{pmatrix}\alpha\frac{\partial  f_j(\vvec{x})}{\partial x_i}+\frac{\partial g_j(\vvec{x})}{\partial x_i}\end{pmatrix}_{i,j}\\
# &= \alpha\begin{pmatrix}\frac{\partial  f_j(\vvec{x})}{\partial x_i}\end{pmatrix}_{i,j} + \begin{pmatrix}\frac{\partial g_j(\vvec(x))}{\partial x_i}\end{pmatrix}_{i,j}\\
# &=\alpha\frac{\partial\vvec{f}(\vvec{x})}{\partial\vvec{x}}+\frac{\partial\vvec{g}(\vvec{x})}{\partial\vvec{x}}
# \end{align*}
# ```
# ```{prf:theorem} Chain Rule for the Jacobian
# For any continuously differentiable functions $\vvec{f}:\mathbb{R}^c\rightarrow \mathbb{R}^p$ and $\vvec{g}:\mathbb{R}^d\rightarrow \mathbb{R}^c$, the Jacobian of the composition $\vvec{f}\circ\vvec{g}$ is given by the chain rule:
# \begin{align*}
#         \frac{\partial \vvec{f}(\vvec{g}(\vvec{x}))}{\partial\vvec{x}}
#         &= \frac{\partial \vvec{f}(\vvec{g}(\vvec{x}))}{\partial \vvec{g}(\vvec{x})} \frac{\partial \vvec{g}(\vvec{x})}{\partial \vvec{x}} 
#     \end{align*}
# ```
# 
