#!/usr/bin/env python
# coding: utf-8

# ## Analytic Solutions
# You probably know from your highschool math classes that every local minimizer $x_0$ of a function $f:\mathbb{R}\rightarrow\mathbb{R}$ is a stationary point: $\frac{d}{dx}f(x_0)=0$. This is known as the first order neccessary condition. This property is easily understood, considering that the derivative indicates the slope of a function at a specified point. If we have a real-valued minimizer, then the slope is zero at that minimizer. If the slope would be positive, then we can go to the left to decrease the function value further, and if the slope is negative, we can go to the right. By means of this property, we can identify all the candidates that could be minimizers. Maximizers and saddle points are stationary points too. With the second order neccessary condition, we can filter further the minimizers from the pool of candidates. The second order neccessary condition states that at a minimizer the second derivative is nonnegative $\frac{d^2}{dx^2}f(x_0)\geq 0$. The second derivative is the slope of the slope. If we have a minimizer, then the slope increases: first we go down, and then we go up. Hence, we need that the slope of the slope does at least not decrease. 
# 
# ````{prf:example}
# Let's have a look at a seemingly simple example. The function $f(x) = \frac14x^4 + \frac13x^3 -x^2$ is plotted below and we see that there are two minimizers $x_1=-2$ and $x_2=1$. The question is just if those are all minimizers, or if there is another one beyond the scope of what is plotted here.
# ```{tikz}
# \begin{axis}[width=.8\textwidth,xlabel=$x$,ylabel=$y$,axis lines = center, 
# domain=-3:2,yticklabels={,,},xticklabels={,,}]
# \addplot[blue,thick]
# {x^4/2 + 2*x^3/3 - 2*x^2};
# \end{axis}
# ```
# To find all the minimizers of the function, we apply the first and second order neccessary condition. We compute the first and second derivative.
# \begin{align*}
#     \frac{d}{dx} f(x) &= x^3 + x^2 -2x \\
#     \frac{d^2}{dx^2}f(x) & = 3x^2 + 2x -2
# \end{align*}
# Now we solve the equation setting the first derivative to zero and get three stationary points:
# $$\frac{d}{dx} f(x) =0 \quad \Leftrightarrow \quad x_1=-2, x_2 = 0, x_3=1$$
# Given the plot, we already know which of these are minimizers, but to conclude our example, we apply the second order sufficient condition to identify the local minimizers $x_1=-2$ and $x_2=3$.
# 
# $$\frac{d^2}{dx^2}f(-2)=6\geq 0,\quad \frac{d^2}{dx^2}f(0)=-2< 0,\quad \frac{d^2}{dx^2}f(1)=3\geq 0 $$
# ````
# 
# ### Finding Stationary Points in higher dimensions
# The principles of the first and second order conditions can be generalized to functions $f:\mathbb{R}^d\rightarrow \mathbb{R}$ mapping from a $d$-dimensional vector space to real values. The main difficulty is that we have now more to consider than just left and right when looking for a direction into which we could minimize the function further. In fact, any vector $\vvec{v}\in\mathbb{R}^d$ could indicate a possible direction in which the function might decrease. Luckily, we can show that we just have to check one direction, given by the negative *gradient*, which points into the direction of steepest descent. The gradient indicates the slope of a function in the directions of the coordinates, which are called the *partial derivatives*. A partial derivative $\frac{\partial f(\vvec{x})}{\partial x_i}$ is computed like a one-dimensional derivative by treating all variables except for $x_i$ as  a constant. The gradient gathers those partial derivatives in a vector. The transposed of the gradient is called the *Jacobian*.
#     
# \begin{align*}
#     \frac{\partial f(\vvec{x})}{\partial \vvec{x}} &=
#     \begin{pmatrix}
#     \frac{\partial f(\vvec{x})}{\partial x_1} & \ldots & \frac{\partial f(\vvec{x})}{\partial x_d}
#     \end{pmatrix}\in\mathbb{R}^{1\times d} &\text{(Jacobian)}\\
#       \nabla_\vvec{x} f(\vvec{x}) &=
#     \begin{pmatrix}
#     \frac{\partial f(\vvec{x})}{\partial x_1} \\ \vdots \\ \frac{\partial f(\vvec{x})}{\partial x_d}
#     \end{pmatrix}\in\mathbb{R}^{d} &\text{(Gradient)}
# \end{align*}
# 
# With the gradient, we get a first order neccessary condition (FONC) for functions mapping from a vector space $\mathbb{R}^d$. 
# ````{prf:theorem} FONC   
# If $\vvec{x}$ is a local  minimizer of $f:\mathbb{R}^d\rightarrow\mathbb{R}$ and $f$ is continuously 
# differentiable in an open neighborhood of $\vvec{x}$, then
#     $$\nabla f(\vvec{x})=0$$
# ````    
# Likewise, a vector $\vvec{x}$ is called *stationary point* if $\nabla f(\vvec{x})=0$. The second order neccessary condition (SONC) uses the generation of the second order derivative to vector spaces, called the *Hessian*. We state this condition here for reasons of completeness, but we will not need this property for the machine learning models that we discuss in this course.
# 
# ````{prf:theorem} SONC
# If $\vvec{x}$ is a local  minimizer of $f:\mathbb{R}^d\rightarrow\mathbb{R}$ and $\nabla^2f$ is continuous in an open 
# neighborhood of $\vvec{x}$, then
# $$\nabla f(\vvec{x})=0 \text{ and } \nabla^2f(\vvec{x}) \text{ is positive semidefinite}$$
# ````   
# 
# A matrix $A\in\mathbb{R}^{d\times d}$ is **positive semidefinite** if
# $$\vvec{x}^\top A \vvec{x}\geq 0 \text{ for all } \vvec{x}\in\mathbb{R}^d$$
# ````{prf:example}
# :label: expl_fonc
# ```{figure} /images/optimization/rosenbrock.png
# ---
# height: 200px
# name: rosenbrock
# align: left
# ---
# The Rosenbrock function
# ```
# In this example we apply FONC and SONC to find the minimizers of the Rosenbrock function, which is given by
# \begin{align*}
#     f(\vvec{x})&= 100(x_2-x_1^2)^2 +(1-x_1)^2.
# \end{align*}
# In order to apply FONC, we need to compute the gradient. We do so by computing the partial derivatives. The partial derivatives are computed by the same rules as you know it from computing the derivative of a one-dimensional function.
# \begin{align*}
#     \frac{\partial}{\partial x_1}f(\vvec{x})&= 400x_1(x_1^2-x_2) +2(x_1-1)\\
#     \frac{\partial}{\partial x_2}f(\vvec{x})&= 200(x_2-x_1^2)
# \end{align*}
# FONC says that every minimizer has to be a stationary point. Stationary points are the vectors at which the gradient of $f$ is zero. We compute the set of stationary points by setting the gradient to zero and solving for $\vvec{x}$.
# \begin{align*}
#       \frac{\partial}{\partial x_2}f(\vvec{x})&=200(x_2-x_1^2)=0
#       &\Leftrightarrow x_2 =x_1^2\\
#       \frac{\partial}{\partial x_1}f\begin{pmatrix}x_1\\x_1^2\end{pmatrix}&= 2(x_1-1) =0 
#       &\Leftrightarrow x_1=1
# \end{align*}
# 
# According to FONC we have a stationary point at $\vvec{x}=(1,1)$. Now we check with SONC if the stationary point could be a minimizer (it could also be a maximizer or a saddle point). SONC says that every minimizer has a positive definite Hessian. Hence, we require the Hessian, the second derivative of the Rosenbrock function. To that end, we compute the partial derivatives of the partial derivatives: 
# \begin{align*}
# \frac{\partial^2}{\partial^2 x_1}f(\vvec{x})&= \frac{\partial}{\partial x_1}\left(\frac{\partial}{\partial x_1}f(\vvec{x})\right)= 1200x_1^2-400x_2 +2\\
# \frac{\partial^2}{\partial^2 x_2}f(\vvec{x})&=  \frac{\partial}{\partial x_2} \left(\frac{\partial}{\partial x_2}f(\vvec{x})\right)= 200\\
# \frac{\partial^2}{\partial x_1\partial x_2}f(\vvec{x})&=\frac{\partial^2}{\partial x_2\partial x_1}f(\vvec{x})= -400x_1
# \end{align*}
# The Hessian is given by
# \begin{align*}
#     \nabla^2 f(\vvec{x})&=  \begin{pmatrix}\frac{\partial^2}{\partial^2 x_1} f(\vvec{x}) & \frac{\partial^2}{\partial x_1x_2} f(\vvec{x})\\ \frac{\partial^2}{\partial x_2x_1}f(\vvec{x}) & \frac{\partial^2}{\partial^2 x_2} f(\vvec{x})\end{pmatrix}\\
#     &=200\begin{pmatrix} 6x_1^2-2x_2 + 0.01& -2x_1\\ -2x_1 &1 \end{pmatrix}
# \end{align*}
# We insert our stationary point $\vvec{x}_0=(1,1)$ into the Hessian and get
# $$\nabla^2f(\vvec{x}_0)= 200\begin{pmatrix} 4.01& -2\\ -2 & 1\end{pmatrix}$$
# Now we check if the Hessian at the stationary point is positive definite. Let $\vvec{x}\in\mathbb{R}^2$, then 
# \begin{align*}
#     \vvec{x}^\top \nabla^2f(\vvec{x}_0) \vvec{x} &= 200 \begin{pmatrix}x_1 & x_2\end{pmatrix} \begin{pmatrix}
#      4.01& -2\\ -2 & 1
#     \end{pmatrix}\begin{pmatrix}x_1\\x_2\end{pmatrix}\\
#     &= 200\begin{pmatrix}x_1 & x_2\end{pmatrix} \begin{pmatrix}
#     4.01x_1-2x_2\\ -2x_1+ x_2
#     \end{pmatrix}\\
#     &=200(4.01x_1^2 -2x_1x_2 -2x_1x_2 +x_2^2)\\
#     &= 200(4.01x_1^2 -4x_1x_2 + x_2^2)\\
#     &= 200((2x_1-x_2)^2 +0.01x_1^2) \geq 0
# \end{align*}
# The last inequality follows because the sum of quadratic terms can not be negative.
# We conclude that the Hessian at our stationary point is positive semi-definite. As a result, FONC and SONC yield that $\vvec{x}=(1,1)$ is the only possible local minimizer of $f$.
# ````
# Nice, we have now a strategy yo find local minimizers if we have an unconstrained objective with an objective function which is continuously differentiable. Let's consider a more complex setting, introducing constraints.
# ````{prf:example} Solving of a dual
# We solve the following constrained optimization problem:
# \begin{align*}
# \min_{\vvec{w}}\ & w_1^2+w_2^2\\
# \text{ s.t } & w_2\geq 1\\
# & w_2\geq -w_1+2
# \end{align*}
# Geometrically, the problem looks as follows:
# ```{tikz}
# \begin{tikzpicture}
# \begin{axis}[
#     axis equal,
#     width=10cm,
#     height=10cm,
#     xmin=-1, xmax=3,
#     ymin=-1, ymax=3,
#     xlabel=$w_1$,
#     ylabel=$w_2$,
#     axis lines=middle,
#     samples=100,
#     clip=false, 
#     view={0}{90},
# ]
# 
# % Level sets: manually plotted circles of constant x^2 + y^2
# \addplot [domain=0:360, samples=200, thick, blue!60] ({cos(x)}, {sin(x)});
# \addplot [domain=0:360, samples=200, thick, blue!60] ({sqrt(2)*cos(x)}, {sqrt(2)*sin(x)});
# \addplot [domain=0:360, samples=200, thick, blue!60] ({sqrt(0.5)*cos(x)}, {sqrt(0.5)*sin(x)});
# \addplot [domain=0:360, samples=200, thick, blue!60] ({sqrt(0.25)*cos(x)}, {sqrt(0.25)*sin(x)});
# \addplot [domain=0:360, samples=200, thick, blue!60] ({sqrt(0.125)*cos(x)}, {sqrt(0.125)*sin(x)});
# 
# % Feasible region
# \addplot [
#     name path=f,
#     domain=-1:3, 
# ]
# {max(1, -x+2)};
# 
# \path[name path=axis] (axis cs:-1,3) -- (axis cs:3,3)-- (axis cs:3,1);
# \addplot[blue!10,opacity=0.4] fill between[of=f and axis];
# 
# % Labels
# \node at (axis cs:2,2) {$\mathcal{C}$};
# 
# \end{axis}
# \end{tikzpicture}
# ```
# We have an objective function that is visualized over the level sets (the rings). Each ring indicates the vectors $\vvec{w}$ that return the same function value. We see that the minimum is at $(0,0)$, but that minimum does not lie in the feasible set $\mathcal{C}$.    
# 
# To solve this constrained objective, we formulate the dual, which requires the Lagrangian first. To formulate the Lagrangian, we put the constraints into the form $g(\vvec{w})\geq 0$:
# \begin{align*}
# w_2-1&\geq 0\\
# w_2+w_1-2 &\geq 0
# \end{align*}
# The Lagrangian is then given by
# $$\mathcal{L}(\vvec{w},\bm{\lambda})=w_1^2 +w_2^2 -\lambda_1(w_2-1)-\lambda_2(w_2+w_1-2)$$
# The dual objective function returns the minimum of the Lagrangian:
# $$\mathcal{L}_{dual}=\min_{\vvec{x}}\mathcal{L}(\vvec{w},\bm{\lambda}).$$
# We can compute the dual objective function analytically over the stationary point, since the Lagrangian is convex in $\vvec{w}$ (it is the sum of convex functions: the squared norm and affine functions). Hence, we compute the gradient and set it to zero.
# \begin{align*}
# \nabla_\vvec{w}\mathcal{L}(\vvec{w},\bm{\lambda}) &= \begin{pmatrix}
# 2w_1-\lambda_2\\
# 2w_2-\lambda_1-\lambda_2
# \end{pmatrix}
# =\begin{pmatrix}
# 0\\ 0
# \end{pmatrix}
# \Leftrightarrow & \begin{cases}
# w_1 = \frac12 \lambda_2\\
# w_2 = \frac12 \lambda_1 + \frac12 \lambda_2.
# \end{cases}
# \end{align*}
# We plug in the minimizer $\vvec{w}$ defined above in the Lagrangian and obtain the dual objective function:
# \begin{align*}
# \mathcal{L}_{dual}(\bm{\lambda}) &= -\frac14 \lambda_1^2-\frac12 \lambda_2^2 -\frac12\lambda_1\lambda_2 + \lambda_1 +2\lambda_2.
# \end{align*}
# Hence, we need to maximize the function above, which is equivalent to minimizing the negative dual function. The negative dual function is convex, since it can be written as
# \begin{align*}
# -\mathcal{L}_{dual}(\bm{\lambda}) &= \frac14\left\lVert \begin{pmatrix}1 & 1\\0 & 1\end{pmatrix}\bm{\lambda}\right\rVert^2 - \begin{pmatrix}1 &2 \end{pmatrix}\bm{\lambda}
# \end{align*} 
# which is the sum of a convex and an affine function. Hence, to solve the dual objective, we set again the gradient to zero:
# \begin{align*}
# -\nabla\mathcal{L}_{dual}(\bm{\lambda}) = \begin{pmatrix}\frac12\lambda_1 +\frac12\lambda_2-1\\ 
# \frac12\lambda_1+\lambda_2-2\end{pmatrix} = \vvec{0} 
# \end{align*}
# which is the case for $\lambda_1=0$ and $\lambda_2=2$. To get the solution nof our primal objective, we plug in the optimal $\bm{\lambda}$ in the optimal $\vvec{w}$ definition and get
# \begin{align*}
# w_1^* &= \frac12\lambda_2^* = 1\\
# w_2^* &= \frac12\lambda_1^* +\frac12\lambda_2^* = 1
# \end{align*}
# 
# 
# ````
