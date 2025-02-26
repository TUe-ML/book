#!/usr/bin/env python
# coding: utf-8

# ## Numerical Optimization
# We have seen some strategies to find a solution to optimization objectives by means of solving a system of equations. These strategies work usually fine for simple objectives, but if I have a lot of valleys and hills in my optimization objective, then solving this system of equations analytically will not always be possible. What can we do then?       
# 
# If the minimizers can not be computed directly/analytically, then numerical optimization can come to the rescue. The idea is that I start somewhere in my hilly landscape and then try to walk to a valley with a specified strategy. For those types of methods it's good to know how good the strategy is. Important is for example to ask whether I will ever arrive at my minimum if I just walk long enough, or if I have to wonder endlessly around in a bad case. And what happens if I just walk a few steps, would I then have improved upon my starting position or might I not have descended at all? We state here two very popular numerical optimization methods: coordinate descent and gradient descent. Both are presented for the optimization of an unconstrained objective, but they do have extensions to incorporate constraints as well. This is beyond of the scope of this course, though.  
# The general scheme of numerical optimization methods is typically 
# 
# ```{prf:algorithm} Numerical Optimization
# 
# **Input**: the function $f$ to minimize, the maximum number of steps $t_{max}$ 
# 1. $\vvec{x}_0\gets$ `Initialize`($\vvec{x}_0$)  
# 2. **for** $t\in\{1,\ldots,t_{max}-1\}$
#     1. $\vvec{x}_{t+1}\gets $`Update`($\vvec{x}_t,f$)
# 3. **return** $\vvec{x}_{t_{max}}$
# ```
# 
# ### Coordinate descent
# The coordinate descent method is promising if we can not determine the minimum to the function analytically, but the minimum in a coordinate direction. The update is performed by cycling over all coordinates and walking to the minimum subject to that coordinate in each step.
# 
# ```{prf:algorithm} Coordinate Descent
# **Input**: the function $f$ to minimize 
# 1. $\vvec{x}_0\gets$ `Initialize`($\vvec{x}_0$)  
# 2. **for** $t\in\{1,\ldots,t_{max}-1\}$
#     1. **for** $i\in\{1,\ldots d\}$ do
#         1. $\displaystyle {x_i}^{(t+1)}\leftarrow \argmin_{x_i} f({x_1}^{(t+1)},\ldots ,{x_{i-1}}^{(t+1)}, x_i,{x_{i+1}}^{(t)},\ldots ,{x_d}^{(t)})$
# 3. **return** $\vvec{x}_{t_{max}}$
# ```
# The figure below shows the level set of a function, where every ring indicates the points in the domain of the function that have a specified function value $\{\vvec{x}\mid f(\vvec{x})=c\}$. The plotted function has a minimum at the center of the ellipses. We start at $\vvec{x}_0$ and then move to the minimum in direction of the vertical coordinate. That is, we move to the smallest ellipse (smallest diameter) we can touch in direction of the vertical coordinate. Then we move to the minimum in direction of the horizontal coordinate and we are already at our minimum where the method stops.
# ```{tikz}
# \begin{tikzpicture}[samples=200,smooth]
#         \begin{scope}
#             \clip(-4,-1) rectangle (4,4);
#             \draw plot[domain=0:360] ({cos(\x)*sqrt(20/(sin(2*\x)+2))},{sin(\x)*sqrt(20/(sin(2*\x)+2))});
#             \draw plot[domain=0:360] ({cos(\x)*sqrt(16/(sin(2*\x)+2))},{sin(\x)*sqrt(16/(sin(2*\x)+2))});
#             \draw plot[domain=0:360] ({cos(\x)*sqrt(12/(sin(2*\x)+2))},{sin(\x)*sqrt(12/(sin(2*\x)+2))});
#             \draw plot[domain=0:360] ({cos(\x)*sqrt(8/(sin(2*\x)+2))},{sin(\x)*sqrt(8/(sin(2*\x)+2))});
#             \draw plot[domain=0:360] ({cos(\x)*sqrt(4/(sin(2*\x)+2))},{sin(\x)*sqrt(4/(sin(2*\x)+2))});
#             \draw plot[domain=0:360] ({cos(\x)*sqrt(1/(sin(2*\x)+2))},{sin(\x)*sqrt(1/(sin(2*\x)+2))});
#             \draw plot[domain=0:360] ({cos(\x)*sqrt(0.0625/(sin(2*\x)+2))},{sin(\x)*sqrt(0.0625/(sin(2*\x)+2))});
# 
#             \draw[->,blue,ultra thick] (-2,3.65) to (-2,0);
#             \draw[->,blue,ultra thick] (-2,0) to (0,0);
#             
#             \node at (-2.9,3.5){ $(x_1^{(0)},x_2^{(0)})$};
#             \node at (-2.9,0.4){ $(x_1^{(1)},x_2^{(0)})$};
#             \node at (0,0.5){$(x^{(1)}_1,x^{(1)}_1)$};
#         \end{scope}
#     \end{tikzpicture}
# ```
# Coordinate descent minimizes the function value in every step:
# $$ f(\vvec{x}^{(0)})\geq f(\vvec{x}^{(1)})\geq f(\vvec{x}^{(2)})\geq\ldots$$
# 
# ````{prf:example} Coordinate descent of the Rosenbrock function
# ```{figure} /images/optimization/coordinateDescentRosenbrock.png
# ---
# height: 200px
# name: coord_desc_rosen
# align: right
# ---
# Coordinate descent updates
# ``` 
# Let's try to apply coordinate descent to find the minimum of the Rosenbrock function. From {prf:ref}`expl_fonc` we know the partial derivatives of the function 
# \begin{align*}
#     f(\vvec{x})&= 100(x_2-x_1^2)^2 +(1-x_1)^2.\\
#     \frac{\partial}{\partial x_1}f(\vvec{x})&= 400x_1(x_1^2-x_2) +2(x_1-1)\\
#     \frac{\partial}{\partial x_2}f(\vvec{x})&= 200(x_2-x_1^2).
# \end{align*}
# We compute the minima of the function in direction of the coordinates by means of FONC. The derivatives subject to each coordinate are exactly given by the partial derivatives, hence we set them equal to zero:
# \begin{align*}
#       \frac{\partial}{\partial x_1}f\begin{pmatrix}x_1\\x_1^2\end{pmatrix}&= 2(x_1-1) =0 
#       &\Leftrightarrow x_1=1\\
#       \frac{\partial}{\partial x_2}f(\vvec{x})&=200(x_2-x_1^2)=0
#       &\Leftrightarrow x_2 =x_1^2
# \end{align*}
# We have here only one minimizer candidate for each coordinate, and since we know from {prf:ref}`expl_fonc` that the function has only one minimizer (and no other maximizer or such), we know that the coordinate-wise minimizer candidates actually minimize the function in each coordinate. From this, we derive our update rules: 
# \begin{align*}
#     \argmin_{x_1\in\mathbb{R}} f(x_1,x_2) =1 \\
#     \argmin_{x_2\in\mathbb{R}} f(x_1,x_2) =x_1^2
# \end{align*}
# {numref}`coord_desc_rosen` shows the result of these update rules when starting at $(-2,2)$. We see that the minimum is quickly reached after one cycle of updating each coordinate. 
# ````

# ### Gradient Descent
# If we can't solve the system of equations given by FONC, also no coordinate-wise, but the function is differentiable, then we can apply gradient descent. Gradient descent is a strategy according to which you take a step in the direction which goes down the most steeply from where you stand.
# 
# ```{prf:algorithm} Gradient Descent
# **Input**: the function $f$ to minimize, step-size $\eta$ 
# 1. $\vvec{x}_0\gets$ `Initialize`($\vvec{x}_0$)  
# 2. **for** $t\in\{1,\ldots,t_{max}-1\}$
#     1. $\vvec{x}_{t+1}\leftarrow \vvec{x}_t - \eta \nabla f(\vvec{x}_t)$
# 3. **return** $\vvec{x}_{t_{max}}$
# ```
# 
# The parameter $\eta$ doesn't have to be a constant, it might also be a function that returns the *step size* depending on the amount of steps that have already performed. Setting the step size well is often a difficult task. The figure below shows how gradient descent makes the updates based on the local information.
# 
# ```{tikz}
# \begin{tikzpicture}[samples=200,smooth]
# \begin{scope}
#     \clip(-4,-1) rectangle (4,4);
#     \draw plot[domain=0:360] ({cos(\x)*sqrt(20/(sin(2*\x)+2))},{sin(\x)*sqrt(20/(sin(2*\x)+2))});
#     \draw plot[domain=0:360] ({cos(\x)*sqrt(16/(sin(2*\x)+2))},{sin(\x)*sqrt(16/(sin(2*\x)+2))});
#     \draw plot[domain=0:360] ({cos(\x)*sqrt(12/(sin(2*\x)+2))},{sin(\x)*sqrt(12/(sin(2*\x)+2))});
#     \draw plot[domain=0:360] ({cos(\x)*sqrt(8/(sin(2*\x)+2))},{sin(\x)*sqrt(8/(sin(2*\x)+2))});
#     \draw plot[domain=0:360] ({cos(\x)*sqrt(4/(sin(2*\x)+2))},{sin(\x)*sqrt(4/(sin(2*\x)+2))});
#     \draw plot[domain=0:360] ({cos(\x)*sqrt(1/(sin(2*\x)+2))},{sin(\x)*sqrt(1/(sin(2*\x)+2))});
#     \draw plot[domain=0:360] ({cos(\x)*sqrt(0.0625/(sin(2*\x)+2))},{sin(\x)*sqrt(0.0625/(sin(2*\x)+2))});
# 
#     \draw[->,blue,ultra thick] (-2,3.65) to (-1.93,3);
#     \draw[->,blue,ultra thick] (-1.93,3) to (-1.75,2.4);
#     \draw[->,blue,ultra thick] (-1.75,2.4) to (-1.5,1.8);
#     \draw[->,blue,ultra thick] (-1.5,1.8) to (-1.15,1.3);      \node at (-1.4,3.8){ $\mathbf{x}_0$};
#     \node at (-1.2,3.2){$\mathbf{x}_1$};
#     \node at (-1.05,2.6){ $\mathbf{x}_2$};
#     \node at (-0.8,2){ $\mathbf{x}_3$};
#     \node at (-0.6,1.4){ $\mathbf{x}_4$};
# \end{scope}
# \end{tikzpicture}
# ```
# 
# The negative gradient points into the direction of steepest descent. Hence, for a small enough step size we will go down each step:
# $$f(\vvec{x}_0)\geq f(\vvec{x}_1)\geq f(\vvec{x}_2)\geq\ldots$$
# However, decreasing the function value in every step is in practice not neccessarily desirable. In particular in the beginning of the optimization, it's useful to take larger steps to survey the landscape before converging to a local minimum.
# ````{prf:example} Gradient Descent on the Rosenbrock Function
# We illustrate the effect of the step-size by means of the Rosenbrock function. {numref}`grad_desc_smallstep` shows the trajectory when using a small step-size, {numref}`grad_desc_goodstep` shows a moderate step size and {numref}`grad_desc_bigstep` shows a larger step-size. We observe that all three step-sizes result in a sequence that converges to the minimim $(1,1)$. However, in particular the very small step-size results in a trajectory that requires many many steps. The fastest convergence has the larger step-size, but even this one needs approximately 800 iterations. In comparison to coordinate descent, that just required to update each coordinate once, gradient descent is much more inefficient. We also observe that the trajectory for the larger step-size is already zig-zagging, which often indicates that the step-size is actually too large, since it overshoots in each step. Hence, this function is not very efficiently optimizable with gradient descent. But feel free to try it yourself. 
# ```{figure} /images/optimization/gradientDescentRosenbrockSmall.png
# ---
# height: 220px
# name: grad_desc_smallstep
# ---
# Gradient Descent with $\eta=0.0005$ on the Rosenbrock Function
# ``` 
# ```{figure} /images/optimization/gradientDescentRosenbrock.png
# ---
# height: 220px
# name: grad_desc_goodstep
# ---
# Gradient Descent with $\eta=0.00125$ on the Rosenbrock Function
# ``` 
# ```{figure} /images/optimization/gradientDescentRosenbrockLarge.png
# ---
# height: 220px
# name: grad_desc_bigstep
# ---
# Gradient Descent with $\eta=0.0016$ on the Rosenbrock Function
# ``` 
# 
# 
# ````
