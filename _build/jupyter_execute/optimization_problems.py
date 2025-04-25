#!/usr/bin/env python
# coding: utf-8

# # Optimization Problems
# 
# ````{figure} /images/optimization/mountain.jpg
# ---
# height: 200px
# name: mountain
# align: left
# ---
# ````     
# Imagine yourself standing in the middle of a vast, rugged mountain landscape. Your goal is to reach the lowest point in this terrain—a hidden valley where the air is still. Easy, you might think. I'll just look at the map and see where the lowest point is. But what if the map is infinitely big? And what if the mountain landscape is in a high-dimensional hyperspace, where you can't see anything but three dimensional projections of this hyperspace? The optimization challenges in machine learning are a bit like that: the task to find the lowest valley an infinitely large space extending within hundreds of dimensions. What we need to solve this task is to get some hints where to look for the valley, and this is what optimization theory can deliver.
# 
# We start with looking at vanilla optimization problems that consist only of an objective function that is to be minimized, having no additional constraints on the solution.
# ## Unconstrained Optimization Problems
# We define an optimization problem, also called *the objective* as follows. 
# ````{prf:definition} unconstrained optimization objective
# :label: objective
# Given an **objective function** $f:\mathbb{R}^n\rightarrow \mathbb{R}$, the **objective of an unconstrained optimization problem** is:
#     $$\begin{align*}
#         \min_{x\in\mathbb{R}^n} f(\vvec{x})
#     \end{align*}$$
# We say that:
# -  $\displaystyle \vvec{x}^*\in\argmin_{x\in\mathbb{R}^n}f(\vvec{x})$ is a **minimizer**
# - $\displaystyle \min_{\vvec{x}\in\mathbb{R}^n}f(\vvec{x})$ is the **minimum** 
# ````  
#    
# A minimizer can be *local* or *global*. Following our analogy, the global minimizer is the lowest valley and a local minimizer can be any valley. 
# ````{prf:definition} minimizers
# :label: minimizers
# Given an unconstrained objective as defined above. A **global minimizer** is a vector $\vvec{x}^*$ satisfying 
# $$
#   f(\vvec{x}^*)\leq f(\vvec{x}) \text{ for all } \vvec{x}\in\mathbb{R}^n
# $$
# A **local minimizer** is a vector $\vvec{x}_0$ satisfying
# $$f(\vvec{x}_0)\leq f(\vvec{x}) \text{ for all } \vvec{x}\in\mathcal{N}_\epsilon(\vvec{x}_0),$$
# where $\mathcal{N}_\epsilon(\vvec{x}_0) = \{\vvec{x}\in\mathbb{R}^n\vert \lVert x-x_0\rVert\leq \epsilon\}$
# ```` 
# In an unconstrained optimization problem, the minimizers are the points where the function does not decrease in any direction. These points are a subset of the *stationary points*, which can be identified analytically or by numerical optimization methods (to be discussed later). 
# 
# 

# ## Constrained Optimization Problems
# ````{prf:definition} Constrained Objective
# :label: constr_objective
# Given an objective function $f:\mathbb{R}^d\rightarrow \mathbb{R}$ and constraint functions $c_i,g_k:\mathbb{R}^d\rightarrow \mathbb{R}$, then  the **objective** of an constrained optimization problem is
# \begin{align*}
#     \min_{x\in\mathbb{R}^n}&\ f(\vvec{x}) \\
#     \text{s.t. }& c_i(\vvec{x}) =0  &\text{ for } 1\leq i \leq m,\\
#                 & g_k(\vvec{x})\geq 0 &\text{ for }1\leq k\leq l
# \end{align*}
# We call the set of vectors satisfying the constraints the **feasible set**:
# $$\mathcal{C}=\{\vvec{x}\mid c_i(\vvec{x})=0, g_k(\vvec{x})\geq 0 \text{ for }1\leq i\leq m, 1\leq k\leq l\}.$$
# ````
# Adding constraints to an optimization problem often makes it more challenging. Now, there are two types of minimizers to consider:
# 
# 1. Interior Minimizers: Points that lie in a "valley" of the objective function, where the function value only increases or remains the same when moving in any direction.
# 2. Boundary Minimizers: Points on the edge of the feasible set, where the function value increases or stays the same when moving *inward* along the feasible region (the function value might decrease when moving out of the feasible set).
# 
# Boundary minimizers can be difficult to identify because traditional methods like the First-Order Necessary Condition (FONC) do not directly apply. However, by using the Lagrangian approach, we can transform a constrained objective into an (almost) unconstrained one, making it possible to find solutions at the boundary.
# ````{prf:definition} Lagrangian
# :label: lagrangian
# Given a constrained optimization objective as in {prf:ref}`constr_objective`, then  the the **Lagrangian function** is defined as
# $$\mathcal{L}(\vvec{x},\bm\lambda,\bm\mu) = f(\vvec{x}) - \sum_{i=1}^m\bm\lambda_i c_i(\vvec{x}) - \sum_{k=1}^l\bm\mu_k g_k(\vvec{x}).$$
# The parameters $\lambda_i\in\mathbb{R}$ and $\mu_k\geq 0$ are called _Lagrange multipliers_.
# ````     
# The Lagrangian introduces an adversarial approach to handling constraints. We remove the constraints on $\vvec{x}$, allowing us to minimize over all $\vvec{x}\in\mathbb{R}^d$, while simultaneously maximizing over the Lagrange multipliers $\bm\lambda\in\mathbb{R}^m$ and $\bm\mu\geq\vvec{0}$.      
# 
# Imagine a scenario where you are trying to minimize the Lagrangian by adjusting $\vvec{x}$, while an "adversary" seeks to maximize it by adjusting $\bm\lambda$ and $\bm\mu$. If $\vvec{x}$ lies outside the feasible set, the adversary can exploit this by driving up the Lagrangian value. For instance, if a constraint $c_i(\vvec{x})=0.1\neq 0$ is unmet, the adversary could set $\lambda_i=-10000$, raising the Lagrangian by $-(-10000)\cdot 0.1=1000$.     
# 
# To prevent these large increases, we need to ensure $\vvec{x}$ is within the feasible set, effectively limiting the adversary's ability to inflate the Lagrangian.     
# 
# The Lagrangian can be used to transform the *primal problem*, which is the original unconstrained objective from {prf:ref}`constr_objective`, to the dual problem.
# ````{prf:definition} Dual Problem
# :label: dual_problem
# Given a primal optimization objective as defined in {prf:ref}`constr_objective`, we define the **dual objective function** $\mathcal{L}_{dual}$, returning the minimum (infimum to be precise) of the Lagrangian, given any $\bm\lambda,\bm\mu\geq\vvec{0}$:
# $$\inf_{\vvec{x}\in\mathbb{R}^d}\mathcal{L}(\vvec{x},\bm\lambda,\bm\mu) = \mathcal{L}_{dual}(\bm\lambda,\bm\mu).$$ the **dual optimization objective** is defined as
# \begin{align*}
# \max_{\bm\lambda, \bm\mu }&\ \mathcal{L}_{dual}(\bm\lambda,\bm\mu) \\
# \text{s.t. }& \bm\lambda\in\mathbb{R}^m, \bm\mu\in\mathbb{R}_+^l
# \end{align*}
# ````
# ```{note}
# The defintion of the dual objective uses the infimum and not the minimum, because there might be some cases where you don't reach the minimum of the Lagrangian for any specific $\vvec{x}$, but only in a limit, e.g. $\vvec{x}\rightarrow\infty$. The infimum returns then the minimum in the limit. If you're not familiar with the concept of the infimum, you can think of it as the minimum.
# ```
# The solution to the dual objective are saddle points: points $(\vvec{x},\bm\lambda,\bm\mu)$ that minimize the Lagrangian subject to $\vvec{x}$ and maximize it subject to $(\bm\lambda,\bm\mu)$. 
# 
# 
# We can easily show that the Lagrangian forms a lower bound of the objective function. For feasible $\vvec{x}\in\mathcal{C}$ and $\bm\lambda\in\mathbb{R}^m,\bm\mu\in\mathbb{R}^l$, $\bm\mu\geq \vvec{0}$ we have
# $$\mathcal{L}(\vvec{x},\bm\lambda,\bm\mu) = f(\vvec{x}) - \sum_{i=1}^m\bm\lambda_i \underbrace{c_i(\vvec{x}}_{=0}) - \sum_{k=1}^l\underbrace{\bm\mu_k}_{\geq 0} \underbrace{g_k(\vvec{x})}_{\geq 0}\leq f(\vvec{x})$$
# There are some cases, where *strong duality* holds, such that every minimizer of the primal objective is a maximizer of the dual objective $f(\vvec{x}^*)= \mathcal{L}_{dual}(\bm\lambda^*,\bm\mu^*)$. One of those cases is if we have a convex optimization objective, which will be discussed in the next section.      
# 
# 
# 

# This allows us to formulate necessary conditions (similar to FONC), called the Karush-Kuhn-Tukker (KKT) conditions that have to be met for a solution to the dual problem.
# ````{prf:theorem} KKT conditions
# Suppose that the objective function $\displaystyle f\colon \mathbb {R} ^{n}\rightarrow \mathbb {R}$ and the constraint functions $\displaystyle c_{i}\colon \mathbb{R} ^{n}\rightarrow \mathbb {R} $ and $\displaystyle g_{k}\colon \mathbb {R} ^{n}\rightarrow \mathbb{R}$ are continuously differentiable. If $ \vvec{x}^*$ is a local minimum, and the constraint functions $c_{i},g_k$ are affine functions, then there exist multipliers $\bm\lambda^*$ and $\bm\mu^*$ such that the following conditions hold: 
# 
# 1. **Stationarity**:
#    $$
#    \nabla_{\vvec{x}} \mathcal{L}(\vvec{x}^*,\bm\lambda^*,\bm\mu^*) = \nabla f(x^*) + \sum_{i=1}^m \lambda_i^* \nabla c_i(x^*) + \sum_{k=1}^l \mu_k^* \nabla g_k(x^*) = 0
#    $$
# 
# 2. **Primal feasibility**:
#    \begin{align*}
#    c_i(\vvec{x}^*) = 0, \quad \forall i
#    g_k(\vvec{x}^*) \geq 0, \quad \forall k
#    \end{align*}
# 
# 3. **Dual feasibility**:
#    $$
#    \mu_k^* \geq 0, \quad \forall k
#    $$
# 
# 4. **Complementary slackness**:
#    $$
#    \mu_k^* g_k(\vvec{x}^*) = 0, \quad \forall k
#    $$
# ````
# The KKT conditions state a set of linear equations, that can be used to obtain candidates for potential minimizers $\vvec{x}^*$.
