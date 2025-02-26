#!/usr/bin/env python
# coding: utf-8

# # Regression Functions
# 
# The possibilities to select a class of functions that are suited to model a given regression task are numerous. Typical function choices are exponential, logarithmic, polynomial functions or any composition of those. How can we make a machine learn which function would be best when we have so many choices?
# 
# The trick which is used in regression is to learn a linear combiation of predefined functions. For example, if we are unsure if the feature-target relation is exponential or logarithmic, then we can define our regression function as a weighted sum of exponential and logarithmic functions:
# ```{math}
# :label: eq_regression_f
# f(x) = \beta_1 \log(x) + \beta_2 \exp(x) 
# ```
# This formalization allows us to learn a function $f$ by means of the parameters $\beta_1$ and $\beta_2$. Those parameters can then be optimized after we define a loss function for the regression task $\ell(f(x),y)$. The functions which we use to compose our function $f$ (here $\log(x)$ and $\exp(x)$) are called **basis functions**.
# 
# The  main insight (which we will discuss in the following for various basis functions) is that nonlinear functions such as $f(x)$ in Eq. {eq}`eq_regression_f` can be represented as linear functions in a _transformed feature space_. Linear functions $f:\mathbb{R}^d\rightarrow \mathbb{R}$ have the form
# $$f(\vvec{x}) = \beta_d x_d + \ldots + \beta_1x = \bm\beta^\top \vvec{x},$$
# that is, they are defined by an inner product of the weights $\bm\beta$ and the feature vector $\vvec{x}$.

# ## Affine Functions 
# We start with a simple function class: the affine functions. Affine functions are linear functions with a bias term. 
# ````{prf:example} Affine functions in two dimensions
# ```{tikz}
# \begin{tikzpicture}
# \begin{axis}[width=.5\textwidth,xlabel=$x$,ylabel=$y$,axis lines = left,xmin=0,xmax=3,ymin=0,ymax=5,yticklabels={,,},extra y ticks={1},
#     extra y tick labels={$\beta_0$},]
# \addplot[blue,thick]
# {1.5*x+1};
# \draw[thick] (1,2.5) -- node[below] {$1$}(2,2.5);
# %\draw[thick] (1,2.5) -- (2,4);
# \draw[thick] (2,2.5) -- node[right] {$\beta_1$}(2,4);
# \end{axis}
# \end{tikzpicture}
# ```
# An affine function $f:\mathbb{R}\rightarrow\mathbb{R}, f(x)= \beta_1x+\beta_0$ has a slope $\beta_1$ and a bias term $\beta_0$. The slope is visible in the graph: if we move one to the right on the horizontal axis, then we go the slope value up (or down, if the slope is negative). The bias term indicates the value where the graph meets the vertical axis. We can write an affine function mapping from the real values as a linear function mapping from the $\mathbb{R}^2$. 
# \begin{align*}
#     f(x)&= \beta_1x+\beta_0 
#     = \begin{pmatrix}1& x\end{pmatrix}
#     \begin{pmatrix}\beta_0  \\ \beta_1
#     \end{pmatrix}
#     = \bm{\phi}(x)^\top\bm{\beta} 
# \end{align*}
# The function $\bm{\phi}(x)=\begin{pmatrix}1\\x\end{pmatrix}$ is called a feature transformation, the vector $\bm\beta\in\mathbb{R}^{2}$ determines the parameters of the function. 
# ````
# ````{prf:example} Affine Functions in Three Dimensions (d=2)
# ```{tikz}
# \begin{tikzpicture}
# \begin{axis}[width=.6\textwidth,
# xlabel=$x_1$,xlabel style={right},
# ylabel=$x_2$,ylabel style={above right},
# zlabel=$y$,zlabel style={anchor=south},
# zmin=0,axis lines = center,zmax=4.1,xmin=0,xmax=2,ymin=0,ymax=5,view={45}{45},
# yticklabels={,,},xticklabels={,,},zticklabels={,,},extra z ticks={1},
#     extra z tick labels={$\beta_0$}]
# \addplot3[
#     mesh, samples=10, domain=0:2,
# ]
# {0.5*x+0.7*y+1};
# \draw[thick] (0,0,1) -- node[below] {$1$}(1,0,1);
# \draw[thick] (0,0,1) -- (1,0,1.5);
# \draw[thick] (1,0,1) -- node[right] {$\beta_1$}(1,0,1.5);
# %
# \draw[thick] (2,0,2) -- node[below] {$1$}(2,1,2);
# \draw[thick] (2,0,2) -- (2,1,2.7);
# \draw[thick] (2,1,2) -- node[right] {$\beta_2$}(2,1,2.7);
# \end{axis}
# \end{tikzpicture}
# ```
# An affine function $f:\mathbb{R}^2\rightarrow\mathbb{R}, f(\vvec{x})= \beta_2x_2+\beta_1x_1+\beta_0$ has two slopes $\beta_1,\beta_2$ and a bias term $\beta_0$. In direction of the $x_1$ coordinate, the function has slope $\beta_1$ and in direction of the $x_2$ coordinate it has the slope $\beta_2$. As a result, an affine function in two variables looks like a flat surface. The bias term incates again the value where the graph meets the horizontal axis. We can write an affine function mapping from $\mathbb{R}^2$ as a linear function mapping from the $\mathbb{R}^3$. 
# \begin{align*}
#     f(\vvec{x})&= \beta_2x_2+\beta_1x_1+\beta_0 
#     =\begin{pmatrix}
#     1& x_1&x_2\end{pmatrix}
#     \begin{pmatrix}
#     \beta_0 \\ \beta_1 \\ \beta_2\end{pmatrix}
#     =\bm{\phi}(\vvec{x})^\top\bm{\beta},
# \end{align*}
# where the feature transformation is defined as
# $\bm{\phi}(\vvec{x})=\begin{pmatrix}1\\x_1\\x_2\end{pmatrix}$ and the parameters are given by $\bm\beta\in\mathbb{R}^{3}$.
# ````
# 
# From the examples, we can see how to generalize the feature transformation $\bm\phi$ to any affine function $f:\mathbb{R}^d\rightarrow \mathbb{R}$:  
# \begin{align*}
#     \bm{\phi}_{aff}(\vvec{x}) = \begin{pmatrix}
#     1\\\vvec{x}
#     \end{pmatrix}\in\mathbb{R}^{d+1}
# \end{align*}
# 
# As a result, we get a parametrization of the function class of affine functions as the inner product of the feature transformation vector and the parameter vector, which indicates a linear function:
# $$\mathcal{F}_{aff}=\left\{f:\mathbb{R}^d\rightarrow \mathbb{R}, f(\vvec{x})= \bm{\phi}_{aff}(\vvec{x})^\top\bm{\beta}\middle\vert \bm{\beta}\in\mathbb{R}^{d+1}\right\}
# $$
# 
# ## Polynomials
# Functions that contain curvature can be modelled as a linear function in a transformed function space. Here we explore the function class of polynomials.
# ````{prf:example} Polynomials of Degree k=2 (d=1)
# ```{tikz}
# \begin{tikzpicture}
# \begin{axis}[width=.5\textwidth,xlabel=$x_1$,ylabel=$y$,axis lines = left,xmin=-0.1,xmax=4,ymin=0,ymax=5,yticklabels={,,},xticklabels={,,},extra y ticks={1},
#     extra y tick labels={$c$},extra x ticks={2},extra x tick labels={$b$}]
# \addplot[blue,thick,smooth]
# {(x-2)^2+1};
# \end{axis}
# \end{tikzpicture}
# ```
# A function $f:\mathbb{R}\rightarrow\mathbb{R},\ f(x)= a(x-b)^2+c$, mapping a real value to a polynomial of degree two, is defined over three parameters $(a,b,c)$. The minimizer is given by the point $(b,c)$ and the value of $a$ determines how narrow or wide the parabola is. We can write this function as a linear function in a three-dimensional transformed feature space.  
# \begin{align*} 
#     f(x)&= a(x-b)^2+c\\
#     &=ax^2 -2abx+ab^2+c\\
#     &=\beta_2x^2+\beta_1x+ \beta_0\\
#     &= 
#     \begin{pmatrix} 
#         1&x& x^2
#     \end{pmatrix}
#     \begin{pmatrix}
#         \beta_0 \\ \beta_1 \\ \beta_2
#     \end{pmatrix}= \bm{\phi}(x)^\top\bm{\beta}.
# \end{align*}
# The feature transformation $\bm{\phi}(x)=\begin{pmatrix}1\\ x\\ x^2\end{pmatrix}$ maps to three basis functions $f_1(x)=1$, $f_2(x)=x$ and $f_3(x)=x^2$. A weighted sum of these basis functions defines then the parabola, where the parameters are given by the vector $\bm\beta\in\mathbb{R}^{3}$. Note that we can't directly read the properties of the parabola from the $\beta$-parameters as we did with the parameters $(a,b,c)$. If needed, we could compute the original parameters from $\beta$ though. 
# ````
# ````{prf:example} Polynomials of Degree k (d=1)
# ```{tikz}
# \begin{tikzpicture}
# \begin{axis}[width=.5\textwidth,xlabel=$x$,ylabel=$y$,axis lines = center, domain=0:4.5,yticklabels={,,},xticklabels={,,}]
# \addplot[blue,thick]
# {0.001*(x+1)*(x-0.5)*(x-2)*(x-3)*(x-4)*(x-7)*(x-10)+100};
# \end{axis}
# \end{tikzpicture}
# ```
# The parametrization of a parabola over the linear product of the basis functions and a weight vector $\bm\beta$ is easily generalizable to the parametrization of a polynomial of degree $k$. We consider here still only functions $f:\mathbb{R}\rightarrow\mathbb{R}$, mapping from the one-dimensional space. 
# \begin{align*}
#     f(x)&=\beta_kx^k+\ldots+\beta_1x+ \beta_0\\
#     &= \begin{pmatrix}
#     1&\ldots & x^k\end{pmatrix}
#     \begin{pmatrix} \beta_0 \\ \vdots \\ \beta_k
#     \end{pmatrix}= \bm{\phi}(x)^\top\bm{\beta}
# \end{align*}
# The feature transformation is here $\bm{\phi}(x)=\begin{pmatrix}1\\ x\\\vdots\\x^k\end{pmatrix}$, and the parameter vector is $\bm\beta\in\mathbb{R}^{k+1}$.
# ````
# ````{prf:example} Multivariate Polynomials of Degree k (d=2)
# ```{tikz}
# \begin{tikzpicture}
# \begin{axis}[width=.6\textwidth,xlabel=$x_1$,ylabel=$x_2$,zlabel=$y$,zmin=0,axis lines = center,zmax=4.1,xmin=0,xmax=2.5,ymin=0,ymax=5,view={45}{45},
# yticklabels={,,},xticklabels={,,},zticklabels={,,}]
# \addplot3[
#     mesh, samples=10, domain=0:2.5,blue
# ]
# {0.5*x^2-0.4*y^3+y^2+1};
# \end{axis}
# \end{tikzpicture}
# ```
# We discuss now a degree $k$ polynomial $f:\mathbb{R}^2\rightarrow \mathbb{R}$ mapping from the two-dimensional space $\mathbb{R}^2$. There are actually various ways to define the polynomial in a vector space. A popular way to define a polynomial in more than two variables is over a weighted sum of all combinations of the one-dimensional basis functions. Using now a multi index, a polynomial in two variables is defined as 
# \begin{align*}
#     f(\vvec{x})&=\sum_{i_1=0}^k\sum_{i_2=0}^k\beta_{i_1i_2}x_1^{i_1}x_2^{i_2}\\
#     &= \underbrace{\begin{pmatrix}1 &  \ldots & x_1^{k}x_2^{k-1}&x_1^kx_2^k\end{pmatrix}}_{=:\bm\phi(\vvec{x})^\top}
#     \begin{pmatrix}\beta_{00} \\ \\\vdots \\ 
#     \beta_{k(k-1)}\\
#     \beta_{kk}
#     \end{pmatrix}= \bm{\phi}(\vvec{x})^\top\bm{\beta},
# \end{align*}
# where the feature transformation maps now to a $(k+1)^2$-dimensional vector space, $\bm\phi(\vvec{x}),\bm\beta\in\mathbb{R}^{(k+1)^2}$. The basis functions are here the set of $\{x_{i_1}x_{i_2}\mid 1\leq i_1,i_2\leq k\}$. 
# ````
# We generalize now the defintion of polynomials of degree $k$ as a linear function by the multiplication of all possible one-dimensional basis functions:
# $$\bm{\phi}_{pk}(\vvec{x}) = (x_1^{i_1}\cdot \ldots \cdot x_d^{i_d})_{1\leq i_1\ldots i_d\leq k} \in\mathbb{R}^{(k+1)^d}, \text{ for }\vvec{x}\in\mathbb{R}^d$$
# Hence, our function class of polynomial functions in a $d$-dimensional vector space is given as:
# $$\mathcal{F}_{pk}=\left\{f:\mathbb{R}^d\rightarrow \mathbb{R},f(\vvec{x})=\bm{\phi}_{pk}(\vvec{x})^\top \bm{\beta} \middle\vert 
#  \bm{\beta}\in\mathbb{R}^{(k+1)^d} 
# \right\}
# $$
# Another definition of polynomials multiplies only basis functions such that the sum of all exponents is at most $k$. In this case we get the following feature transformation, called $\bm{\phi}_{pka}$, where the *a* stands for alternative:
# $$\bm{\phi}_{pka}(\vvec{x}) = (x_1^{i_1}\cdot \ldots \cdot x_d^{i_d})_{i_1+\ldots +i_d\leq k}.$$
# The sklearn function to obtain polynomial features uses this definition. This feature transformation maps to a lower-dimensional transformed feature space than $\bm{\phi}_pk$, but a general issue is that the dimensionality of the transformed feature space increases vastly in the degree or the amount of features. 
# 
# 
# ##  Gaussian Functions
# We introduce now a third way to define a function class for the regression task. This method has the advantage that the dimensionality of the transformed feature space is easy to adjust. The idea is to use Gaussian functions as basis functions.  
# ````{tikz}
# \begin{tikzpicture}
# 	\begin{axis}[ylabel=$y$,xlabel=$x$,width=.7\textwidth,height=.6\textwidth, axis x line=left,ymax=1.1, domain=-4:8, axis y line=center, tick align=outside,legend to name=zelda,legend entries={$\exp(-(x-1)^2)$,$\exp(-\frac{(x-1)^2}{4})$,$\exp(-4(x-1)^2)$}]
# 		\addplot+[mark=none,smooth,thick,magenta] (\x,{exp(-(\x-1)^2)});
#         \addplot+[mark=none,smooth,blue,thick] (\x,{exp(-(\x-1)^2/4)});
#         \addplot+[mark=none,smooth,green,thick] (\x,{exp(-(\x-1)^2*4)});
#         \coordinate (top) at (rel axis cs:0,1);
#         \coordinate (bot) at (rel axis cs:0,0);
# 	\end{axis}
# 	\path (top)--(bot) coordinate[midway] (group center);
#     \node[right=1em,inner sep=0pt] at(group center -| current bounding box.east) {\pgfplotslegendfromname{zelda}};
# \end{tikzpicture}
# ````
# The Gaussian radial functions are parametrized by a scaling factor $\gamma$ and the mid point $\bm\mu$.  
# $$
#     \kappa(\mathbf{x})=\exp\left(-\gamma\lVert\mathbf{x}-\bm\mu\rVert^2\right)
# $$
# Those parameters ($\gamma$ and $\bm\mu$) have to be set by the user. We can't learn them in the linear regression framework, since we can only learn coefficients of the basis functions. The parameters $\gamma$ and $\bm\mu$ are however within the exponential term.
# 
# ````{prf:example} Local Gaussian Radial Basis Functions
# ```{tikz}
# \begin{tikzpicture}
# \begin{axis}[width=.6\textwidth,xlabel=$x$,ylabel=$y$,axis lines = center, domain=0:4.5,yticklabels={,,},xticklabels={,,},extra x ticks={0.5,3},extra x tick labels={$\mu_1$,$\mu_2$}]
# \addplot+[magenta,thick, mark=none]
# {0.5*exp(-(x-0.5)^2)};
# \addplot+[magenta,thick, mark=none]
# {exp(-(x-3)^2)};
# \addplot+[blue,ultra thick,mark=none]
# {0.5*exp(-(x-0.5)^2/1.1) + exp(-(x-3)^2/1.1)+0.02};
# \end{axis}
# \end{tikzpicture}
# ```
# The plot above shows the graph that we get when defining our function as $f:\mathbb{R}\rightarrow\mathbb{R}, \ f(x)=0.5\exp(-(x-0.5)^2) + \exp(-(x-3)^2)$. The graph of the added Gaussians has two maxima. Approximating a graph with a polynomial that has two maximizers requires a degree of four, which translates to five basis functions. With Gaussian basis functions, we need only two. 
# ````
# The sum of weighted Gaussian basis functions is modelled by a linear function as follows:
# 
# \begin{align*}
#     f(x)&=\sum_{i=1}^k\beta_i\exp\left(-\frac{\lVert x-\mu_i\rVert^2}{2\sigma^2}\right)\\
#     &= \begin{pmatrix}\kappa_1(x)&\ldots & \kappa_k(x)\end{pmatrix}
#     \begin{pmatrix}\beta_1 \\ \vdots \\ \beta_k
#     \end{pmatrix}\\
#     &= \bm{\phi}(x)^\top\bm{\beta},
# \end{align*}
# The feature transformation $\bm{\phi}(x)$ has a dimensionality equal to the number of selected basis functions.
# We define the function class of a sum of $k$ Gaussians as:
# \begin{align*}
#     \bm{\phi}_{Gk}(\vvec{x}) = \begin{pmatrix}\exp(-\gamma\lVert\mathbf{x}-\bm\mu_1\rVert^2)\ldots \exp(-\gamma\lVert\mathbf{x}-\bm\mu_k\rVert^2)\end{pmatrix} 
# \end{align*}
# The drawback of using Gaussian radial basis functions is that we need to determine the mean values beforehand. A popular strategy is to select a subset of the training data points as the mean values or a predefined grid of  points.
# ## Summary
# In summary, we have defined three function classes:
# 1. Affine functions:
# $$\mathcal{F}_{aff}=\left\{f:\mathbb{R}^d\rightarrow \mathbb{R}, f(\vvec{x})= \bm{\phi}_{aff}(\vvec{x})^\top\bm{\beta}\middle\vert \bm{\beta}\in\mathbb{R}^{d+1}\right\}
# $$
# 2. Polynomials of degree $k$:
# $$\mathcal{F}_{pk}=\left\{f:\mathbb{R}^d\rightarrow \mathbb{R},f(\vvec{x})=\bm{\phi}_{pk}(\vvec{x})^\top \bm{\beta} \middle\vert 
#  \bm{\beta}\in\mathbb{R}^{(k+1)^d} 
# \right\}
# $$
# 3. Sum of $k$ Gaussians:
# $$
# \mathcal{F}_{Gk}=\left\{f:\mathbb{R}^d\rightarrow\mathbb{R},f(\vvec{x})=\bm{\phi}_{Gk}(\vvec{x})^\top\bm{\beta}\middle\vert \bm{\beta}\in\mathbb{R}^k\right\}
# $$
