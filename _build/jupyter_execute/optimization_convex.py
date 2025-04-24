#!/usr/bin/env python
# coding: utf-8

# ## Convex Optimization
# Convex optimization represents a somewhat ideal scenario in optimization problems. With a convex objective, there is only one minimum to find, simplifying the task. Many optimization problems in machine learning are convex, or at least convex in certain coordinates when others are fixed. Knowing whether an objective is convex is valuable for understanding how to improve a model. If a model derived from minimizing the objective function performs poorly on other metrics, this may indicate that the objective does not fully capture what defines a good model. To establish convex optimization objectives, we first define convex sets and convex functions.
# ````{prf:definition} Convex set
# :label: convex_set 
# A **set** $\mathcal{X}\subseteq \mathbb{R}^d$ is **convex** if and only if the line segment between every pair of points in the set is in the set. 
# That is, if for all $x,y\in \mathcal{X}$ and $\alpha\in[0,1]$ we have
# $$\alpha \vvec{x}+(1-\alpha) \vvec{y}\in\mathcal{X}.$$
# ````
# The definition states that in a convex set you can connect any two dots within the set and you will never leave the set when drawing that connecting line. The set on the left is convex, and the set on the right is not convex.
# ```{tikz}
# \begin{tikzpicture}
#     \draw[rotate=-45,fill=blue!30] (-3,-3) ellipse (30pt and 45pt);
#     \draw[fill=blue!30] (0,0) to [out=140,in=90] (-1,-1)
#         to [out=-90,in=240] (0.8,-0.6)
#         to [out=60,in=-60] (1.2,1.2)
#         to [out=120,in=90] (0.3,0.7)
#         to [out=-90,in=20] (0.3,0)
#         to [out=200,in=-40] (0,0);
#     \draw (-0.5,-0.5) -- (0.7,0.7);
#     \fill (-0.5,-0.5) circle[radius=1.5pt];
#     \fill (0.7,0.7) circle[radius=1.5pt];
# \end{tikzpicture}
# ```
# 
# ````{prf:definition} Convex function
# :label: convex_function
# A function $f:\mathbb{R}^d\rightarrow\mathbb{R}$ is **convex** if and only if for every $\alpha \in[0,1]$, and $\vvec{x},\vvec{y}\in\mathbb{R}^d$:
# \begin{equation}
#  f(\alpha \vvec{x}+ (1-\alpha)\vvec{y})\leq\alpha f(\vvec{x})+(1-\alpha)f(\vvec{y})\nonumber
# \end{equation}
# ````
# A convex function is a function whose graph is always below the line that connects two dots on the graph. Below you see an illustration of that property.
# ```{tikz}
# :libs: arrows.meta, intersections
# \pgfplotsset{compat=newest}
# \pgfplotsset{plot coordinates/math parser=false}
# \pgfplotsset{
#     every non boxed x axis/.style={
#         xtick align=center,
#         enlarge x limits=true,
#         x axis line style={line width=0.8pt, -latex}
# },
#     every boxed x axis/.style={}, enlargelimits=false
# }
# \pgfplotsset{
#     every non boxed y axis/.style={
#         ytick align=center,
#         enlarge y limits=true,
#         y axis line style={line width=0.8pt, -latex}
# },
#     every boxed y axis/.style={}, enlargelimits=false
# }
# 
# 
# \begin{tikzpicture}
# \begin{axis}[width=\textwidth,axis equal image,
#     axis lines=middle,
#     xmin=0,xmax=9,
#     ymin=-0.8,ymax=3,
#     xtick={\empty},ytick={\empty}, axis on top
# ]
#  
# \addplot[ultra thick,domain=0.5:5.5,magenta,name path = A]  {-x/3 + 2.75} coordinate[pos=0.4] (m) ;
# \draw[ultra thick,blue, name path =B] (0.15,4) .. controls (1,1) and (4,0) .. (6,2) node[pos=1, color=black, right]  {$f(x)$} coordinate[pos=0.075] (a1)  coordinate[pos=0.95] (a2);
# \path [name intersections={of=A and B, by={a,b}}];
# 
# 
# \draw[densely dashed] (0,0) -| node[pos=0.5, color=black, label=below:$x$] {}(a);
# \draw[densely dashed, name path=D] (3,0) -|node[pos=0.5, color=black, label={[label distance=0.2cm]below:$\alpha x+ (1-\alpha)y$}] {} node[pos=1, fill,circle,inner sep=1pt] {}(m);
# \draw[densely dashed] (0,0) -|node[pos=0.5, color=black, label=below:$y$] {}(b);
#  
# \path [name intersections={of=B and D, by={c}}] node[fill,circle,inner sep=1pt] at (c) {}; 
#  
# \node[anchor=south west, text=black] (d) at (6,0.9) {$f(\alpha x+(1-\alpha) y )$};
# \node[anchor=south west, text=black] (e) at (5,2.5) {$\alpha f(x)+(1-\alpha)f(y)$};
# \draw[ densely dashed] (d) -- (c);
# \draw[ densely dashed] (e) -- (m);
# \end{axis}
# \end{tikzpicture}
# ```
# A convex optimization problem, or objective, is now characterized by two things: a convex objective function and a convex feasible set.
# ````{prf:definition} Convex optimization problem
# Given a convex objective function $f:\mathbb{R}^d\rightarrow \mathbb{R}$ and a convex feasible set $\mathcal{C}\subseteq\mathbb{R}^d$
# then the objective of a **convex optimization problem** can be written in the following form:
# \begin{align*}
#     \min_{\vvec{x}\in\mathbb{R}^d}&\ f(\vvec{x}) &
#     \text{s.t. } \vvec{x}\in\mathcal{C}
# \end{align*}
# ````
# The fact that not only the objective function, but also the feasible set has to be convex might come here as a surprise. This is more easily understood if you transform the constrained optimization problem into an unconstrained one. By using the indicator function 
# $$\mathbb{1}_\mathcal{C}(\vvec{x})=\begin{cases}0 & \text{if }\vvec{x}\in\mathcal{C}\\ \infty& \text{if }\vvec{x}\notin\mathcal{C} \end{cases}$$ 
# we can write any constrained objective as the following unconstrained one:
# \begin{align*}
# \min_{\vvec{x}\in\mathbb{R}^d} f(\vvec{x}) +\mathbb{1}_\mathcal{C}(\vvec{x})
# \end{align*}
# Since the objective function above returns infinity for any $\vvec{x}$ outside of the feasible set, any potential minimizer has to be feasible. If we now apply the definition of a convex function, then we require that the graph of the function is below the connecting line. However, if we have a nonconvex feasible set, then there exist two points in the feasible set whose connecting line is partially outside of the feasible set. On this part, where the connecting line is outside of the feasible set, the objective function returns infinity. Hence, the graph of the objective will not be under the connecting line. The example below illustrates this property further.
# 
# ````{prf:example}
# The figure below displays the function $f(x)=x^2 + \mathbb{1}_{\{\lvert x\rvert\geq 2\}}$. The feasible set $\{\lvert x\rvert\geq 2\}$ is nonconvex, since the line connecting $x_1=-2.5$ and $x_2=3.5$ leaves the feasible set. We indicate domain where the function is returning infinity by the dashed vertical lines. If we now connect two function values as indicated by the red line, then the graph is above the line within the interval $[-2,2]$. This shows that the function $f$, indicating an optimization objective with a convex objective function and a nonconvex feasible set is not a convex function. 
# ```{tikz}
# \begin{tikzpicture}
#     \begin{axis}[
#         domain=-4:4,
#         ymin=-1, ymax=20,
#         samples=100,
#         xlabel={$x$},
#         ylabel={$f(x)$},
#         xtick={-4,-2,0,2,4},
#         ytick={0,4,8,12,16,20},
#         yticklabel style={/pgf/number format/fixed},
#         restrict y to domain=-1:20,
#         extra y tick labels={------}
#     ]
#     
#     % Plot x^2 for x < -2
#     \addplot[domain=-4:-2, thick, blue] {x^2};
#     
#     % Plot infinity line for -2 <= x <= 2
#     \draw[thick, dashed, blue] (axis cs:2,4) -- (axis cs:2,20);
#     \draw[thick, dashed, blue] (axis cs:-2,4) -- (axis cs:-2,20);
#     
#     % Plot x^2 for x > 2
#     \addplot[domain=2:4, thick, blue] {x^2};
#     
#     % Plot specific points at x = -2.5 and x = 3.5
#     \addplot[only marks, mark=*] coordinates {(-2.5,6.25) (3.5,12.25)};
#     
#     % Draw line connecting the points (-2.5, 6.25) and (3.5, 12.25)
#     \addplot[thick, magenta] coordinates {(-2.5,6.25) (3.5,12.25)};
#     
#     \end{axis}
# \end{tikzpicture}
# ```
# ````
# ````{prf:theorem}
# :label: thm_convex
#  If a function $f:\mathbb{R}^d\rightarrow\mathbb{R}$ is convex, then every local minimizer $x^*$ of $f$ is a global minimizer.
# ````
# ````{toggle}
# ```{prf:proof} 
# Assume that a convex function $f$ has a local minimizer $x_{loc}$ which is not a global minimizer: $f(x_{loc})>f(x^*)$. Then going towards $x^*$ from $x_{loc}$ minimizes the function value, because the graph is below the connecting line from $x_{loc}$ to $x^*$. More formally, we have for any $\alpha<1$:
# \begin{align*}
# f(\alpha x_{loc} + (1-\alpha)x^*)&\leq \alpha f(x_{loc}) + (1-\alpha)f(x^*)\\
# &< \alpha f(x_{loc}) + (1-\alpha)f(x_{loc})\\
# & = f(x_{loc})
# \end{align*}
# where the first inequality stems from the definition of convex functions, and the second inequality from the fact that $f(x_{loc})>f(x^*)$. As a result, we can find in any neighborhood of $x_{loc}$ a point $v = \alpha x_{loc} + (1-\alpha)x^*$, by choosing $\alpha$ close enough to one, such that $f(v)<f(x_{loc})$.
# Hence $x_{loc}$ is not a local minimizer.
# ```
# ````
# Since we can state any (constrained) optimization objective as an unconstrained objective by means of the indicator function as discussed above, we can apply {prf:ref}`thm_convex` also to constrained optimization objectives and deduce that any local minimizer of a constrained objective is also a global one.      
# 
# Conversely, not every function with one global minimum has to be convex. An example of a nonconvex function that has just one local, and therewith also global minimum is the Rosenbrock function.
# 
# ### Properties of Convex Functions
# We can show that a function is convex by applying the definition. However, in practice it's useful to rely on a couple of properties that allow to quickly assess whether a function is convex. First of all, there are some very basic convex functions, like affine functions and norms.
# ```{prf:lemma}
# Every norm $\lVert\cdot\rVert:\mathbb{R}^d\rightarrow \mathbb{R}_+$ is a convex function.
# ```
# ```{prf:proof} 
# For any $\vvec{x},\vvec{y}\in\mathbb{R}^d$ and $\alpha\in[0,1]$ we have:
# \begin{align*}
#     \lVert\alpha\vvec{x} +(1-\alpha)\vvec{y} \rVert &\leq \lVert\alpha\vvec{x}\rVert +\lVert(1-\alpha)\vvec{y} \rVert &\text{(triangle inequality)}\\
#     &= \lvert\alpha\rvert\lVert\vvec{x}\rVert +\lvert1-\alpha\rvert\lVert\vvec{y} \rVert &\text{(homogeneity)}\\
#     &=\alpha\lVert\vvec{x}\rVert +(1-\alpha)\lVert\vvec{y} \rVert
# \end{align*}
# ```
# The plot below shows the graphs of the $L_p$ norm, which looks the same for $x\in\mathbb{R}$, regardless of $p$ because $\lVert x\rVert = \sqrt[p]{\lvert x\rvert^p} = \lvert x\rvert$. We can clearly observe the typical convex function shape.
# ```{tikz}
# \begin{tikzpicture}
# \begin{axis}[
#     width=.5\textwidth, % Slightly larger width for more space
#     xlabel=$x$,
#     ylabel=$y$,
#     axis lines = center,
#     domain=-1:1,
#     ymin=0, ymax=1,
#     yticklabels={,,},
#     xticklabels={,,},
#     legend pos=north west % Positioning the legend in the upper left corner
# ]
# 
# % Plot of the l1-norm
# \addplot[blue, thick] {abs(x)};
# \addlegendentry{$L_p$-norm}
# \end{axis}
# \end{tikzpicture}
# ```
# ````{prf:lemma}
# The squared $L_2$-norm $f:\mathbb{R}^d\rightarrow \mathbb{R}$, $f(\vvec{x})=\lVert\vvec{x}\rVert^2$ is convex. 
# ````
# ````{toggle}
# ```{prf:proof}
# Let $\vvec{x}_1,\vvec{x}_2\in\mathbb{R}^d$ and $\alpha\in[0,1]$. Then we have to show  that
# :::{math}
# :label: eq:convex
# \begin{align}
#     \lVert\alpha\vvec{x}_1 +(1-\alpha)\vvec{x}_2\rVert^2 \leq  \alpha\lVert\vvec{x}_1\rVert^2 + (1-\alpha)\lVert\vvec{x}_2\rVert^2
# \end{align}
# :::
# We apply the binomial formula for the squared $L_2$-norm, which derives directly from the definition of the squared $L_2$-norm by an inner product (see linear algebra lecture). Then we have:
# :::{math}
# :label: eq:term
# \begin{align}
#     \lVert\alpha\vvec{x}_1 +(1-\alpha)\vvec{x}_2\rVert^2 &=
#     \lVert\alpha\vvec{x}_1\rVert^2 +2\alpha(1-\alpha)\vvec{x}_1\vvec{x}_2 +\lVert(1-\alpha)\vvec{x}_2\rVert^2\nonumber\\
#     &=
#     \lvert\alpha\rvert^2\lVert\vvec{x}_1\rVert^2 +2\alpha(1-\alpha)\vvec{x}_1\vvec{x}_2 +\lvert 1-\alpha\rvert^2\lVert\vvec{x}_2\rVert^2 &\text{(homogenity of the norm)}\nonumber\\
#     &=
#     \alpha^2\lVert\vvec{x}_1\rVert^2 +2\alpha(1-\alpha)\vvec{x}_1\vvec{x}_2 +( 1-\alpha)^2\lVert\vvec{x}_2\rVert^2,
# \end{align}
# :::
# where the last equation derives from the fact that the squared absolute value of a real value is equal to the  squared real value.
# 
# What is standing above, is not yet what we want, and it is difficult to see which step has to be taken next to derive the Inequality {eq}`eq:convex`. Hence, we apply a trick. Instead of showing that  Inequality {eq}`eq:convex` holds as it stands, we show that an equivalent inequality holds:
# \begin{align*}
#     \lVert\alpha\vvec{x}_1 +(1-\alpha)\vvec{x}_2\rVert^2 - \alpha\lVert\vvec{x}_1\rVert^2 - (1-\alpha)\lVert\vvec{x}_2\rVert^2\leq 0 
# \end{align*}
# We substitute now the result of Eq. {eq}`eq:term` into the term on the left of the inequality above:
# \begin{align*}
#     &\lVert\alpha\vvec{x}_1 +(1-\alpha)\vvec{x}_2\rVert^2 - \alpha\lVert\vvec{x}_1\rVert^2 - (1-\alpha)\lVert\vvec{x}_2\rVert^2\\ 
#      &\quad=
#     \alpha^2\lVert\vvec{x}_1\rVert^2 +2\alpha(1-\alpha)\vvec{x}_1\vvec{x}_2 +( 1-\alpha)^2\lVert\vvec{x}_2\rVert^2- \alpha\lVert\vvec{x}_1\rVert^2 - (1-\alpha)\lVert\vvec{x}_2\rVert^2
#     \\
#     &\quad= -\alpha(1-\alpha)\lVert\vvec{x}_1\rVert^2  +2\alpha(1-\alpha)\vvec{x}_1\vvec{x}_2 -( 1-\alpha)(1-1+\alpha)\lVert\vvec{x}_2\rVert^2\\
#     &\quad= - \alpha(1-\alpha) \lVert\vvec{x}_1-\vvec{x}_2\rVert^2&\text{(binomial formula)}\\
#     &\quad\leq 0
# \end{align*}
# This concludes what we wanted to show.
# ```
# ````
# ```{prf:lemma}
# Every linear function $f:\mathbb{R}^d\rightarrow\mathbb{R}$ is convex and concave (that is -$f$ is convex).
# ```
# ```{prf:proof} 
# Linear functions mapping to real values are parametrized by a matrix (more specifically a row vector in this case) $A\in\mathbb{R}^{1\times d}$, having the form $f(\vvec{x})=A\vvec{x}$. This type of function is further characterized by the property that for any $\vvec{x},\vvec{y}\in\mathbb{R}^d$ and $\alpha\in[0,1]$ we have:
# \begin{align*}
#     f(\alpha\vvec{x} +(1-\alpha)\vvec{y}) &= \alpha f(\vvec{x}) +(1-\alpha)f(\vvec{y}).
# \end{align*}
# The equation above satisfies the inequality defining convex functions. This equality also applies to $-f$, hence showing that $f$ is also concave.
# ```
# The plot below shows the graph of a linear function. We can see how this is a special case, since any line connecting two function values lie exactly on the graph.
# ```{tikz}
# \begin{tikzpicture}
# \begin{axis}[width=.4\textwidth,xlabel=$x$,ylabel=$y$,axis lines = center, domain=-1:1,yticklabels={,,},xticklabels={,,}]
# \addplot[blue,thick]
# {x};
# \end{axis}
# \end{tikzpicture}
# ```
# Further, there are some specific compositions of functions that result in convex functions.
# ```{prf:lemma}
# Nonnegative weighted sums of convex functions are convex:     
# for all $\lambda_1,\ldots,\lambda_k\geq 0$ and convex functions $f_1,\ldots,f_k$ the function 
# $$f(\vvec{x}) = \lambda_1 f_1(\vvec{x})+\ldots + \lambda_k f_k(\vvec{x})$$
# is convex.
# ```
# ```{prf:proof} 
# Exercise
# ```
# ```{prf:lemma}
# If $g:\mathbb{R}^d\rightarrow \mathbb{R}^k$ is an affine map (that is, $g$ has the form $g(\vvec{x})=A\vvec{x}+\vvec{b}$ for a matrix $A$ and vector $\vvec{b}$), and $f:\mathbb{R}^k\rightarrow \mathbb{R}$ is a convex function, then the composition 
# $$f(g(\vvec{x}))=f(A\vvec{x}+\vvec{b})$$
#         is a convex function. 
# ```
# ```{prf:proof} 
# Exercise
# ```
# 
# 
