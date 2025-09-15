# Minimizing the RSS

Assuming that we have now selected a function class, and that it can be modelled as a linear function $f(\vvec{x})=\bm\phi(\vvec{x})^\top\bm\beta$, we need to train the parameter vector $\bm\beta$ to fit the dataset. For that reason, we need to define an objective function that is small if the model $f(\vvec{x})$ is suitable. Since our goal is to approximate the target values, we can simly measure the distance of our prediction $f(\vvec{x})$ and the target value $y$.   
````{tikz}
%\pgfplotsset{
%	colormap={test}{[2pt]
%    	rgb=(0.8,0.2,0.4);
%       rgb=(0.8,0.2,0.4);
%    },
%}
\pgfplotsset{compat=newest}
\pgfmathsetseed{10} % set the random seed
\pgfplotstableset{ % Define the equations for x and y
    create on use/x/.style={create col/expr={42+2*\pgfplotstablerow}},
    create on use/y/.style={create col/expr={(0.6*\thisrow{x}+130)+8*rand}}
}
% create a new table with 30 rows and columns x and y:
\pgfplotstablenew[columns={x,y}]{30}\loadedtable

% Calculate the regression line
\pgfplotstablecreatecol[linear regression]{regression}{\loadedtable}

\pgfplotsset{
    colored residuals/.style 2 args={
        only marks,
        scatter,
        point meta=explicit,
        colormap={redblue}{color=(#1) color=(#2)},
        error bars/y dir=minus,
        error bars/y explicit,
        error bars/draw error bar/.code 2 args={
            \pgfkeys{/pgf/fpu=true}
            \pgfmathtruncatemacro\positiveresidual{\pgfplotspointmeta<0}
            \pgfkeys{/pgf/fpu=false}
            \ifnum\positiveresidual=0
                \draw [#2] ##1 -- ##2;
            \else
                \draw [#1] ##1 -- ##2;
            \fi
        },
        /pgfplots/table/.cd,
            meta expr=(\thisrow{y}-\thisrow{regression})/abs(\thisrow{y}-\thisrow{regression}),
            y error expr=\thisrow{y}-\thisrow{regression}
    },
    colored residuals/.default={magenta}{magenta}
}
\begin{tikzpicture}
\begin{axis}[
xlabel=$x$, % label x axis
ylabel=$y$, % label y axis
axis lines=left, %set the position of the axes
xmin=40, xmax=105, % set the min and max values of the x-axis
ymin=150, ymax=200, % set the min and max values of the y-axis
yticklabels={,,},xticklabels={,,}
]

\makeatletter
\addplot [colored residuals] table {\loadedtable};
\addplot [
    no markers,
    thick, blue
] table [y=regression] {\loadedtable} ;
\end{axis}
\end{tikzpicture}
````
The plot above shows a set of datapoints that are approximated by an affine model (blue). The distance to the target $y$, plotted on the vertical axis, is indicated by the red bars. The distance indicated by the red bars reflect the absolute values $\lvert y_i - f(\vvec{x}_i)\rvert$. However, the absolute value is not so easy to optimize, since it is non-differentiable at value zero. Instead, we can minimize the squared distances, which gives us a smooth objective function. 

The squared approximation error of a function $f$ to the target values $y$ can be compactly written as follows for a linear model $f(\vvec{x})=\bm\phi(\vvec{x})^\top\bm\beta$
\begin{align*}
    RSS(\bm{\beta}) &= \sum_{i=1}^n(y_i-f(\vvec{x}_i))^2\\
    &= \sum_{i=1}^n(y_i-\bm{\phi}(\vvec{x}_i)^\top\bm{\beta})^2\\
    &= \sum_{i=1}^n(y_i-X_{i\cdot}\bm{\beta})^2\\
    &=\lVert \vvec{y}-X\bm{\beta}\rVert^2.
\end{align*}
The function $RSS(\bm{\beta})$ is called the **Residual Sum of Squares**. We have defined above the matrix $X$, that gathers the transformed feature vectors $\bm{\phi}(\vvec{x}_i)^\top = X_{i\cdot}$ over its rows. 
The matrix $X$ is called the **design matrix**. Likewise, we can gather the target values in the vector $\vvec{y}$.
\begin{align*}
    X&= 
    \begin{pmatrix}
    -- & \bm{\phi}(\vvec{x}_1)^\top &--\\
    &\vdots&\\
    --& \bm{\phi}(\vvec{x}_n)^\top &--
    \end{pmatrix}
    \in\mathbb{R}^{n\times p},&
    \vvec{y}&=
    \begin{pmatrix}
    y_1\\ \vdots\\ y_n
    \end{pmatrix}
    \in\mathbb{R}^n
\end{align*}
We can now specify the linear regression task, using linear regression models and the squared Euclidean distance to measure the fit of the model.

`````{admonition} Task (Linear Regression with Basis Functions)
:class: tip
:name: regr_task
**Given** a dataset of $n$ observations
\begin{equation*}\mathcal{D}=\left\{(\vvec{x}_i,y_i)\vert \vvec{x}_i\in\mathbb{R}^{d}, y_i\in\mathbb{R}, 1\leq i \leq n\right\}\end{equation*}   
**Choose** a basis function $\bm\phi:\mathbb{R}^d\rightarrow \mathbb{R}^p$, and create the design matrix $X\in\mathbb{R}^{n\times p}$, where $X_{i\cdot}=\bm\phi(\vvec{x}_i)^\top$     
**Find** the regression vector $\bm\beta$, solving the following objective
\begin{align*}
    \min_{\bm\beta} \ RSS(\bm\beta) = \lVert \vvec{y}-X\bm\beta\rVert^2 &\ 
    \text{s.t. } \bm\beta\in\mathbb{R}^p.
\end{align*}
**Return** the predictor function $f:\mathbb{R}^d\rightarrow\mathbb{R}$, $f(\vvec{x})=\bm\phi(\vvec{x})^\top\bm\beta$.  
`````

## Convexity of the RSS
The RSS is a convex optimization objective as it is a composition of an affine function and a convex function (the squared $L_2$-norm), which is again convex.
````{prf:theorem}
The function $RSS(\bm\beta)=\lVert \vvec{y}-X\bm{\beta}\rVert^2$ is convex.
````
````{prf:proof}
The squared $L_2$-norm $\lVert\cdot\rVert^2$ is a convex function. 

The composition of the affine function $g(\bm{\beta})=\vvec{y}-X\bm{\beta}$ with the convex function $\Vert\cdot\rVert^2$, given by the $RSS(\bm{\beta})=\lVert g(\bm{\beta})\rVert^2$ is then also convex.
````
As a corollary, the linear regression optimization objective
\begin{align*}
    \min_{\bm{\beta}}&\ RSS(\bm{\beta})& \text{s.t. }\bm{\beta}\in\mathbb{R}^p
\end{align*}
is convex, since the feasible set is the vector space of $\mathbb{R}^p$, which is convex.
So, we have an unconstrained convex optimization problem with a smooth objective function. That means that all stationary points must be minimizers. Let's try to find all stationary points.
## Minimizers of the RSS
We compute the stationary points by setting the gradient to zero. The gradient of the $RSS(\bm{\beta}) = \lVert \vvec{y}-X\bm{\beta}\rVert^2=f(\vvec{g}(\bm{\beta}))$ is computed by the chain rule, as discussed in {ref}`opt_exercises_gradients`. 
:::{math}
:label: eq:minimizers
\begin{align*}
\nabla_{\bm{\beta}} RSS(\bm{\beta})= -2X^\top(\vvec{y}-X\bm{\beta}) =0 
\quad \Leftrightarrow \quad
     X^\top X{\bm{\beta}} = X^\top \vvec{y}
\end{align*}
:::
According to FONC and convexity of the optimization objective, the global minimizers of the regression problem are given by the set of regression parameter vectors satisfying the equation above 
$$\{\bm{\beta}\in\mathbb{R}^p\mid X^\top X\bm{\beta} =X^\top\vvec{y} \}.$$

If the matrix $X^\top X$ is invertible, then there is only one minimizer. In this case we can solve the equation for $\bm{\beta}$ by multiplying with $(X^\top X)^{-1}$
$$\bm{\beta}= (X^\top X)^{-1}X^\top\vvec{y}.$$

However, there also might be _infinitely many_ global minimizers of $RSS(\bm{\beta})$. 

````{tikz}
\begin{tikzpicture}
\begin{axis}[
width=\textwidth,
xlabel=$x_1$, % label x axis
ylabel=$y$, % label y axis
axis lines=left, %set the position of the axes
xmin=0, xmax=7, % set the min and max values of the x-axis
domain=0:6,
ymax=12, % set the min and max values of the y-axis
legend pos=outer north east]
\addplot [only marks, black, mark = *] 
coordinates {
(5,2)
(3,5)
(1,3)
};
\addplot+[magenta,thick,smooth, mark=none]
{x^3-(5/8+9)*x^2+(7/2+23)*x+1/8-15};
\addplot+[blue,thick,smooth, mark=none]
{x^3/2-(5/8+9/2)*x^2+(7/2+23/2)*x+1/8-15/2};
\addplot+[green,thick,smooth, mark=none]
{x^3/4-(5/8+9/4)*x^2+(7/2+23/4)*x+1/8-15/4};
\end{axis}
\end{tikzpicture}
````