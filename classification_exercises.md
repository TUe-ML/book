---
math:
  '\diag': '\mathrm{diag}'
  '\tr': '\mathrm{tr}'
  '\argmin': '\mathrm{arg\\,min}'
  '\argmax': '\mathrm{arg\\,max}'
  '\sign': '\mathrm{sign}'
  '\softmax': '\mathrm{softmax}'
  '\concat': '\mathbin{{+}\mspace{-8mu}{+}}'
  '\vvec': '\mathbf{ #1 }'
  '\bm': '{\boldsymbol #1 }'
---

(class_exercises)=
# Exercises
1. Show that for any p.s.d. and symmetric matrix $Q\in\mathbb{R}^{d\times d},\ Q^\top=Q$ the function $f(\vvec{x})=\vvec{x}^\top Q\vvec{x}$ is convex. Recall that a positive semidefinite matrix (p.s.d.) $Q$ satisfies $\vvec{x}^\top Q\vvec{x}\geq 0$ for all $\vvec{x}\in\mathbb{R}^d.$

````{toggle}
The proof is very similar to the one showing that $\lVert\vvec{x}\rVert^2=\vvec{x}^\top I\vvec{x}$ is convex. Let $\alpha\in[0,1]$, then the definition of a convex function
$$f(\alpha\vvec{x}+(1-\alpha)\vvec{z})\leq \alpha f(\vvec{x})+(1-\alpha)f(\vvec{z})$$
is satisfied for the function above if
$$
    (\alpha\vvec{x}+(1-\alpha)\vvec{z})^\top Q(\alpha\vvec{x}+(1-\alpha)\vvec{z})
    &= \alpha^2\vvec{x}^\top Q\vvec{x} +2\alpha(1-\alpha)\vvec{x}^\top Q\vvec{z} +(1-\alpha)^2\vvec{z}^\top Q\vvec{z}\\
    &\leq \alpha\vvec{x}^\top Q\vvec{x} +(1-\alpha)\vvec{z}^\top Q\vvec{z}.
$$
We subtract the terms on the left side of that equation and get
$$
    \alpha (1-\alpha) \vvec{x}^\top Q\vvec{x} -2\alpha(1-\alpha)\vvec{z}^\top Q\vvec{x}+\alpha(1-\alpha)\vvec{z}^\top Q\vvec{z}&\geq 0\\
    \Leftrightarrow \quad \vvec{x}^\top Q\vvec{x} -2\vvec{z}^\top Q\vvec{x}+\vvec{z}^\top Q\vvec{z}&\geq 0\\
    \Leftrightarrow \quad (\vvec{x}-\vvec{z})^\top Q(\vvec{x}-\vvec{z})&\geq 0.
$$
The equation above is always true, because $Q$ is p.s.d.
````
