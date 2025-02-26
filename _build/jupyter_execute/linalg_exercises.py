#!/usr/bin/env python
# coding: utf-8

# # Exercises
# 
# ## Linear Algebra Trivia
# 
# 1. $A\in\mathbb{R}^{n\times r}$, $B\in\mathbb{R}^{m\times r}$, which product is well-defined?    
#     A.  $BA$    
#     B.  $A^\top B$    
#     C.  $AB^\top$    
#     ```{toggle}
#     The correct solution is C. $AB^\top$, the connecting dimension is $r$. 
#     ```
# 2. $A\in\mathbb{R}^{n\times r}$, $B\in\mathbb{R}^{m\times r}$, what is equal to $(AB^\top)^\top ?$     
#     A. $A^\top B$     
#     B. $B^\top A^\top$     
#     C. $BA^\top$    
#     ```{toggle}
#     The correct solution is C. $BA^\top$, we have $(AB^\top)^\top = {B^\top}^\top A^\top = BA^\top$.
#     ```
# 3. What is the matrix product computed by $C_{ji}=\sum_{s=1}^rA_{is}B_{js} ?$     
#     A. $C=AB^\top$     
#     B. $C=B^\top A$     
#     C. $C=BA^\top$ 
#     ```{toggle}
#     The correct solution is C. $BA^\top$, we have 
#     $C_{ji}=\sum_{s=1}^rA_{is}B_{js} = \sum_{s=1}^rB_{js}A_{is} = B_{j\cdot}A_{i\cdot}^\top$, hence $C=BA^\top$.
#     ```
# 4. $A,B\in\mathbb{R}^{n\times n}$ have an inverse $A^{-1},B^{-1}$, what is generally **not** equal to $AA^{-1}B$?    
#     A. $A^{-1}BA$     
#     B. $B$       
#     C. $BB^{-1}B$ 
#     ```{toggle}
#     The correct answer is A. $A^{-1}BA$, since the matrix product between the matrices $A^{-1}$ and $BA$ is generally not commutative. Answers B. and C. are correct because
#     $$AA^{-1}B = IB =B = BB^{-1}B.$$
#     ```
# 5. Let $v,w\in\mathbb{R}^{d}$, $\alpha\in\mathbb{R}$, then $\lVert\alpha v + w\rVert\leq$    
#     A. $\alpha\lVert v+w\rVert$     
#     B. $\lvert\alpha\rvert\lVert v\rVert+\lVert w\rVert$     
#     C. $\alpha\lVert v\rVert+\lVert w\rVert$
#     ```{toggle}
#     The correct answer is B. $\lvert\alpha\rvert\lVert v\rVert+\lVert w\rVert$, because 
#     \begin{align*}
#     \lVert\alpha v + w\rVert&\leq \lVert\alpha v\rVert + \lVert w\rVert  &\text{(triangle inequality)}\\
#     & = \lvert\alpha\rvert\lVert v\rVert + \lVert w\rVert  &\text{(homogenity)}.
#     \end{align*}
#     ```
# 6. Let $A,B\in\mathbb{R}^{n\times r}$, $\alpha\in\mathbb{R}$, then $\lVert A\rVert\leq$
#     A. $\lVert A-B\rVert + \lVert B\rVert$
#     B. $\alpha\lVert\frac{1}{\alpha}A\rVert$
#     C. $\lVert A\rVert^2$
#     ```{toggle}
#     The correct answer is A. $\lVert A-B\rVert + \lVert B\rVert$, because
#     \begin{align*}
#     \lVert A\rVert\leq \lVert A-B+B\rVert \leq \lVert A -B \rVert +\lVert B\rVert,
#     \end{align*}
#     where the last inequality derives from the triangle inequality. Answer B. is nott correct because the inequalitty does not hold for negative $\alpha$ and C. is not correct for matrices $A$ having small entries (for example take the $1\times 1$ matrix A=(0.1), then $\lVert A\rVert = 0.1 > 0.1^2$).
#     ```
# 7. Let $A,B\in\mathbb{R}^{n\times n}$, $A$ is orthogonal, what is **not** equal to $\tr(ABA^\top)$?     
#     A. $\tr(A^\top BA)$    
#     B. $\tr(B)$     
#     C. $\tr(ABA)$      
#     ```{toggle}
#     The correct answer is C. $\tr(ABA)$. The other answers are not correct because the cycling property of the trace yields $\tr(ABA^\top)  = \tr(BA^\top A)$, and
#     \begin{align*}
#     \tr(ABA^\top) & = \tr(BA^\top A) = \tr(BI) =\tr(B) & \text{(orthogonality of $A$)}\\
#     & = \tr(IB) = \tr(AA^\top B) = \tr(A^\top B A). &\text{(cycling property and orthogonality)}
#     \end{align*}
#     ```
# ## Exercises
# 1. Compute the matrix product $AB$ inner-product-wise and outer-product-wise
#     \begin{align*}
#     A=\begin{pmatrix} 1 & 2 & 0 \\
#     0 & 2& 4\end{pmatrix},\quad B=\begin{pmatrix} 0 & 2 \\
#     3 &1 \\
#     1 & 2\end{pmatrix}.
#     \end{align*}
#     ```{toggle}
#     Computing the matrix product _inner product wise_:
#     \begin{align*}
#     AB = 
#     \begin{pmatrix}
#         1\cdot 0 + 2\cdot3+0\cdot 1 & 1\cdot2+2\cdot1 + 0\cdot 2\\
#         0\cdot 0 + 2\cdot3+4\cdot 1 & 0\cdot2+2\cdot1 + 4\cdot 2
#     \end{pmatrix}
#     =
#     \begin{pmatrix}
#         6 & 4\\
#         10 & 10
#     \end{pmatrix}
#     \end{align*}
#     Computing the matrix product _outer product wise_:
#     \begin{align*}
#     AB &= 
#     \begin{pmatrix}1 \\0\end{pmatrix}\begin{pmatrix}0&2\end{pmatrix} +
#     \begin{pmatrix}2 \\2\end{pmatrix}\begin{pmatrix}3&1\end{pmatrix} +
#     \begin{pmatrix}0 \\4\end{pmatrix}\begin{pmatrix}1&2\end{pmatrix} \\
#     &=
#     \begin{pmatrix}1\cdot 0 &  1\cdot 2\\0\cdot 0 & 0\cdot 2\end{pmatrix} +
#     \begin{pmatrix}2\cdot 3 & 2\cdot 1 \\2\cdot 3 & 2\cdot 1\end{pmatrix} +
#     \begin{pmatrix}0\cdot 1&0\cdot2 \\4\cdot1&4\cdot2\end{pmatrix}\\
#     &=
#     \begin{pmatrix}0 &  2\\0 & 0\end{pmatrix} +
#     \begin{pmatrix}6 & 2 \\6 & 2\end{pmatrix} +
#     \begin{pmatrix}0&0 \\4&8\end{pmatrix}\\
#     &= \begin{pmatrix}6&4\\ 10 &10\end{pmatrix}
#     \end{align*}
#     ```
# 2. You have observations of $5$ symptoms of a disease for three patients represented in the binary matrix 
#     $$ A = \begin{pmatrix}
#   1 & 0 & 1 & 1 & 0\\
#   1 & 1 & 0 & 0 & 0\\
#   0 & 1 & 0 & 0 & 1\end{pmatrix}$$
#   Compute the matrix $AA^\top$ and $A^\top A$ and interpret the result with regard to the scenario.
#     ```{toggle}
#     The matrix represents a data table of the following format, where $\mathtt{S}_i$ denotes the feature of sympton $i$ and $\mathtt{P}$ stands for patient ID:
#     
#     |$\mathtt{P}$| $\mathtt{S}_1$ | $\mathtt{S}_2$ | $\mathtt{S}_3$ |  $\mathtt{S}_4$ |  $\mathtt{S}_5$ |
#     |------------|----------------|----------------|----------------|-----------------|-----------------|    
#     |1           | 1              | 0              | 1              | 1               |               0 | 
#     |2|1 | 1 | 0  |0 | 0 |  
#     |3|0 | 1 | 0  |0 | 1 | 
#     
#     The matrix products 
#     \begin{align*}
#         AA^\top &= 
#         \begin{pmatrix}
#             3 & 1 & 0\\
#             1 & 2 & 1\\
#             0 & 1 & 2
#         \end{pmatrix},
#         & A^\top A &=
#         \begin{pmatrix}
#             2 & 1 & 1 & 1 & 0\\
#             1 & 2 & 0 & 0 & 1\\
#             1 & 0 & 1 & 1 & 0\\
#             1 & 0 & 1 & 1 & 0\\
#             0 & 1 & 0 & 0 & 1
#         \end{pmatrix}
#     \end{align*}
#     contrast either the features or the patients. That is, the representation of the matrix product in the table format would look as follows:
#     
#     |$AA^\top$ | $\mathtt{P}_1$ | $\mathtt{P}_2$ | $\mathtt{P}_3$ |
#     |----------|----------------|----------------|----------------|
#     |$\mathtt{P}_1$ |3 | 1 | 0  | 
#     |$\mathtt{P}_1$ |1 | 2 | 1  | 
#     |$\mathtt{P}_1$ |0 | 1 | 2  |
#     
#     |$A^\top A$ | $\mathtt{S}_1$ | $\mathtt{S}_2$ | $\mathtt{S}_3$ |  $\mathtt{S}_4$ |  $\mathtt{S}_5$ | 
#     |-----------|----------------|----------------|----------------|-----------------|-----------------|
#     |$\mathtt{S}_1$| 2 | 1 | 1  |1 | 0 |
#     |$\mathtt{S}_2$| 1 | 2 | 0  |0 | 1 |
#     |$\mathtt{S}_3$| 1 | 0 | 1  |1 | 0 |
#     |$\mathtt{S}_4$| 1 | 0 | 1  |1 | 0 |
#     |$\mathtt{S}_5$| 0 | 1 | 0  |0 | 1 |
#     
#     The table of $AA^\top$ denotes in entry $jl$ the number of symptoms patient $j$ and patient $l$ have in common. The table of $A^\top A$ denotes in entry $ik$ the number of patients which exibit symptoms $i$ and $k$.
#     ```
# 3.  Find a matrix/vector notation to compute the vector of average feature values for a matrix $A\in\mathbb{R}^{n\times d}$, representing $n$ observations of $d$ features. Make an example for your computation.
#     ```{toggle}
#      The vector representing the average for every feature value is computed by $\bm{\mu} = \frac{1}{n} A^\top \mathbf{1}$ where $\mathbf{1}$ is the $n$-dimensional one-vector, having all values equal to one. This is the case, because we have according to the definition of matrix multiplications by the row-times-column column scheme:
#      \begin{align*}
#          \bm{\mu}^\top = \frac{1}{n}\mathbf{1}^\top A = \frac1n \begin{pmatrix}\mathbf{1}^\top A_{\cdot 1} &\ldots & \mathbf{1}^\top A_{\cdot d}\end{pmatrix},
#      \end{align*}
#      that is, for $1\leq j \leq n$ we have that the $i$-th entry of $\bm \mu$ is given as
#      \begin{align*}
#          \bm\mu_{i} = \frac1n \mathbf{1}^\top A_{\cdot i} = \frac1n \sum_{j=1}^n 1\cdot A_{ji}= \frac1n \sum_{j=1}^n A_{ji},
#      \end{align*}
#      which is equal to the average value for feature $i$.
#     ```
# 4. Every system of linear equations can be written as a matrix equation $A\vvec{x}=\vvec{y}$. Given the following system of linear equations, what would be the matrix $A$ and vector $\vvec{y}$ such that the system of linear equations is equivalent to solving $A\vvec{x}=\vvec{y}$? 
#     \begin{align}
#         2x_1 &+& 3x_2 && &=&4\\
#         x_1  &-& 2x_2 &+& x_3 &=& 3\\
#         -x_1 &+& 2x_2 &+& 3x_3 &=& 1
#     \end{align}.
#     Can you solve the system of linear equations by using the inverse of $A$ (`np.linalg.inv(A)`)?  
#     ```{toggle}
#     You can easily verify that the equation $A\vvec{x}=\vvec{y}$ is equivalent to the above system of equations when using
#     \begin{align*}
#         A &= 
#         \begin{pmatrix}
#             2 & 3 & 0\\
#             1 & -2 & 1\\
#             -1 & 2 & 3
#         \end{pmatrix},
#         &y &= 
#         \begin{pmatrix}
#             4 \\
#             3 \\
#             1 
#         \end{pmatrix}.
#     \end{align*}
#     We can solve the system of linear equations by multiplying with $A^{-1}$ from left:
#     \begin{align*}
#         A\vvec{x}=\vvec{y} \Leftrightarrow A^{-1}A\vvec{x}=A^{-1}\vvec{y} \Leftrightarrow \vvec{x}=A^{-1}\vvec{y}
#     \end{align*}
#     That is, the solution is given by $\vvec{x}=A^{-1}\vvec{y}$. This is of course only possible if $A$ is invertible. In Python, we can solve the system of linear equations as follows:
#     
#         import numpy as np
#         A = np.array([[2,3,0],[1,-2,1],[-1,2,3]])
#         y = np.array([4,3,1])
#         print("x=",np.linalg.inv(A)@y) 
#     Alternatively, we can use the following function to solve a system of linear equations:
#     
#         print("x=",np.linalg.solve(A,y))
#     ```
# 4. Show that $\lVert A - B \rVert^2 = -2\tr(AB^\top) + 2n $ for orthogonal matrices $A,B\in\mathbb{R}^{n\times n}$.
#     ```{toggle}
#     _Proof:_ Let $A,B\in\mathbb{R}^{n\times n}$ be orthogonal matrices. Orthogonal matrices satisfy the property $AA^\top =A^\top A=I$ and $BB^\top =B^\top B=I$. Thus, we have
#     \begin{align*}
#         \lVert A-B\rVert^2 &= \lVert A\rVert^2 -2\tr(AB^\top) + \lVert B\rVert^2 &\text{(binomial formula for matrix norms)}\\
#         &= \tr(A^\top A) -2\tr(AB^\top) +\tr(B^\top B) &\text{(definition of elementwise matrix $L_2$-norm)}\\
#         &= \tr(I) -2\tr(AB^\top) +\tr(I) &\text{(orthogonality of $A$ and $B$)}\\
#         &= -2\tr(AB^\top) +2n, 
#     \end{align*}
#     because $\tr(I) = \underbrace{1 +\ldots +1}_{n \text{ times}} =n$. This concludes the proof.
#     ```
# 5. Show that the following norms are orthogonal invariant
#     * the vector $L_2$-norm
#     * the Frobenius norm (matrix $L_2$-norm)
#     * the operator norm
#     ```{toggle}
#     A norm is orthogonal invariant if multiplying the argument with an orthogonal matrix from the left does not change the value of the norm. Let $A$ be a $(n\times n)$ orthogonal matrix. For the $L_2$-norm of a vector $\vvec{v}\in\mathbb{R}^n$ we have:
#     \begin{align*}
#         \lVert A\vvec{v} \rVert^2 &= (A\vvec{v})^\top A\vvec{v} & \text{(Definition)}\\ 
#         &= \vvec{v}^\top A^\top A\vvec{v}\\ 
#         &= \vvec{v}^\top \vvec{v} &(A^\top A =I)\\
#         &= \lVert\vvec{v}\rVert^2
#     \end{align*}
#     For the $L_2$-norm of a $(n\times d)$ matrix $AD$ we have:
#     \begin{align*}
#         \lVert AD \rVert^2 &= \tr((AD)^\top AD) &\text{(Definition)}\\ 
#         &= \tr(D^\top A^\top AD) &\\ 
#         &= \tr(D^\top D) & (A^\top A=I)\\
#         &= \lVert D\rVert^2 &
#     \end{align*}
#     For the operator norm of the matrix $AD$, we have:
#     \begin{align*}
#         \lVert AD \rVert_{op} &= \max_{\vvec{x}:\lVert\vvec{x}\rVert=1}\lVert AD\vvec{x}\rVert &\text{(Definition)}\\ 
#         &= \max_{\vvec{x}:\lVert\vvec{x}\rVert=1}\lVert D\vvec{x}\rVert & \text{(orthogonal invariance of $L_2$ vector norm)}\\ 
#         &= \lVert D\rVert_{op} &
#     \end{align*}
#     ```
