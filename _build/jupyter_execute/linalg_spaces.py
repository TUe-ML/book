#!/usr/bin/env python
# coding: utf-8

# ## Vector Spaces
# 
# ````{prf:definition}
# A **vector space** over the real numbers is a set of vectors $\mathcal{V}$ with two operations $+$ and $\cdot$ such that the following properties hold: 
# * Addition: for $\vvec{v},\vvec{w}$ we have $\vvec{v}+\vvec{w}\in\mathcal{V}$. The set of vectors with the addition $(\mathcal{V},+)$ is an abelian group.
# * Scalar multiplication: for $\alpha\in\mathbb{R}$ and $\vvec{v}\in\mathcal{V}$, we have $\alpha\vvec{v}\in\mathcal{V}$ such that the following properties hold:
#      - $\alpha(\beta\vvec{v}) = (\alpha\beta) \vvec{v}$ for $\alpha,\beta\in\mathbb{R}$ and $\vvec{v}\in\mathcal{V}$
#      - $1\vvec{v}=\vvec{v}$ for $\vvec{v}\in\mathcal{V}$
# * Distributivity: the following properties hold:
#      - $(\alpha + \beta)\vvec{v} = \alpha\vvec{v} +\beta \vvec{v}$ for $\alpha,\beta\in\mathbb{R}$ and $\vvec{v}\in\mathcal{V}$
#      - $\alpha (\vvec{v}+\vvec{w})=\alpha \vvec{v}+\alpha\vvec{w}$ for $\alpha\in\mathbb{R}$ and $\vvec{v},\vvec{w}\in\mathcal{V}$
# ````
# 
# A vector space is a structure where you can do most operations you know from real numbers, but not all. Let $\alpha\in\mathbb{R}, \vvec{v},\vvec{w}\in\mathcal{V}$. The following operations are well-defined:
# * $\vvec{v}/\alpha = \frac1\alpha \vvec{v}$ for $\alpha\neq 0$
# * $\vvec{v}-\vvec{w}$
# What you can not do:
# * $\vvec{v}\cdot \vvec{w}$
# * $\alpha/\vvec{v}$
# 
# ````{prf:example} The vector space $\mathbb{R}^2$
# The elements of the vector space $\mathbb{R}^d$ are $d$-dimensional vectors
# $$\vvec{v} = \begin{pmatrix}v_1\\\vdots\\v_d\end{pmatrix},\quad v_i\in\mathbb{R} \text{ for } 1\leq i \leq d. $$
#     For vectors, the addition between vectors and the scalar multiplication are defined for $\vvec{v},\vvec{w}\in\mathbb{R}^d$ and $\alpha\in\mathbb{R}$ as
# \begin{align*}
#     \vvec{v}+\vvec{w} = \begin{pmatrix}v_1+w_1\\\vdots\\v_d+w_d\end{pmatrix},
#     \alpha\vvec{v} = \begin{pmatrix}\alpha v_1\\\vdots\\\alpha v_d\end{pmatrix} 
# \end{align*}
# ````
# ````{prf:example} The geometry of $\mathbb{R}^2$
# :label: ex_vector_space
# ```{tikz}
# \begin{tikzpicture}
# \begin{axis}[
#     	%height=\textwidth,
#     	width=.5\textwidth,
#     	xmin=-0.1, xmax=4, % set the min and max values of the x-axis
#     	axis lines=center, %set the position of the axes
#     	ymin=-0.1, ymax=4,
#     	xlabel=$x_1$, % label x axis
#         ylabel=$x_2$, % label y axis
#         %xtick=\empty, ytick=\empty,
#         scale only axis=true,
# ]
# \draw [->, ultra thick,  green] (axis cs:0,0) -- (axis cs:1,3) node[left]{$2\mathbf{v}$};
# \draw [->, ultra thick,  magenta] (axis cs:0,0) -- (axis cs:0.5,1.5) node[left]{$\mathbf{v}$};
# \draw [->, ultra thick,  yellow] (axis cs:0,0) -- (axis cs:2,0.5) node[below]{$\mathbf{w}$};
# \draw [->, ultra thick,  blue] (axis cs:0,0) -- (axis cs:2.5,2) node[right]{$\mathbf{v}+\mathbf{w}$};
# \draw [-, ultra thick, dashed, black] (axis cs:0.5,1.5) -- (axis cs:2.5,2); %node[right]{$\mathbf{v}+\mathbf{w}$};
# \draw [-, ultra thick, dashed, black] (axis cs:2,0.5) -- (axis cs:2.5,2); 
# \end{axis}
# \end{tikzpicture}
# ```
# \begin{align*}
#      \vvec{v} &= \begin{pmatrix}0.5\\1.5\end{pmatrix}
#      &\vvec{w} &= \begin{pmatrix}2\\0.5\end{pmatrix}
#      &\vvec{v}+\vvec{w} &= \begin{pmatrix}2.5\\2\end{pmatrix}
#      &2\vvec{v} &= \begin{pmatrix}1\\3\end{pmatrix}
#  \end{align*}
# ````
# Are there other important vector spaces next to $\mathbb{R}^d$?
# Yes, the vector space of matrices $\mathbb{R}^{n\times d}.$
# Why are matrices important?
# Because data is represented as a matrix. A data table of $n$ observations of $d$ features is represented by a $(n\times d)$ matrix.
# 
# |ID | $\mathtt{F}_1$ | $\mathtt{F}_2$ | $\mathtt{F}_3$ | $\ldots$ | $\mathtt{F}_d$|
# |---|----------------|----------------|----------------|----------|---------------|
# |1  |5.1 | 3.5 | 1.4 | $\ldots$ | 0.2 |
# |2 | 6.4 | 3.5 | 4.5 | $\ldots$ | 1.2 |
# |$\vdots$ | $\vdots$ | $\vdots$ | $\vdots$ | $\vdots$ | $\vdots$|
# |$n$|5.9 | 3.0 | 5.0 | $\ldots$ | 1.8 |
# 
# An $(n\times d)$ matrix concatenates $n$ $d$-dimensional vectors column-wise ($A_{\cdot j}$ denotes the column-vector $j$ of $A$)
# \begin{align*}
#     A= \begin{pmatrix}
#         \vert & & \vert\\
#        A_{\cdot 1}&\ldots & A_{\cdot d}\\
# \vert & & \vert
# \end{pmatrix}
# =\begin{pmatrix}
#         A_{11 } & \ldots& A_{1d}\\
#        \vdots& & \vdots\\
# A_{n1} &\ldots & A_{nd} 
# \end{pmatrix}
# \end{align*}
# Simultaneously, we can see a matrix as concatenation of $d$ row-vectors ($A_{i\cdot}$):
# \begin{align*}
#     A= \begin{pmatrix}
#         - & A_{1\cdot} & -\\
#        &\vdots & \\
# - & A_{n\cdot} & -
# \end{pmatrix}
# =\begin{pmatrix}
#         A_{11 } & \ldots& A_{1d}\\
#        \vdots& & \vdots\\
# A_{n1} &\ldots & A_{nd} 
# \end{pmatrix}
# \end{align*}
# This notation is actuallly quite close to how we select rows and columns of matrices in Python. For example, consider the following matrix:

# In[1]:


import numpy as np
A=np.array([[1,2,3],[4,5,6],[7,8,9]])
A


# In mathematical notation, the first column would be denoted as $A_{\cdot 1}$. In Python we write

# In[2]:


A[:,0]


# Note, that the indices differ conventionally: in mathematical notation we start with index 1 while the array indices in Python start with 0. But apart from that, both notations are quite close. The dot in the mathematical notation $A_{\cdot 1}$ and the colon in the Python syntax `A[:,0]` indicates that we choose all row-indices. Likewise, we can select the first row $A_{\cdot 1}$ in Python as follows:  

# In[3]:


A[0,:]


# ````{prf:example} The Vector Space $\mathbb{R}^{n\times d}$
# The elements of the vector space $\mathbb{R}^{n\times d}$ are $(n\times d)$-dimensional matrices.
#     
# The addition between matrices and the scalar multiplication are defined for $A,B\in\mathbb{R}^{n\times d}$ and $\alpha\in\mathbb{R}$ as
# \begin{align*}
#     A+B &= \begin{pmatrix}
#         A_{11 } + B_{11} & \ldots& A_{1d}+B_{1d}\\
#        \vdots& & \vdots\\
# A_{n1}+B_{n1} &\ldots & A_{nd}+B_{nd} 
# \end{pmatrix}\\
#     \alpha A &= \begin{pmatrix}
#         \alpha A_{11 } & \ldots& \alpha A_{1d}\\
#        \vdots& & \vdots\\
# \alpha A_{n1} &\ldots & \alpha A_{nd} 
# \end{pmatrix}
# \end{align*}
# ````

# ### Matrix Operations

# The **transpose** of a matrix changes row-vectors into column vectors and vice versa:
# \begin{align*}
#     A&= \begin{pmatrix}
#         \vert & & \vert\\
#        A_{\cdot 1}&\ldots & A_{\cdot d}\\
# \vert & & \vert
# \end{pmatrix}
# &=\begin{pmatrix}
#         A_{11 } & \ldots& A_{1d}\\
#        \vdots& & \vdots\\
# A_{n1} &\ldots & A_{nd} 
# \end{pmatrix}\in\mathbb{R}^{n\times d}\\
# A^\top&= \begin{pmatrix}
#         - & A_{\cdot 1}^\top & -\\
#        &\vdots & \\
#  -&A_{\cdot d}^\top &- 
# \end{pmatrix}
# &=\begin{pmatrix}
#         A_{11 } & \ldots& A_{n1}\\
#        \vdots& & \vdots\\
# A_{1d} &\ldots & A_{nd} 
# \end{pmatrix}\in\mathbb{R}^{d\times n}
# \end{align*}
# The transpose of a $d$-dimensional vector has an interpretation as transpose of a $(d\times 1)$ matrix:
# \begin{align*}
#     v&= \begin{pmatrix}
#         v_1\\
#         \vdots\\
#         v_d\\
# \end{pmatrix}
# &\in\mathbb{R}^{d\times 1}\\
# v^\top&= \begin{pmatrix}
#         v_1 & \ldots & v_d
# \end{pmatrix}&\in\mathbb{R}^{1\times d}
# \end{align*}

# The transpose of the transpose returns the original matrix.
# For any matrix $A\in\mathbb{R}^{n\times d}$ we have \alert{${A^\top}^\top = A$

# In[4]:


A= np.array([[1 , 2 , 3],[4 , 5 , 6]])
A


# In[5]:


A.T


# In[6]:


A.T.T


# A **symmetric matrix** is a matrix $A\in\mathbb{R}^{n\times n}$ such that $A^\top = A$.

# In[7]:


A= np.array([[1 , 2 , 3],[2 , 4 , 5],[3, 5, 6]])
A


# In[8]:


A.T


# A **diagonal matrix** is a symmetric matrix having only nonzero elements on the diagonal:
# $$\diag(a_1,\ldots, a_n) = 
#     \begin{pmatrix}a_1 & 0 & \ldots & 0 \\
#     0 & a_2 &\ldots & 0\\
#       &     & \ddots &\\
#     0 & 0   & \ldots & a_n\end{pmatrix}$$

# In[9]:


np.diag([1,2,3])


# Okay, great, we can add, scale and transpose matrices/data. Isn't that kinda lame?
# Yah, it gets interesting with the matrix product.

# ### Vector and Matrix Products
# The **inner product** of two vectors $\vvec{v},\vvec{w}\in\mathbb{R}^d$ returns a scalar:
# \begin{align*}
# \vvec{v}^\top\vvec{w} = \begin{pmatrix}v_1&\ldots & v_d\end{pmatrix}\begin{pmatrix}w_1\\\vdots\\w_d\end{pmatrix}=\sum_{i=1}^dv_iw_i 
# \end{align*}
# The **outer product** of two vectors $\vvec{v}\in\mathbb{R}^d$ and $\vvec{w}\in\mathbb{R}^n$ returns a ($d\times n$) matrix:
# \begin{align*}
#     \vvec{v}\vvec{w}^\top = \begin{pmatrix}v_1\\ \vdots \\ v_d\end{pmatrix}\begin{pmatrix}w_1 & \ldots &w_n\end{pmatrix} = 
#     \begin{pmatrix} v_1\vvec{w}^\top\\
#     \vdots\\
#     v_d \vvec{w}^\top
#     \end{pmatrix}
#     = \begin{pmatrix}
#      v_1w_1 &\ldots & v_1w_n\\
#      \vdots &       & \vdots\\
#      v_dw_1    &\ldots & v_dw_n     
#     \end{pmatrix}
# \end{align*}

# In[10]:


v = np.array([1,2,3])
w = np.array([4,5,6])


# In[11]:


np.inner(v,w)


# In[12]:


np.outer(v,w)


# Given $A\in \mathbb{R}^{n\times r}$ and $B\in\mathbb{R}^{r\times d}$, the matrix product $C=AB\in\mathbb{R}^{n\times d}$ is defined as
# $$C= \begin{pmatrix} A_{1\cdot}B_{\cdot 1}&\ldots & A_{1\cdot}B_{\cdot d}\\
# \vdots & & \vdots\\ A_{n\cdot}B_{\cdot 1}&\ldots & A_{n\cdot}B_{\cdot d} \end{pmatrix} = \begin{pmatrix}-&A_{1\cdot}&-\\&\vdots&\\-&A_{n\cdot}&-\end{pmatrix} \begin{pmatrix} \vert& & \vert\\ B_{\cdot 1}&\ldots&B_{\cdot d}\\\vert& & \vert \end{pmatrix}$$
# 
# Every element $C_{ji}$ is computed by the inner product of row $j$ and column $i$ (_row-times-column_)
# $$C_{ji}=A_{j\cdot}B_{\cdot i} = \sum_{s=1}^r A_{js}B_{si}$$

# In[13]:


np.set_printoptions(precision=1)


# In[14]:


A = np.random.rand(2,3)
B = np.random.rand(3,5)


# In[15]:


A@B


# In[16]:


(A@B).shape


# Given $A\in \mathbb{R}^{n\times r}$ and $B\in\mathbb{R}^{r\times d}$, we can also state the product $C=AB$ in terms of the outer product:
# $$C=\sum_{s=1}^r \begin{pmatrix} A_{1 s}B_{s1} &\ldots & A_{1 s}B_{sd}\\ \vdots & & \vdots\\ A_{n s}B_{s1} &\ldots & A_{n s}B_{sd} \end{pmatrix} = \begin{pmatrix} \vert & & \vert\\ A_{\cdot 1}&\ldots & A_{\cdot r}\\ \vert & & \vert \end{pmatrix} \begin{pmatrix} - &B_{1\cdot } & -\\  & \vdots & \\ -& B_{r\cdot} & -\end{pmatrix} $$
# The matrix product is the sum of outer products of corresponding column- and row-vectors (_column-times-row_):
# $$ C =\sum_{s=1}^r \begin{pmatrix} \vert\\ A_{\cdot s}\\ \vert  \end{pmatrix} \begin{pmatrix} - & B_{s\cdot } & -
# \end{pmatrix}=\sum_{s=1}^r A_{\cdot s}B_{s \cdot}$$

# The **identity matrix** $I$ is a diagonal matrix having only ones on the diagonal:
# $$I_3 = \begin{pmatrix} 1 & 0 & 0\\ 0 & 1 & 0\\ 0& 0& 1 \end{pmatrix}$$
# 
# Given $A\in\mathbb{R}^{n\times d}$, and $I_n$ the $(n\times n)$ identity matrix and $I_d$ the $(d\times d)$ identity matrix, then we have
# $$I_n A = A = AI_d $$

# In[17]:


I = np.eye(3)
I


# In[18]:


A@I


# In[19]:


A


# We have for $A\in \mathbb{R}^{n\times r}$,  $B\in\mathbb{R}^{r\times d}$ and $C=AB$
# \begin{align*}
#     C^\top&= \begin{pmatrix}
#     A_{1\cdot}B_{\cdot 1}&\ldots & A_{1\cdot}B_{\cdot d}\\
#     \vdots & & \vdots\\
#     A_{n\cdot}B_{\cdot 1}&\ldots & A_{n\cdot}B_{\cdot d}
#     \end{pmatrix}^\top = 
#     \begin{pmatrix}
#     A_{1\cdot}B_{\cdot 1}&\ldots & A_{n\cdot}B_{\cdot 1}\\
#     \vdots & & \vdots\\
#     A_{1\cdot}B_{\cdot d}&\ldots & A_{n\cdot}B_{\cdot d}
#     \end{pmatrix}
#     \\
#     &= \begin{pmatrix}
#     B_{\cdot 1}^\top A_{1\cdot}^\top&\ldots & B_{\cdot 1}^\top A_{n\cdot}^\top\\
#     \vdots & & \vdots\\
#     B_{\cdot d}^\top A_{1\cdot}^\top &\ldots & B_{\cdot d}^\top A_{n\cdot}^\top 
#     \end{pmatrix}= B^\top A^\top
# \end{align*}
# 
# If we can multiply matrices, can we then also divide by them?
# Just sometimes, if the matrix has an _inverse_.

# The **inverse matrix** to a matrix $A\in\mathbb{R}^{n\times n}$ is a matrix $A^{-1}$ satisfying
# $$AA^{-1} = A^{-1}A = I$$
# Diagonal matrices with nonzero elements on the diagonal have an inverse:
# $$\begin{pmatrix} 1& 0 & 0\\
# 0 & 2 & 0\\
# 0 & 0 & 3\end{pmatrix}
# \begin{pmatrix}1& 0 & 0\\
# 0 & \frac12 & 0\\
# 0 & 0 & \frac13\end{pmatrix} = I$$

# In[20]:


A = np.diag([1,2,3])
np.linalg.inv(A)


# Okay, but why is this now interesting?
# 
# Because matrix multiplication is computable fast, and almost every data operation can be written as a matrix operation.

# In[21]:


def matrix_mult(A,B):
    C = np.zeros((A.shape[0],B.shape[1]))
    for i in range(0,A.shape[0]):
        for j in range(0,B.shape[1]):
            for s in range(0,A.shape[1]):
                C[i,j]+= A[i,s]*B[s,j]
    return C
A = np.random.rand(200,100)
B = np.random.rand(100,300)
import time
startTime = time.time()
matrix_mult(A,B)
executionTime = (time.time() - startTime)
print('Execution time of our naive implementation in seconds: ' + str(executionTime))
startTime = time.time()
A@B
executionTime = (time.time() - startTime)
print('Execution time of the numpy implementation in seconds: ' + str(executionTime))


# In[ ]:




