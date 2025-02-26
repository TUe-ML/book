#!/usr/bin/env python
# coding: utf-8

# # Kernel k-means

# ## Informal Problem Definition
# $k$-means can Only Identify Convex Clusters

# In[1]:


from sklearn.cluster import KMeans
from sklearn import datasets
import matplotlib.pyplot as plt
epsilon=0.08
D, labels = datasets.make_circles(n_samples=500, factor=.5, noise=epsilon)
kmeans = KMeans(n_clusters=2,n_init=1)
kmeans.fit(D)
plt.scatter(D[:, 0], D[:, 1], c=kmeans.labels_, s=10)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='magenta', s=50, marker = 'D')
plt.axis('equal')
plt.show()


# The cluster-separating boundary between two centroids is always linear. What do we do if we have nonlinearly separated clusters?
# 
# Feature Transformation and Kernel Trick
# Use a feature transformation to map points to a space where clusters are linearly separable: 
# $$\vvec{x}\rightarrow \phi(\vvec{x}).$$
#     
# Problem: Computing $\phi(\vvec{x})$ for every data point might be costly or impossible, $\phi(\vvec{x})$ might be infinite-dimensional (see RBF kernel).
# 
# Solution: We don't need $\phi$, we just need the inner product 
# $\phi(\vvec{x})^\top\phi(\vvec{y})$
# 
# Defining for $D\in\mathbb{R}^{n\times d}$ the row-wise applied feature transformation
# $${\bm\phi}(D)= \begin{pmatrix}--& {\bm\phi}(D_{1\cdot }) & --\\ &\vdots&\\ --&{\bm\phi}(D_{n\cdot})&--\end{pmatrix},$$
# the kernel matrix is given by
# $$ K = {\bm\phi}(D){\bm\phi}(D)^\top\in\mathbb{R}^{n\times n}.$$

# ## Formal Problem Definition
# `````{admonition} Task (Kernel $k$-means)
# :class: tip
# **Given** a data matrix $D\in\mathbb{R}^{n\times d}$, a feature transformation $\phi:\mathbb{R}^{d}\rightarrow \mathbb{R}^p$ mapping into a $p$-dimensional feature space, where $p\in\mathbb{N}\cup\{\infty\}$, and the number of clusters $r$.      
# **Find** clusters indicated by the matrix $Y\in\mathbb{1}^{n\times r}$ which minimize the within cluster scatter in the transformed feature space
# \begin{align}
# \min_{Y} \|{\bm\phi}(D)-YX^\top\|^2 \text{  s.t. } X= {\bm\phi}(D^\top) Y(Y^\top Y)^{-1}, Y\in\mathbb{1}^{n\times r} \label{eq:kernelKMPhi}
# \end{align}
# **Return** the clustering $Y\in\mathbb{1}^{n\times r}$
# `````
# 

# ## Optimization
# If we want to apply the kernel trick, then we need to state the kernel $k$-means objective with respect to the inner product of data points.
# ```{prf:theorem} $k$-means trace objective
# The $k$-means objective in Eq.~\eqref{eq:kmeans} is equivalent to 
# \begin{align}
# &\max_{Y}\ \tr(Z^\top DD^\top Z)&\text{ s.t. } Z= Y(Y^\top Y)^{-1/2}, Y\in\mathbbm{1}^{n\times r} \label{eq:kmeansTr}
# \end{align}
# ```
# Interpretation: Clusters are now defined with respect to the inner product similarity:
# $$sim(i,j) = D_{i\cdot}D_{j\cdot}^\top =\cos(\sphericalangle(D_{i\cdot},D_{j\cdot}))\lVert D_{i\cdot}\rVert\lVert D_{j\cdot}\rVert$$
# Points within one cluster need to be similar:
# $$\tr(Z^\top DD^\top Z)=\sum_{s=1}^r\frac{Y_{\cdot s}^\top DD^\top Y_{\cdot s}}{\lvert Y_{\cdot s}\rvert}
# =\sum_{s=1}^r\frac{1}{\lvert \mathcal{C}_{s}\rvert}\sum_{i,j\in\mathcal{C}_s} D_{i\cdot}D_{j\cdot}^\top$$

# In[2]:


import numpy as np

# Example points
epsilon=0.00
D, labels = datasets.make_blobs(n_samples=50,centers=2, cluster_std=[epsilon + 0.5, epsilon + 0.8],random_state=3)
points = D

# Calculate Euclidean distances between consecutive points
distances = np.sqrt(np.sum(np.diff(points, axis=0)**2, axis=1))

# Normalize distances for line thickness
thickness = D@D.T  # Scale for visualization purposes
#thickness = thickness- np.min(thickness)
thickness = thickness/ np.max(thickness)/2

# Create the scatter plot
plt.scatter(points[:, 0], points[:, 1], zorder=5)

# Plot lines with varying thickness
for i in range(len(points) - 1):
    for j in range(i+1,len(points)):
        if thickness[i,j]>0:
            plt.plot([points[i, 0], points[j, 0]], [points[i, 1], points[j, 1]], linewidth=thickness[i,j], color='orange', zorder=1)

# Customize the plot
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Scatter Plot with Line Thickness Based on Inner Product Similarity')
plt.axis('equal')
plt.show()


# In[3]:


from scipy import sparse
from matplotlib.collections import LineCollection
epsilon=0.00
D, labels = datasets.make_blobs(n_samples=50,centers=2, cluster_std=[epsilon + 0.5, epsilon + 0.8],random_state=3)

A=D@D.T
A = A/ np.max(A)/2
A *= np.tri(*A.shape)
A*= (A>0)
sA = sparse.csr_matrix(A)
segments = np.concatenate([D[sA.nonzero()[0],:].reshape(-1, 1, 2),D[sA.nonzero()[1],:].reshape(-1, 1, 2)],axis=1)
lc = LineCollection(segments, linewidths=np.array(sA[sA.nonzero()])[0],color='orange')
fig,ax = plt.subplots()

ax.add_collection(lc)
ax.scatter(D[:, 0], D[:, 1],  zorder=10)
ax.set_xlim(np.min(D[:,0])-1, np.max(D[:,0])+1)
ax.set_ylim(np.min(D[:,1])-1, np.max(D[:,1])+1)
ax.set_aspect("equal") 


# In[4]:


from scipy import sparse
epsilon=0.05
D, labels = datasets.make_circles(n_samples=50, factor=.5, noise=epsilon)

A=D@D.T
A = A/ np.max(A)/2
A *= np.tri(*A.shape)
A*= (A>0)
sA = sparse.csr_matrix(A)
segments = np.concatenate([D[sA.nonzero()[0],:].reshape(-1, 1, 2),D[sA.nonzero()[1],:].reshape(-1, 1, 2)],axis=1)
lc = LineCollection(segments, linewidths=np.array(sA[sA.nonzero()])[0],color='orange')
fig,ax = plt.subplots()

ax.add_collection(lc)
ax.scatter(D[:, 0], D[:, 1],  zorder=10)
ax.set_xlim(-1.1, 1.1)
ax.set_ylim(-1.1, 1.1)
ax.set_aspect("equal") 


# ```{prf:theorem} Equivalent kernel $k$-means objectives
# Given the kernel matrix $K={\bm\phi}(D){\bm\phi}(D)^\top$,
# the following objectives are equivalent:
# \begin{align}
# &\min_{Y} \|{\bm\phi}(D)-YX^\top\|^2 \text{  s.t. } X= {\bm\phi}(D^\top) Y(Y^\top Y)^{-1}, Y\in\mathbbm{1}^{n\times r} \label{eq:kernelKMPhi}\\
# &\max_{Y} \tr(Z^\top K Z)\qquad\text{ s.t. } Z= Y(Y^\top Y)^{-1/2}, Y\in\mathbbm{1}^{n\times r} \label{eq:kernelKMtr}
# \end{align}
# ```
# Problem: We do not know how to optimize Eq. \eqref{eq:kernelKMtr}, we only know how to optimize Eq. \eqref{eq:kernelKMPhi}, but we do not want to compute $\phi$!        
# 
# Idea: We go the other way round: from the kernel matrix to the inner product.
# ```{prf:theorem} Eigendecomposition of symmetric matrices
# For every symmetric matrix $K=K^\top\in\mathbb{R}^{n\times n}$ there exists an orthogonal matrix $V\in\mathbb{R}^{n\times n}$ and a diagonal matrix $\Lambda=\diag(\lambda_1,\ldots,\lambda_n)$ where $\lvert \lambda_1\rvert\geq \ldots \geq \lvert \lambda_n\rvert$ such that
#     $$K=V\Lambda V^\top$$
# ```
# Every symmetric matrix $K\in\mathbb{R}^{n\times n}$ has a symmetric decomposition $K=A^\top A$ if and only if the eigenvalues of $K$ are nonnegative. 
# 
# This is equivalent to $K$ being positive semi-definite.
# Kernel matrices are positive semi-definite!

# In[ ]:




