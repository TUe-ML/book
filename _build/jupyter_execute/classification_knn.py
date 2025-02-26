#!/usr/bin/env python
# coding: utf-8

# ## K-Nearest Neighbor
# 
# In this section, we introduce first the Nearest Neighbor (NN) classifier to better motivate the K-Nearest Neighbor (KNN) classifier. In particular, we illustrate how the KNN overcomes some of the NN limitations by means of a particular classification task.
# 
# ### Nearest Neighbor (NN) classifier
# For a given observed data item $ \check{\bf x} $, the predicted class value $ \hat{y} $ is the label $ y_{c} $ associated with the closest data item $ {\bf x}_{c} $ -- the nearest neighbor -- within the training dataset $ {\cal D} $. In precise mathematical terms, for an arbitrary data item $ {\bf x} \in {\cal X} $, the NN classifier can be written as
# \begin{eqnarray}
# h_{NN}({\bf x}) &=& y_{c} \nonumber \\
# \left( {\bf x}_{c},  y_{c} \right) &=& \argmin_{ \left( {\bf x}',  y' \right) \in {\cal D}} || {\bf x}' - {\bf x} ||, \nonumber
# \end{eqnarray}
# where the operator $ ||\cdot|| $ stands for a suitable distance metric, e.g.
# * the Manhattan distance ($ L_{1} $ norm),
# * the Euclidean distance ($ L_{2} $ norm),
# * the Hamming distance for binary feature vectors or
# * any user-defined distance metric satisfying the triangular inequality $ || {\bf x} + {\bf x}' || \leq || {\bf x} || + || {\bf x}' || $, $ \forall {\bf x}, {\bf x}' \in {\cal X} $.
# 
# ````{margin}
# ```{note}
# **Pros:** The NN classifier is non-parametric, i.e. there are no additional parameters to learn besides the training dataset $ {\cal D} $ itself.
# ```
# 
# ```{note}
# **Cons:** We need to store the full data set $ {\cal D} $. The prediction has cost $ {\cal O}(N) $. The NN classifier is also very sensitive to outliers / noise within the training dataset $ {\cal D} $.
# ```
# ````
# 
# ```{figure} /images/classification/nn_classifier.svg
# ---
# height: 320px
# name: nn_classifier_fig
# align: left
# ---
# The NN classifier being fooled by an outlier. Data items are colored according to their corresponding class values (<span style="color: red;">red</span> and <span style="color: blue;">blue</span> labels). The dashed straight line indicates a possible linear decision boundary between the two classes -- not the decision boundary produced by the NN classifier itself. The observed data item $ \check{\bf x} $ is classified according to the label (in this case, <span style="color: blue;">blue</span>) of the closest data item $ {\bf x}_{c} $ from the training dataset $ {\cal D} $. Something looks odd...
# ```
# 
# ### K-Nearest Neighbor (KNN) classifier
# 
# Find the $ K $ nearest neighbors of the observed data item $ \check{\bf x} $ within the training dataset $ {\cal D} $ and let them vote to predict the label $ \hat{y} $. Specifically, let the set
# \begin{equation}
# {\cal D}_{K}({\bf x}) \triangleq \bigcup_{i=1}^{K} \lbrace \left( {\bf x}_{i}', y_{i}' \right) \rbrace \subseteq {\cal D}
# \end{equation}
# collect the $ K $ nearest neighbors in $ {\cal D} $ of an arbitrary $ {\bf x} \in {\cal X} $ such that $$ || {\bf x} - {\bf x}_{i}' || \leq || {\bf x} - {\bf x}' ||, $$ $ \forall \left( {\bf x}_{i}', y_{i}' \right) \in {\cal D}_{K}({\bf x}) $ and $ \forall \left( {\bf x}', y' \right) \in {\cal D} \setminus {\cal D}_{K}({\bf x}) $.
# 
# ````{prf:definition}
# We define the KNN classifier as
# ```{math}
# :label: knn
# h_{KNN} ({\bf x}) = \argmax_{y \in {\cal Y}} F(y;{\cal D}_{K}({\bf x})),
# ```
# where the function 
# ```{math}
# :label: freq_labels
# F(y;{\cal D}_{K}({\bf x})) = \sum_{i=1}^{K} \left[ y_{i}' = y \right]
# ```
# indicates the frequency of the label $ y $ within $ {\cal D}_{K}({\bf x}) $. 
# ````
# 
# ````{margin}
# ```{note}
# Note that the set $ \lbrace \left( y, F(y;{\cal D}_{K}({\bf x})) \right) \mid y \in {\cal Y} \rbrace $ defines a histogram with the frequency of each class value $ y \in {\cal Y} $ in $ {\cal D}_{K}({\bf x}) $. The KNN prediction is the mode among the class values collected by $ {\cal D}_{K}({\bf x}) $.
# ```
# ````
# 
# ```{figure} /images/classification/knn_classifier.svg
# ---
# height: 320px
# name: knn_classifier_fig
# align: left
# ---
# The KNN classifier dealing with an outlier. The observed data item $ \check{\bf x} $ is classified according to the most often label (in this case, <span style="color: red;">red</span>) in $ {\cal D}_{K}(\check{\bf x}) $ associated with the $ K = 5 $ nearest data items within the training dataset $ {\cal D} $. Outliers are properly filtered out by allowing neighbors to vote. Again, the dashed straight line illustrates a possible linear decision boundary between the two classes. Note however that the boundary produced by the KNN classifier is typically non-linear as it is governed by the spread of the training samples over the feature space $ {\cal X} $.
# ```
# 
# ````{margin}
# ```{note}
# For binary classification, it is important to employ an odd number of $ K $ nearest neighbors to prevent a voting tie between the two classes. 
# ```
# ````
# 
# ```{prf:remark}
# **Pros:** The KNN classifier is simple (majority vote) and is non-parametric. Training is trivially achieved by storing the full dataset $ {\cal D} $. The KNN classifier is also Bayes consistent, i.e. the $ R(h_{KNN}) - R(h^{\ast}) \rightarrow 0 $ as both the number of training examples $ N \rightarrow \infty $ and the number of neighbors $ K \rightarrow \infty $, but $ N $ grows faster than $ K $, i.e $ \frac{K}{N} \rightarrow 0 $.
# 
# **Cons:** We need to store the full dataset $ {\cal D} $. Moreover, a naive implementation of KNN which checks all $ N $ training samples has cost $ \Theta(N) $. However, we can improve its computational complexity by storing the training samples into some computational structure -- e.g. kd-tree -- to efficiently retrieve the $ K $ nearest data items to an observed sample $ \check{\bf x} $. 
# ```
# 
# ### KNN for regression
# We can straightforwardly change the KNN classifier to perform regression by storing continuous values $ z \in {\cal Z} \subseteq \mathbb{R} $ instead of labels. In this case, the training dataset can be rewritten in this case as
# \begin{eqnarray}
# {\cal D} &=& \bigcup_{i=1}^{N} \lbrace \left( {\bf x}_{i}, {z}_{i} \right) \rbrace \nonumber \\
# &\equiv& \lbrace \left( {\bf x}_{1}, {z}_{1} \right), \left( {\bf x}_{2}, {z}_{2} \right), \ldots, \left( {\bf x}_{N}, {z}_{N} \right) \rbrace \nonumber
# \end{eqnarray}
# with $ {\bf x}_{i} \in {\cal X} $ and $ z_{i} \in {\cal Z} $, $ \forall i \in \lbrace 1, 2, \ldots, N \rbrace $.
# 
# Next, let the set
# \begin{equation}
# {\cal D}_{K}({\bf x}) \triangleq \bigcup_{i=1}^{K} \lbrace \left( {\bf x}_{i}', z_{i}' \right) \rbrace \subseteq {\cal D} 
# \end{equation}
# collect the $ K $ nearest neighbors in $ {\cal D} $ of a given data item $ {\bf x} $ such that $$ || {\bf x} - {\bf x}_{i}' || \leq || {\bf x} - {\bf x}' ||, $$ $ \forall \left( {\bf x}_{i}', z_{i}' \right) \in {\cal D}_{K}({\bf x}) $ and $ \forall \left( {\bf x}', z' \right) \in {\cal D} \setminus {\cal D}_{K}({\bf x}) $.
# 
# ````{margin}
# ```{note}
# Note though that the characteristics of the particular regression problem -- i.e. the function we are trying to approximate -- will drive the selection of the best interpolation method.
# ```
# ````
# 
# The values in $ {\cal D}_{K}({\bf x}) $ can be used then to predict the class value by means e.g. of a convex linear combination 
# \begin{equation}
# h_{KNN}({\bf x}) = \sum_{i=1}^{K} \alpha_{i}({\bf x}; {\bf x}_{i}') \, z_{i}',
# \end{equation}
# where the $i$-th coefficient / weight -- parameterized by $ {\bf x}_{i}' $ -- is $ \alpha_{i}({\bf x}; {\bf x}_{i}') \propto || {\bf x} - {\bf x}_{i}' || $ and the coefficients / weights are properly normalized such that $$ \sum_{i=1}^{K} \alpha_{i}({\bf x}; {\bf x}_{i}') = 1. $$
# 
