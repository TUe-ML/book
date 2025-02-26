#!/usr/bin/env python
# coding: utf-8

# ## Support Vector Machines
# 
# ### A word on hyperplanes
# 
# Let the expression $$ {\bf w}^{T} {\bf x} - b = 0 $$ -- parameterized by a coefficient vector $ {\bf w} = \begin{bmatrix} w_{1} & w_{2} & \ldots & w_{D} \end{bmatrix}^{T} $ and an offset term $ b $ -- denote a hyperplane in the space $ {\cal X} $. Note that $$ {\bf w}^{T} {\bf x} = w_{1}x_{1} + w_{2}x_{2} + \ldots + w_{D}x_{D} $$ is the inner product of vectors $ {\bf w} $ and $ {\bf x} $ such that $$ {\bf w}^{T} {\bf x} = 0 \Leftrightarrow {\bf w} \perp {\bf x}. $$ Hence, vectors $ {\bf x} = \begin{bmatrix} x_{1} & x_{2} & \ldots & x_{D} \end{bmatrix}^{T} $ satisfying $ {\bf w}^{T} {\bf x} - b = 0 $ determine a  hyperplane orthogonal to $ {\bf w} $. Reciprocally, the coefficient vector $ {\bf w} $ determines the orientation of the hyperplane $ {\bf w}^{T} {\bf x} - b = 0 $.
# 
# Moreover, we assume that $ {\bf w} $ is oriented to the positive side of the hyperplane. In this sense, the hyperplane segments the space into a *negative* side and a *positive* side. Specifically, a vector $ {\bf x} $ resides in the hyperplane when it satisfies the equality $ {\bf w}^{T} {\bf x} - b = 0 $. On the other hand, it falls in the *negative* or *positive* -- actually, non-negative -- sides when one of the following inequalities holds
# \begin{eqnarray}
# {\bf w}^{T} {\bf x} - b &<& 0\,\, \mbox{(negative side)} \nonumber \\
# {\bf w}^{T} {\bf x} - b &\geq& 0\,\, \mbox{(positive side)}. \nonumber
# \end{eqnarray}
# 
# ```{figure} /images/classification/distance_hyperplane_02.png
# ---
# height: 320px
# name: distance_hyperplane_02_fig
# align: left
# ---
# Hyperplane splitting a bidimensional space with $ {\bf x} = \begin{bmatrix} x_{1} & x_{2} \end{bmatrix}^{T} $. The *positive* and *negative* sides of the hyperplane are highlighted in light blue and light red colors, respectively. The coefficients vector $ {\bf w} $ is normal to the hyperplane by construction. We also assume that it is oriented towards the *positive* side of the hyperplane.
# ```
# 
# Now, let the function
# \begin{equation}
# d({\bf x}; {\bf w}, b) = \frac{ {\bf w}^{T} {\bf x} - b }{|| {\bf w} ||}
# \end{equation}
# define the *signed* distance between an input vector $ {\bf x} $ and the hyperplane defined by $ {\bf w} $ and $ b $, where the operator $ || \cdot || $ denotes the $ L_{2} $ (Euclidean) norm. Note that the signal of this distance measurement indicates whether the input vector $ {\bf x} $ resides on the negative -- $ d({\bf x}; {\bf w}, b) < 0 $ -- or positive -- $ d({\bf x}; {\bf w}, b) > 0 $ -- side of the hyperplane. Lastly, the offset term $ b $ determines the signed distance of the hyperplane from the origin of the feature space, since $$ d({\bf 0}; {\bf w}, b) = \frac{b}{|| {\bf w} ||}. $$
# 
# ```{figure} /images/classification/distance_hyperplane_06.png
# ---
# height: 320px
# name: distance_hyperplane_06_fig
# align: left
# ---
# Hyperplane splitting a bidimensional feature space with $ {\bf x} = \begin{bmatrix} x_{1} & x_{2} \end{bmatrix}^{T} $. Note that the offset term $ b $ determines the distance of the hyperplane from the origin $ {\bf 0} = \begin{bmatrix} 0 & 0 \end{bmatrix}^{T} $. In this case, $ b < 0 $ as the origin falls on the *negative* side of the hyperplane.
# ```
# 
# ### Basic principle
# 
# Support Vector Machines (SVMs) use hyperplanes to separate data items from multiple classes. For the sake of simplicity, let us restrict the discussion first to the binary classification problem, i.e $ {\cal Y} = \lbrace \ell_{1}, \ell_{2} \rbrace $. Now, let us further assume that the training dataset $$ {\cal D} = \bigcup_{i=1}^{N} \lbrace \left( {\bf x}_{i}, y_{i} \right) \rbrace, $$ in which $ {\bf x}_{i} \in {\cal X} $ and $ y_{i} \in {\cal Y} $, $ \forall i \in \lbrace 1, 2, \ldots N \rbrace $, is linearly separable in the sense that there is at least one hyperplane $$ {\bf w}^{T} {\bf x} - b = 0 $$ in the feature space $ {\cal X} $ that is able to separate the training examples such that data items in $ {\cal D} $ assigned to different labels $ \ell_{1} $ and $ \ell_{2} $ are in different sides of the hyperplane. In precise mathematical terms, $ \exists {\bf w} \in \mathbb{R}^{D} \wedge b \in \mathbb{R} $ such that the following holds $ \forall i \in \lbrace 1, 2, \ldots, N \rbrace $
# ```{math}
# :label: cond1
# {\bf w}^{T} {\bf x}_{i} - b < 0\,\, \mbox{ if } y_{i} = \ell_{1}
# ```
# ```{math}
# :label: cond2
# {\bf w}^{T} {\bf x}_{i} - b \geq 0\,\, \mbox{ if } y_{i} = \ell_{2}.
# ```
# 
# Alternatively, let the function $$ \sgn(a) = \left\lbrace \begin{matrix} +1 & \mbox{if } a \geq 0 \\ -1 & \mbox{otherwise} \end{matrix} \right. $$ indicate the sign of a real number $ a \in \mathbb{R} $. For a linearly separable binary classification problem, we can find a hyperplane defined by parameters $ \tilde{\bf w} $ and $ \tilde{b} $ that satisfies the conditions {eq}`cond1` and {eq}`cond2` and use then this hyperplane to build a classifier by plugging $ \tilde{\bf w} $ and $ \tilde{b} $ into
# ```{math}
# :label: hyperplane_classifier
# h({\bf x}; {\bf w}, b) = \sgn({\bf w}^{T} {\bf x}_{i} - b)
# ```
# such that labels $ \ell_{1} = -1 $ and $ \ell_{2} = +1 $.
# 
# ````{prf:remark}
# In general, there are multiple hyperplanes satisfying conditions {eq}`cond1` and {eq}`cond2`. Thus, we need to setup some optimization criteria to select the best hyperplane.
# 
# ```{figure} /images/classification/svm_hyperplane_1.svg
# ---
# height: 320px
# name: svm_hyperplane_1_fig
# align: left
# ---
# A hyperplane parameterized by $ \tilde{\bf w}$ and $ \tilde{b} $ separating the training samples in $ {\cal D} $. Training data items colored in <span style="color: red;">red</span> and <span style="color: blue;">blue</span> are assigned to labels $ \ell_{1} = -1 $ and $ \ell_{2} = +1 $, respectively. The observed data item $ \check{\bf x} $ is assigned to a label $ \hat{y} = h(\check{\bf x}; \tilde{\bf w}, \tilde{b}) $ according to the side of the hyperplane -- illustrated by the solid line -- it resides. The proposed classifier works fine for the particular samples in $ {\cal D} $, but its hyperplane seems too close to some <span style="color: red;">red</span> samples.
# ```
# ````
# 
# Now, let the dataset partitions $$ {\cal D}_{-1} = \lbrace \left( {\bf x}', y' \right) \in {\cal D} \mid y' = -1 \rbrace $$ and $$ {\cal D}_{+1} = \lbrace \left( {\bf x}', y' \right) \in {\cal D} \mid y' = +1 \rbrace $$ collect the training samples assigned respectively to the class labels $ \ell_{1} = -1 $ and $ \ell_{2} = +1 $.
# 
# We define the margin as the distance between two parallel hyperplanes touching the dataset partitions $ {\cal D}_{-1} $ and $ {\cal D}_{+1} $ such that none of the training samples in $ {\cal D} $ fall within the region between the two hyperplanes. The decision boundary correspond to a hyperplane with parameters $ {\bf w} $ and $ b $ equidistant to these parallel hyperplanes.
# 
# We seek to select the decision boundary that maximizes the margin between $ {\cal D}_{-1} $ and $ {\cal D}_{+1} $. Equivalently, we seek to select parameters $ {\bf w}^{\ast} $ and $ b^{\ast} $ defining a linear decision boundary such that the equidistant hyperplanes touching the dataset partitions $ {\cal D}_{-1} $ and $ {\cal D}_{+1} $ have maximum distance. 
# 
# More precisely, let $$ m_{-1}({\bf w}, b) = \min_{\left( {\bf x}', y' \right) \in {\cal D}_{-1}} - d({\bf x}'; {\bf w}, b) $$ and $$ m_{+1}({\bf w}, b) = \min_{\left( {\bf x}', y' \right) \in {\cal D}_{+1}} d({\bf x}'; {\bf w}, b) $$ denote the *non-signed* distances between the hyperplane defined by $ {\bf w} $ and $ b $ and the closest point(s) in the dataset partitions $ {\cal D}_{-1} $ and $ {\cal D}_{+1} $, respectively. The parameters of the hyperplane that provides the maximum margin between the dataset partitions $ {\cal D}_{-1} $ and $ {\cal D}_{+1} $ are given by
# ```{math}
# :label: svm_form_1
# \begin{eqnarray}
# \left( {\bf w}^{\ast}, b^{\ast} \right) &=& \argmax_{\left( {\bf w}, b \right) \in \mathbb{R}^{D+1}} \left\lbrace m_{-1}({\bf w}, b) + m_{+1}({\bf w}, b) \right\rbrace \\
# s.t. && m_{-1}({\bf w}, b) = m_{+1}({\bf w}, b) \\
# && h({\bf x}_{i}; {\bf w}, b) = y_{i}, \,\,\, \forall \left( {\bf x}_{i}, y_{i} \right) \in {\cal D}.
# \end{eqnarray}
# ```
# 
# For a linearly separable training dataset $ {\cal D} $, the maximum margin $ m_{-1}({\bf w}^{\ast}, b^{\ast}) + m_{+1}({\bf w}^{\ast}, b^{\ast}) $ is achieved for a unique pair of parameters $ {\bf w}^{\ast} $ and $ b^{\ast} $. Moreover, the solution is fully determined by a subset of the feature vectors in the dataset partitions $ {\cal D}_{-1} $ and $ {\cal D}_{+1} $ which **support** the hyperplane with maximum margin defined by $ {\bf w}^{\ast} $ and $ b^{\ast} $. These feature vectors are called **support vectors** and, therefore, the classifiers built using this principle are called Support Vector Machines (SVMs). For an arbitrary feature vector $ {\bf x} $, the SVM classifier is obtained by plugging the maximum margin hyperplane parameters $ {\bf w}^{\ast} $ and $ b^{\ast} $ into {eq}`hyperplane_classifier`
# $$
# h_{SVM}({\bf x}) \triangleq h({\bf x}; {\bf w}^{\ast}, b^{\ast}).
# $$
# 
# ```{figure} /images/classification/max_margin_decision_boundary.svg
# ---
# height: 320px
# name: max_margin_decision_boundary_fig
# align: left
# ---
# The maximum margin linear decision boundary separating the <span style="color: red;">red</span> and <span style="color: blue;">blue</span> training samples in $ {\cal D}_{-1} $ and $ {\cal D}_{+1} $, respectively. The data items highlighted with <span style="color: green;">green</span> borders correspond to the support vectors. The dashed lines illustrate the two parallel hyperplanes with maximum margin between them which are in turn uniquely defined by the support vectors. Note that the dashed lines can touch multiple support vectors of the training dataset splits $ {\cal D}_{-1} $ and $ {\cal D}_{+1} $. Lastly, the solid line indicates the decision boundary, i.e. the single hyperplane equidistant to the parallel hyperplanes that maximizes the margin between $ {\cal D}_{-1} $ and $ {\cal D}_{+1} $.
# ```
# 
# ````{prf:remark}
# Real-world training datasets are often not linearly separable. Specifically, for a non-linearly separable training dataset $ {\cal D} $, there is no hyperplane with parameters $ {\bf w} $ and $ b $ such that, $ \forall \left( {\bf x}_{i}, y_{i} \right) \in {\bf D} $,
# \begin{equation}
# h({\bf x}_{i}; {\bf w}, b) = y_{i}.
# \end{equation}
# Multiple causes may lead to non-linearly separable training datasets
# * The training dataset $ {\cal D} $ contains noise due to miss classifications performed by a human supervisor -- responsible for assigning *true* labels $ y_{i} $ to the data items $ {\bf x}_{i} $ within the training dataset;
# * Too much noise in the observed data itself -- each feature vector $ {\bf x}_{i} $ in $ {\cal D} $ contains noisy observations of the features describing a real-world object to be classified -- is preventing the training dataset $ {\cal D} $ to be linearly separable in the feature space $ {\cal X} $; and
# * The data items from different classes are not linearly separable at all in the current feature space $ {\cal X} $ -- perhaps we need more features -- i.e. a higher dimensional feature space $ {\cal X}' $ -- or to apply some transformation to the feature space $ {\cal X} $ to turn it into a linearly separable classification problem.
# 
# ```{figure} /images/classification/noisy_training_dataset.svg
# ---
# height: 320px
# name: noisy_training_dataset_fig
# align: left
# ---
# Noisy training dataset $ {\cal D} $. Circles filled in <span style="color: red;">red</span> and <span style="color: blue;">blue</span> represent labeled data items from the set partitions  $ {\cal D}_{-1} $ and $ {\cal D}_{+1} $, respectively. In particular, the <span style="color: red;">red</span> circle with <span style="color: blue;">blue</span> border corresponds to a data item assigned to the wrong label. On the other hand, the <span style="color: blue;">blue</span> circle with <span style="color: yellow;">yellow</span> border corresponds to a high-noise observed data item in $ {\cal D}_{+1} $ which fell too far in the region of the feature space $ {\cal X} $ containing data items from $ {\cal D}_{-1} $. Note that there is no way to change the dashed line representing a possible linear decision boundary so that it separates $ {\cal D}_{-1} $ from $ {\cal D}_{+1} $.
# ```
# ````
# 
# ### Learning with hard-constraints
# 
# The problem formulation in {eq}`svm_form_1` seeks to find the parameters $ {\bf w}^{\ast} $ and $ b^{\ast} $ that maximize the distances between the training samples $ \left( {\bf x}_{i}, y_{i} \right) \in {\cal D} $ to the hyperplane $ {\bf w}^{T} {\bf x} - b = 0 $ subjected to the correct classification of all training samples, i.e. $ h({\bf x}_{i}; {\bf w}, b) = y_{i} $, $ \forall i \in \lbrace 1, 2, \ldots, N \rbrace $. Unfortunately, this optimization problem is still too abstract for a practical optimizer to solve it. Thus, we need to reformulate it using geometric principles to turn it into a solvable optimization problem.
# 
# Let us consider first the following two parallel hyperplanes
# \begin{eqnarray}
# {\bf w}^{T} {\bf x} - b &=& -1\,\, \mbox{(negative boundary)} \nonumber \\
# {\bf w}^{T} {\bf x} - b &=& +1\,\, \mbox{(positive boundary)} \nonumber
# \end{eqnarray}
# defining *negative* and *positive* boundaries such that the maximum-margin hyperplane $ {\bf w}^{T} {\bf x} - b = 0 $ lies halfway between them. We call the region between these *negative* and *positive* boundaries as the *margin*. Lastly, the distance between the *negative* and *positive* boundaries -- or equivalently, the margin thickness -- is given by $ \frac{2}{||{\bf w}||} $.
# 
# ```{figure} /images/classification/margin_planes_03.png
# ---
# height: 480px
# name: margin_planes_03_fig
# align: left
# ---
# The *margin* between the *negative* and *positive* boundaries with the decision boundary lies half way between them. The *margin* thickness $ \frac{2}{||{\bf w}||} $ is in turn regulated by the reciprocal of $ || {\bf w} ||$.
# ```
# 
# As the training samples are not allowed to fall in the margin region, we must select $ {\bf w} $ and $ b $ such that 
# \begin{eqnarray}
# {\bf w}^{T} {\bf x}_{i} - b &\leq& -1\,\, \mbox{if } y_{i} = -1 \nonumber \\
# {\bf w}^{T} {\bf x}_{i} - b &\geq& +1\,\, \mbox{if } y_{i} = +1 \nonumber
# \end{eqnarray}
# for all training samples $ \left( {\bf x}_{i}, y_{i} \right) \in {\cal D} $. Alternatively, we can write
# \begin{equation}
# y_{i} \left( {\bf w}^{T} {\bf x} - b \right) \geq +1,  \nonumber
# \end{equation}
# since the class label $ y_{i} $ and the function $ h({\bf x}_{i}; {\bf w}, b) = {\bf w}^{T} {\bf x}_{i} - b $ have always[^footnote2] the same sign $ \forall i \in \lbrace 1, 2, \ldots, N \rbrace $.
# 
# [^footnote2]: Assuming that the hyperplane $ {\bf w}^{T} {\bf x} - b = 0$ is a valid decision boundary for the linearly separable training dataset ${\cal D}$.
# 
# Thus, we can rewrite the optimization problem in {eq}`svm_form_1` as
# ```{math}
# :label: svm_form_2
# \begin{eqnarray}
# \left( {\bf w}^{\ast}, b^{\ast} \right) &=& \argmax_{\left( {\bf w}, b \right) \in \mathbb{R}^{D+1}} \frac{2}{||{\bf w}||} \\
# &\equiv& \argmax_{\left( {\bf w}, b \right) \in \mathbb{R}^{D+1}} \frac{1}{||{\bf w}||} \\
# s.t. &&  y_{i} \left( {\bf w}^{T} {\bf x}_{i} - b \right) \geq +1, \,\,\, \forall \left( {\bf x}_{i}, y_{i} \right) \in {\cal D},
# \end{eqnarray}
# ```
# i.e. we seek to find the parameters $ {\bf w}^{\ast} $ and $ b^{\ast} $ that maximize the margin thickness $ \frac{2}{||{\bf w}||} $ subjected to the linear constrains $ y_{i} \left( {\bf w}^{T} {\bf x}_{i} - b \right) \geq +1 $, $ \forall i \in \lbrace 1, 2, \ldots, N \rbrace $. However, this is a non-convex optimization problem as the denominator $ ||{\bf w}|| $ introduces a singularity in the objective function $ \frac{1}{||{\bf w}||} $ in {eq}`svm_form_2`. Fortunately, we can reformulate the problem of maximizing the reciprocal of the norm $ ||{\bf w}|| $ into the problem o minimizing the norm $ ||{\bf w}|| $ itself
# ```{math}
# :label: svm_form_3
# \begin{eqnarray}
# \left( {\bf w}^{\ast}, b^{\ast} \right) &=& \argmin_{\left( {\bf w}, b \right) \in \mathbb{R}^{D+1}} ||{\bf w}|| \\
# s.t. &&  y_{i} \left( {\bf w}^{T} {\bf x}_{i} - b \right) \geq +1, \,\,\, \forall \left( {\bf x}_{i}, y_{i} \right) \in {\cal D},
# \end{eqnarray}
# ```
# which is in turn a convex optimization problem. For convenience, we replace the objective function $ ||{\bf w}|| $ in {eq}`svm_form_3` by $ \frac{1}{2}||{\bf w}||^{2} $ -- which is a quadratic function differentiable everywhere -- and write
# ```{math}
# :label: svm_form_4
# \begin{eqnarray}
# \left( {\bf w}^{\ast}, b^{\ast} \right) &=& \argmin_{\left( {\bf w}, b \right) \in \mathbb{R}^{D+1}} \frac{1}{2}||{\bf w}||^{2} \\
# s.t. &&  y_{i} \left( {\bf w}^{T} {\bf x}_{i} - b \right) \geq +1, \,\,\, \forall \left( {\bf x}_{i}, y_{i} \right) \in {\cal D},
# \end{eqnarray}
# ```
# so that we can use e.g. gradient descent solvers to optimize the parameters $ {\bf w} $ and $ b $.
# 
# In summary, we transformed the non-convex optimization problem of maximizing the margin in {eq}`svm_form_1` into a convex optimization problem of minimizing the squared norm $ \frac{1}{2}||{\bf w}||^{2} $ in {eq}`svm_form_4` which is subjected in turn to several margin constraints $$ y_{i} \left( {\bf w}^{T} {\bf x}_{i} - b \right) \geq +1, $$ for $ i \in \lbrace 1, 2, \ldots, N \rbrace $.
# 
# ### Learning with soft-constraints
# 
# Unfortunately, the hard-margin formulation still requires a linearly separable training dataset $ {\cal D} $ so that the linear constraints $$ y_{i} \left( {\bf w}^{T} {\bf x}_{i} - b \right) \geq +1, $$ $ i \in \lbrace 1, 2, \ldots, N \rbrace $, can be satisfied. To overcome this limitation, we can relax the linear constraints by discounting a fixed amount $ \xi_{i} \geq 0 $ from each constraint as $$ y_{i} \left( {\bf w}^{T} {\bf x}_{i} - b \right) \geq +1 - \xi_{i} $$ such that the original constraint is fully imposed for $ \xi_{i} = 0 $ and it is incrementally relaxed as $ \xi_{i} $ grows for $ \xi_{i} > 0 $. Let the vector $ \boldsymbol{\xi} = \begin{bmatrix} \xi_{1} & \xi_{2} & \ldots & \xi_{N} \end{bmatrix}^{T} $ collect all $ N $ *slack* variables. Thus, we can rewrite the optimization problem in {eq}`svm_form_4` assuming relaxed linear constraints as
# ```{math}
# :label: svm_form_5
# \begin{eqnarray}
# \left( {\bf w}^{\ast}, b^{\ast}, \boldsymbol{\xi}^{\ast} \right) &=& \argmin_{\left( {\bf w}, b, \boldsymbol{\xi} \right) \in \mathbb{R}^{D+N+1}} \left\lbrace \frac{1}{2}||{\bf w}||^{2} + C \sum_{i=1}^{N} \xi_{i} \right\rbrace \\
# s.t. &&  y_{i} \left( {\bf w}^{T} {\bf x}_{i} - b \right) \geq +1 - \xi_{i} \\
# && \xi_{i} \geq 0, \,\, \forall i \in \lbrace 1, 2, \ldots, N \rbrace,
# \end{eqnarray}
# ```
# where the hyperparameter $ C > 0 $ regulates the trade off between the original objective function $ \frac{1}{2}||{\bf w}||^{2} $ and the slacks sum $ \sum_{i=1}^{N} \xi_{i} $ such that $ C \rightarrow \infty $ leads to the original optimization problem with hard margin. Conversely, as $ C \rightarrow 0^{+} $, the constraints are increasingly relaxed. Note that this formulation allows the $i$-th training example to violate the original constraint by a fixed amount $ \xi_{i} $ (slack). However, the optimization problem in {eq}`svm_form_5` is set up such that the sum of these violations $ \sum_{i=1}^{N} \xi_{i} $ is also minimized.
# 
# ````{margin}
# ```{note}
# The slack variables $ \lbrace \xi_{1}, \ldots, \xi_{N} \rbrace $ allow therefore to train a SVM classifier using linearly non-separable datasets. 
# ```
# 
# ```{note}
# We refer to {eq}`svm_form_5` as the **primal** SVM formulation which allows us to build a SVM classifier using either linearly separable or linearly non-separable training datasets.
# ```
# ````
# 
# ````{prf:remark}
# Equivalently, the soft-margin problem in {eq}`svm_form_5` can be reformulated as a convex optimization problem without constraints. In particular, we can write
# ```{math}
# :label: svm_form_6
# \begin{equation}
# \left( {\bf w}^{\ast}, b^{\ast} \right) = \argmin_{\left( {\bf w}, b, \boldsymbol{\xi} \right) \in \mathbb{R}^{D+1}} \left\lbrace \frac{1}{2}||{\bf w}||^{2} + C \sum_{i=1}^{N} \underbrace{ \max \left( 1 - y_{i} \left( {\bf w}^{T} {\bf x}_{i} - b \right), 0 \right)}_{\mbox{hinge loss}} \right\rbrace,
# \end{equation}
# ```
# in which the hinge loss $$ \epsilon_{i} \triangleq \max \left( 1 - y_{i} \left( {\bf w}^{T} {\bf x}_{i} - b \right), 0 \right) $$ associated with the $ i $-th training data item is such that
# \begin{eqnarray}
# y_{i} \left( {\bf w}^{T} {\bf x}_{i} - b \right) \geq 1 &\rightarrow& \epsilon_{i} = 0\,\, \mbox{(no penalty)} \nonumber \\
# 0 \leq y_{i} \left( {\bf w}^{T} {\bf x}_{i} - b \right) < 1 &\rightarrow& 0 < \epsilon_{i} \leq 1\,\, \mbox{(small penalty)} \nonumber \\
# y_{i} \left( {\bf w}^{T} {\bf x}_{i} - b \right) < 0 &\rightarrow& \epsilon_{i} > 1\,\, \mbox{(unbounded penalty)}. \nonumber
# \end{eqnarray}
# Note that there is no penalty in the first case, since the training sample $ \left( {\bf x}_{i}, y_{i} \right) $ is not violating the constraint. On the other hand, there is a small penalty in the second case, as the data item $ {\bf x}_{i} $ falls within the margin, but still in the right side of the hyperplane defined by $ {\bf w} $ and $ b $. The hinge loss grows unbounded in the last case since the data item $ {\bf x}_{i} $ falls on the wrong side of the hyperplane in this case. The objective function in {eq}`svm_form_6` penalizes thus the sum of the hinge losses committed by all data items in $ {\cal D} $. Lastly, as the hinge loss function has a non-linearity around the origin -- it is clamped to zero when the constraint $ y_{i} \left( {\bf w}^{T} {\bf x}_{i} - b \right) \geq 1 $ is satisfied --, the objective function is neither differentiable everywhere nor quadratic with respect to the parameters $ {\bf w} $ and $ b $ anymore.
# ````
# 
# ````{prf:example}
# The effect of the hyperparameter $ C $ on the decision boundary. Note that, as $ C $ grows, the maximum-margin hyperplane becomes more diplomatic in the sense of keeping as much as possible distance to the training examples from different classes. Lastly, note that the final solution becomes influence by more and more data items in $ {\cal D} $ as $ C $ decreases, i.e. more and more support vectors -- corresponding to training samples $ \lbrace \left( {\bf x}', y'\right) \in {\cal D} \mid y' \left( {\bf w}^{T} {\bf x}' - b \right) \leq 1 \rbrace $ -- will contribute to determine the linear decision boundary $ {\bf w}^{T} {\bf x}' - b = 0 $. On the other hand, the training samples $ \lbrace \left( {\bf x}', y'\right) \in {\cal D} \mid y' \left( {\bf w}^{T} {\bf x}' - b \right) > 1 \rbrace $ do not contribute directly to the values of the parameters $ {\bf w} $ and $ b $ in the sense that any changes of their positions in the feature space $ {\cal X} $ without violating the constraints lead to the same solution.
# 
# ```{figure} /images/classification/effect_softmargin_sep.png
# ---
# height: 200px
# name: effect_softmargin_sep_fig
# align: left
# ---
# Linearly separable dataset.
# ```
# ```{figure} /images/classification/effect_softmargin_nonsep.png
# ---
# height: 200px
# name: effect_softmargin_nonsep_fig
# align: left
# ---
# Linearly non-separable dataset.
# ```
# ````
# 
# ### A word on duality
# 
# Consider first the **primal** optimization problem of minimizing an objective function $ f:{\cal Z} \rightarrow \mathbb{R} $ across some space $ {\cal Z} $ subjected to several constraints. We can write down this problem using the standard format as
# ```{math}
# :label: primal_prob1
# \begin{eqnarray}
# \minimize_{{\bf z} \in {\cal Z}} &&  f({\bf z}) \\
# s.t. \,\,\, &&  g_{i} ({\bf z}) \leq 0,\,\,\, \forall i \in \lbrace 1, 2, \ldots, N \rbrace,
# \end{eqnarray}
# ```
# in which we omitted equality constraints of the type $ h_{j}({\bf z}) = 0 $, $ j \in \lbrace 1, 2, \ldots M \rbrace $, for convenience. Now, let
# ```{math}
# :label: lagrandian
# {\cal L}({\bf z}, \boldsymbol{\lambda}) = f({\bf z}) + \sum_{i=1}^{N} \lambda_{i} g_{i}({\bf z})
# ```
# be the Lagrangian of the primal problem as stated in {eq}`primal_prob1` in which the vector $ \boldsymbol{\lambda} \triangleq \begin{bmatrix} \lambda_{1} & \lambda_{2} & \ldots & \lambda_{N} \end{bmatrix}^{T} $ collect the so-called Lagrange multipliers $ \lbrace \lambda_{i} \rbrace $ corresponding to the constraints $ \lbrace  g_{i} ({\bf z}) \leq 0 \rbrace $, $ i \in \lbrace 1, 2, \ldots, N \rbrace $. The $ i $-th Lagrangian multiplier determines how much penalty is assigned to the violation of the constraint $ g_{i} ({\bf z}) \leq 0 $ such that any violation is allowed for $ \lambda_{i} = 0 $ and no violation is allowed at all for $ \lambda_{i} \rightarrow \infty $. Thus, the hard constraints in {eq}`primal_prob1` can be enforced in {eq}`lagrandian` by chosen sufficiently high values for the Lagrange multipliers.
# 
# Let us define
# ```{math}
# :label: lagrandian_primal
# {\cal L}_{primal}({\bf z}) = \max_{\boldsymbol{\lambda} \geq {\bf 0}} {\cal L}({\bf z}, \boldsymbol{\lambda})
# ```
# as the maximum of the Lagrangian in {eq}`lagrandian` for some value of $ {\bf z} \in {\cal Z} $. 
# 
# Assume that all constraints $ \lbrace g_{i} ({\bf z}') \leq 0 \rbrace $, $ \forall i \in \lbrace 1, 2, \ldots, N \rbrace $, are satisfied for a feasible $ {\bf z}' $, then all $ \lbrace g_{i} ({\bf z}') \rbrace $ in the right-hand side of {eq}`lagrandian` will be negative and the best thing we can do to maximize the Lagrangian $ {\cal L}({\bf z}', \boldsymbol{\lambda}) $ in {eq}`lagrandian_primal` is to choose $ \boldsymbol{\lambda} = {\bf 0} $. On the other hand, let us assume that $ {\bf z}' $ is an unfeasible point that violates at least one of the constraints, let us say the $i$-th constraint. In this case, the best thing we can do to maximize the Lagrangian $ {\cal L}({\bf z}', \boldsymbol{\lambda}) $ in {eq}`lagrandian_primal` is to allow the $ i $-th Lagrange multiplier to grow unbounded, i.e. to make $ \lambda_{i} \rightarrow \infty $. 
# 
# As the maximization in {eq}`lagrandian_primal` strongly penalizes unfeasible points, we can restate then the **primal** problem {eq}`primal_prob1` using the Lagrangian {eq}`lagrandian` as
# ```{math}
# :label: primal_prob2
# \min_{{\bf z} \in {\cal Z}} {\cal L}_{primal}({\bf z}).
# ```
# 
# In the sequel, let 
# ```{math}
# :label: lagrandian_dual
# {\cal L}_{dual}(\boldsymbol{\lambda}) = \min_{{\bf z} \in {\cal Z}} {\cal L}({\bf z}, \boldsymbol{\lambda})
# ```
# be the minimum of Lagrangian in {eq}`lagrandian` for some $ \boldsymbol{\lambda} \geq {\bf 0} $. In this case, for a fixed vector $ \boldsymbol{\lambda}' $ modulating how strongly violations to the constraints $ \lbrace g_{i} ({\bf z}') \leq 0 \rbrace $, $ \forall i \in \lbrace 1, 2, \ldots, N \rbrace $ shall be penalized, we find some point $ {\bf z} \in {\cal Z} $ in {eq}`lagrandian_dual` that minimizes the Lagrangian $ {\cal L}({\bf z}, \boldsymbol{\lambda}') $.
# 
# Additionally, one can show that $ \forall \boldsymbol{\lambda} \geq {\bf 0} $
# ```{math}
# :label: dual_prob2
# {\cal L}_{dual}(\boldsymbol{\lambda}) \leq \min_{{\bf z} \in {\cal Z}} {\cal L}_{primal}({\bf z}).
# ```
# That is, the **primal** problem in {eq}`primal_prob2` is lower bounded by $ {\cal L}_{dual}(\boldsymbol{\lambda}) $. Hence, one can find a tighter lower bound to the solution of the original problem by finding the Lagrangian multipliers $ \boldsymbol{\lambda} $ that maximizes {eq}`lagrandian_dual`. Specifically, the following tighter lower-bound holds
# ```{math}
# :label: lower_bound
# \max_{\boldsymbol{\lambda} \geq {\bf 0}} {\cal L}_{dual}(\boldsymbol{\lambda}) \leq \min_{{\bf z} \in {\cal Z}} {\cal L}_{primal}({\bf z}).
# ```
# 
# The alternative formulation
# ```{math}
# :label: dual_prob1
# \max_{\boldsymbol{\lambda} \geq {\bf 0}} {\cal L}_{dual}(\boldsymbol{\lambda})
# ```
# from the left-hand side of {eq}`dual_prob1` is called the **dual** problem and has some amenable properties. In particular, it is a lower bound to the **primal** problem. Finally, it is worth noting that solving the **dual** problem in {eq}`dual_prob1` is equivalent to finding the Lagrange multipliers in $ \boldsymbol{\lambda} \geq {\bf 0} $ that lead to the tightest (best) lower bound in {eq}`dual_prob2`.
# 
# ```{prf:remark}
# From {eq}`lower_bound`, we conclude that, for any $ {\bf z} \in {\cal Z} $ and $ \boldsymbol{\lambda} \geq {\bf 0} $, $$ {\cal L}_{dual}(\boldsymbol{\lambda}) \leq {\cal L}_{primal}({\bf z}). $$ Thus, in general, the **dual** problem $$ \max_{\boldsymbol{\lambda} \geq {\bf 0}} {\cal L}_{dual}(\boldsymbol{\lambda}) $$ is a lower bound to the **primal** problem. Alternatively, we can write $$ \max_{\boldsymbol{\lambda} \geq {\bf 0}} {\cal L}_{dual}(\boldsymbol{\lambda}) \leq \min_{{\bf z} \in {\cal Z}} {\cal L}_{primal}({\bf z}). $$ However, for convex optimization problems, the **primal** and **dual** problems are tight, i.e. their optimal values are the same $$ \max_{\boldsymbol{\lambda} \geq {\bf 0}} {\cal L}_{dual}(\boldsymbol{\lambda}) = \min_{{\bf z} \in {\cal Z}} {\cal L}_{primal}({\bf z}). $$ Thus, in some cases, the **dual** problem also provides a solution to the **primal** one.
# ```
# 
# ````{margin}
# ```{note}
# The **dual** formulation in {eq}`dual_prob1` is also useful even when the **primal** and **dual** problems are not tight. Sometimes the solution to the **primal** problem is hard to achieve while the **dual** problem has a computationally efficient solution. Thus, we can solve the **dual** problem to evaluate how close to the lower bound an iterative solution to the **primal** problem was able to reach so far and use it as a stop criteria.
# ```
# ````
# 
# ### Dual SVM formulation
# 
# Let us rewrite the hard-margin SVM problem in {eq}`svm_form_4` using the standard format as
# ```{math}
# :label: svm_form_7
# \begin{eqnarray}
# \minimize_{{\bf w} \in \mathbb{R}^{D}, b \in \mathbb{R}} &&  \frac{1}{2}||{\bf w}||^{2} \\
# s.t. \,\,\, &&  1 - y_{i} \left( {\bf w}^{T} {\bf x}_{i} - b \right) \leq 0, \,\,\, \forall \left( {\bf x}_{i}, y_{i} \right) \in {\cal D}.
# \end{eqnarray}
# ```
# We can write the Lagrangian of {eq}`svm_form_7` as
# ```{math}
# :label: lagrandian2
# {\cal L}({\bf z}, \boldsymbol{\lambda}) = f({\bf z}) + \sum_{i=1}^{N} \lambda_{i} \left( 1 - y_{i} \left( {\bf w}^{T} {\bf x}_{i} - b \right)  \right),
# ```
# which is convex with respect to the parameters $ {\bf w} $ and $ b $. Thus, we can find the solution to $$ {\cal L}_{dual}(\boldsymbol{\lambda}) = \min_{{\bf z} \in {\cal Z}} {\cal L}({\bf z}, \boldsymbol{\lambda}) $$ by setting both the gradient $$ \nabla_{\bf w} {\cal L}({\bf z}, \boldsymbol{\lambda}) = {\bf w} - \sum_{i=1}^{N} \lambda_{i} y_{i} {\bf x}_{i} $$ and the partial derivative $$ \frac{\partial {\cal L}({\bf z}, \boldsymbol{\lambda})}{\partial b} = \sum_{i=1}^{N} \lambda_{i} y_{i} $$ to zero. Hence, we write
# ```{math}
# :label: w_grad_zero
# \nabla_{\bf w} {\cal L}({\bf z}, \boldsymbol{\lambda}) = 0 \Leftrightarrow {\bf w} = \sum_{i=1}^{N} \lambda_{i} y_{i} {\bf x}_{i}
# ```
# ```{math}
# :label: cond_partial_zero
# \frac{\partial {\cal L}({\bf z}, \boldsymbol{\lambda})}{\partial b} = 0 \Leftrightarrow \sum_{i=1}^{N} \lambda_{i} y_{i} = 0.
# ```
# Now, by plugging {eq}`w_grad_zero` and {eq}`cond_partial_zero` back into the Lagrangian definition {eq}`lagrandian2`, we obtain
# \begin{eqnarray}
# {\cal L}_{dual}(\boldsymbol{\lambda}) &=& \frac{1}{2} \sum_{i=1}^{N} \sum_{j=1}^{N} \lambda_{i} \lambda_{j} y_{i} y_{j} {\bf x}_{i}^{T} {\bf x}_{j} + \sum_{i=1}^{N} \lambda_{i} - \underbrace{\sum_{i=1}^{N} \lambda_{i}y_{i} {\bf x}_{i} \sum_{j=1}^{N} \lambda_{j} y_{j} {\bf x}_{j}}_{= \sum_{i=1}^{N} \sum_{j=1}^{N} \lambda_{i} \lambda_{j} y_{i} y_{j} {\bf x}_{i}^{T} {\bf x}_{j}} + \underbrace{\sum_{i=1}^{N} \lambda_{i} y_{i}}_{= 0} b \nonumber \\
# &=& - \frac{1}{2} \sum_{i=1}^{N} \sum_{j=1}^{N} \lambda_{i} \lambda_{j} y_{i} y_{j} {\bf x}_{i}^{T} {\bf x}_{j} + \sum_{i=1}^{N} \lambda_{i}.
# \end{eqnarray}
# 
# Thus, we can write the dual problem in {eq}`dual_prob1` as
# ```{math}
# :label: svm_form8
# \begin{eqnarray}
# \maximize_{\boldsymbol{\lambda} \geq {\bf 0}} &&  - \frac{1}{2} \sum_{i=1}^{N} \sum_{j=1}^{N} \lambda_{i} \lambda_{j} y_{i} y_{j} {\bf x}_{i}^{T} {\bf x}_{j} + \sum_{i=1}^{N} \lambda_{i} \\
# s.t. \,\,\, &&  \sum_{i=1}^{N} \lambda_{i} y_{i} = 0 \\
# &&  0 \leq \lambda_{i}, \,\,\, \forall i \in \lbrace 1, 2, \ldots, N \rbrace,
# \end{eqnarray}
# ```
# which is clearly a quadratic program leading therefore to a convex optimization problem. Hence, the optimal $ \boldsymbol{\lambda}^{\ast} $ in {eq}`svm_form8` also delivers the optimal solution for the primal problem in {eq}`svm_form_7`. Specifically,
# ```{math}
# :label: w_optimal
# {\bf w}^{\ast} = \sum_{i=1}^{N} \lambda_{i}^{\ast} y_{i} {\bf x}_{i}
# ```
# ```{math}
# :label: b_optimal
# b^{\ast} = ({\bf w}^{\ast})^{T} {\bf x}_{j} - y_{j},
# ```
# in which we plug any training example $ \left( {\bf x}_{j}, y_{j} \right) $ with index $ j \in \lbrace 1, 2, \ldots, N \rbrace $ such that $ \lambda_{j} > 0 $.
# 
# Finally, we offer without proof that {eq}`svm_form8` can be rewritten to consider soft-margin constraints simply by plugging in the hyperparameter $ C $ into the linear constraints as follows
# ```{math}
# :label: svm_form9
# \begin{eqnarray}
# \maximize_{\boldsymbol{\lambda} \geq {\bf 0}}  &&  - \frac{1}{2} \sum_{i=1}^{N} \sum_{j=1}^{N} \lambda_{i} \lambda_{j} y_{i} y_{j} {\bf x}_{i}^{T} {\bf x}_{j} + \sum_{i=1}^{N} \lambda_{i} \\
# s.t. \,\,\, &&  \sum_{i=1}^{N} \lambda_{i} y_{i} = 0 \\
# &&  0 \leq \lambda_{i} \leq C, \,\,\, \forall i \in \lbrace 1, 2, \ldots, N \rbrace.
# \end{eqnarray}
# ```
# 
# Alternatively, we can rewrite the dual problem in {eq}`svm_form9` more concisely in terms of the so-called Gram matrix, a.k.a. pairwise influence matrix, $ {\bf G} \triangleq \left[ G_{i,j} \right] $ with $ G_{i,j} = y_{i} y_{j} {\bf x}_{i}^{T} {\bf x}_{j} $ for $ i,j \in \lbrace 1, 2, \ldots N \rbrace $ as
# ```{math}
# :label: svm_form10
# \begin{eqnarray}
# \maximize_{\boldsymbol{\lambda} \geq {\bf 0}}  &&  - \frac{1}{2} \boldsymbol{\lambda}^{T} {\bf G} \boldsymbol{\lambda} + {\bf 1}^{T} \boldsymbol{\lambda} \\
# s.t. \,\,\, &&   {\bf y}^{T} \boldsymbol{\lambda} = 0 \\
# &&  0 \leq \lambda_{i} \leq C, \,\,\, \forall i \in \lbrace 1, 2, \ldots, N \rbrace,
# \end{eqnarray}
# ```
# with $ {\bf y} \triangleq \begin{bmatrix} y_{1} & y_{2} & \ldots & y_{N} \end{bmatrix}^{T} $ and $ \boldsymbol{\lambda} = \begin{bmatrix} \lambda_{1} & \lambda_{2} & \ldots & \lambda_{N} \end{bmatrix}^{T} $ collecting the labels and the slack variables associated with each training example, respectively, and $ {\bf 1} \triangleq \begin{bmatrix} 1 & 1 & \ldots & 1 \end{bmatrix}^{T} $ representing a vector of $1$'s with appropriate number of dimensions -- in this case $ N $. Note also that {eq}`svm_form10` is equivalent to writing
# ```{math}
# :label: svm_form11
# \begin{eqnarray}
# \minimize_{\boldsymbol{\lambda} \geq {\bf 0}}  &&  \frac{1}{2} \boldsymbol{\lambda}^{T} {\bf G} \boldsymbol{\lambda} - {\bf 1}^{T} \boldsymbol{\lambda} \\
# s.t. \,\,\, &&   {\bf y}^{T} \boldsymbol{\lambda} = 0 \\
# &&  0 \leq \lambda_{i} \leq C, \,\,\, \forall i \in \lbrace 1, 2, \ldots, N \rbrace.
# \end{eqnarray}
# ```
# 
# ```{prf:remark}
# As the objective function $ \frac{1}{2} || {\bf w} ||^{2} $ in {eq}`svm_form9` depends on the coefficient vector $$ {\bf w} = \begin{bmatrix} w_{1} & w_{2} & \ldots & w_{D} \end{bmatrix}^{T}, $$ the primal SVM problem has computational complexity $ {\cal O}(D) $ with $ D $ denoting the number of features in the feature space $ {\cal X} $. On the other hand, the dual SVM problem in {eq}`svm_form9` is clearly quadratic on the number of samples $ N $, i.e it has computational complexity $ {\cal O} (N^{2}) $. Thus, for a large number of features $ D \gg N $, the dual SVM problem can be cheaper, while the solution to the primal SVM problem has a smaller computational burden for large datasets with $ N \gg D $.
# ```
# 
# ````{margin}
# ```{note}
# The objective function of the dual SVM problem formulations in {eq}`svm_form8`--{eq}`svm_form11` depends only on the inner products $ \lbrace {\bf x}_{i}^{T} {\bf x}_{j} \rbrace $ with $ i,j \in \lbrace 1, 2, \ldots, N \rbrace $. This dependence unlocks the kernel trick that will be discussed in the sequel.
# ```
# ````
# 
# ### A word on kernels
# 
# A kernel function $ k({\bf x}_{i}, {\bf x}_{j}) $ maps pairs of vectors $ {\bf x}_{i}, {\bf x}_{j} $ residing in a $D$-dimensional Euclidean space $ \mathbb{R}^{D} $ into real numbers in $ \mathbb{R} $, i.e. $$ k: \mathbb{R}^{D} \times \mathbb{R}^{D} \rightarrow \mathbb{R}. $$ Furthermore, a kernel is positive definite if for any *finite* collection of vectors $ {\bf x}_{1}, \ldots, {\bf x}_{N} $ and any collection of real numbers $ a_{1}, \ldots, a_{N} $, the following holds $$ \sum_{i=1}^{N} \sum_{j=1}^{N} a_{i} a_{j} {\bf x}_{i}^{T} {\bf x}_{j} \geq 0. $$ Alternatively, we can write in vector notation as
# ```{math}
# :label: positive_definite
# {\bf a}^{T} {\bf K} {\bf a} \geq 0,
# ```
# where $ {\bf K} = \left[ K_{i,j} \right] $ is a $ N \times N $ matrix with $ K_{i,j} = k({\bf x}_{i}, {\bf x}_{j}) $ for all $ i,j \in \lbrace 1, 2, \ldots, N \rbrace $ and $ {\bf a} = \begin{bmatrix} a_{1} & a_{2} & \ldots & a_{N} \end{bmatrix}^{T} $ is an arbitrary vector in $ \mathbb{R}^{N} $.
# 
# ### Kernel trick
# 
# Applying non-linear transformations of the type $$ \phi : {\cal X} \rightarrow {\cal Z} $$ mapping feature vectors $ {\bf x} \in {\cal X} $ into a higher-dimensional space $ {\cal Z} $ can significantly boost several Machine Learning (ML) algorithms, for instance SVMs, Principal Component Analysis (PCA) and Linear Discriminant Analysis (LDA).
# 
# Mapping feature vectors into such high-dimensional space $ {\cal Z} $ can be effective but it is often expensive and selecting the proper non-linear transformation $ \phi(\cdot) $ is also a hard task. Fortunately, provided that the ML algorithm works with inner products of feature vectors as in {eq}`svm_form8`--{eq}`svm_form11`, the kernel trick allows one to work with high-dimensional spaces efficiently since it avoids computing $ \phi(\cdot) $ explicitly.
# 
# In most classification problems, decision boundaries are far from linear. In particular, for real-world binary classification problems, chances are that no hyperplane can separate training examples from both classes. On the other hand, high-dimensional data tends to be linearly separable. Intuitively, as the number of features increases, chances are that the data become linearly separable with respect of some of these features.
# 
# One approach to increase the number of dimensions is to apply **non-linear feature transforms** to the original features in $ {\cal X} $ transforming thus the learning problem into a higher dimensional feature space $ {\cal Z} $. Note though that the feature transformations must be non-linear, for instance polynomials, exponential (e.g. $\exp$), logarithm (e.g. $\log$) and trigonometric functions (e.g. $\cos$, $\sin$, $\tanh$). This is a necessary (non-sufficient) condition. Note though that if we apply linear (affine) transformations, the data in ${\cal Z} $ will still linearly non-separable as in the original feature space $ {\cal X} $.
# 
# {numref}`non_linear_transformation_fig` illustrates a particular non-linear transformations of features. Note that, despite being linearly non-separable in the original space $ {\cal X} $, the non-linear features in $ {\cal Z} $ become linearly separable, i.e. one can design a linear classifier of the type 
# $$
# h({\bf z}; {\bf w}, b) = \sgn({\bf w}^{T} {\bf z} - b)
# $$
# residing in the non-linear feature space $ {\cal Z} $ that is able to separate the data from both classes. Equivalently, we can also write
# $$
# h({\bf x}; {\bf w}, b) = \sgn({\bf w}^{T} \phi({\bf x}) - b).
# $$
# 
# ```{figure} /images/classification/degree2_monomials.png
# ---
# height: 320px
# name: non_linear_transformation_fig
# align: left
# ---
# Non-linear transformation $\phi: \mathbb{R}^2 \rightarrow \mathbb{R}^3 $ mapping feature vectors $ {\bf x} = \begin{bmatrix} x_1 & x_2 \end{bmatrix}^{T} $ into non-linear feature vectors $ {\bf z} = \begin{bmatrix} z_1 & z_2 & z_3 \end{bmatrix}^{T} $ such that $ {\bf z} = \phi({\bf x}) := \begin{bmatrix} x_1^2 & \sqrt{2} x_1 x_2 & x_2^2 \end{bmatrix}^{T} $. The original bi-dimensional feature space in the left is unsuitable for a linear classifier. However, the non-linear transformation of features yields to a three-dimensional feature space in the right in which the data is linearly separable for some classifier of the type $ h({\bf z}; {\bf w}, b) = \sgn({\bf w}^{T} {\bf z} - b) $ (borrowed from {cite}`scholkopfmax2013max`).
# ```
# 
# ````{margin}
# ```{note}
# Therefore, a linear classifier operating on a non-linear feature space leads to a non-linear classifier. Schematically, $$ \mbox{non-linear features} + \mbox{linear classifier} = \mbox{non-linear classifier}. $$ 
# ```
# ````
# 
# Unfortunately, there are too many non-linear transforms to the features. In addition to this, selecting a non-linear transform $\phi: {\cal X} \rightarrow {\cal Z} $ that leads to a non-linear feature space $ {\cal Z} $ in which the transformed data from $ {\cal X} $ is linearly separable is not a straightforward task. Furthermore, for a given class of non-linear transformations, the number of possible features might quickly explode. For example, let us consider all polynomials of degree $K$. Thus, there are ${D + K - 1}\choose {K}$ possible features to select -- with $D$ denoting the number of features in the original feature space $ {\cal X}$ -- when designing the non-linear feature space ${\cal Z}$. In particular, $D=100$ and $K=5$ yields $75 \times 10^6$ possible features. Therefore, checking all possible combinations of non-linear features to select a suitable non-linear transform $\phi$ is prohibitive.
# 
# Fortunately, many learning algorithms can be re-formulated such that they work only with labels $ y_{1}, y_{2}, \ldots, y_{N} $ and **inner products** $ {\bf x}_{i}^{T} {\bf x}_{j} $. For those algorithms, we can employ the **Kernel Trick** to efficiently work with high-dimensional features spaces without explicitly transforming the original features.
# 
# More precisely, let us define a $ N \times N $ pairwise similarity matrix $ {\bf K} \triangleq \left[ K_{i,j} \right] $ -- a.k.a. Gram matrix -- such that
# ```{math}
# :label: pairwise_similarity1
# \begin{eqnarray}
# K_{i,j} &=& k({\bf x}_{i}, {\bf x}_{j}) \\
# &=& \phi({\bf x}_{i})^{T} \phi({\bf x}_{j})
# \end{eqnarray}
# ```
# $ \forall i,j \lbrace 1, 2, \ldots, N \rbrace $. As $$ \phi({\bf x}_{i})^{T} \phi({\bf x}_{j}) = 0 \Leftrightarrow \phi({\bf x}_{i}) \perp \phi({\bf x}_{j}), $$ the function $ k({\bf x}_{i}, {\bf x}_{j}) $ is a similarity measure -- a scalar -- of the transformed, non-linear feature vectors $ \phi({\bf x}_{i}) $ and $ \phi({\bf x}_{j}) $. We can also define the similarity measure between the training samples $ {\cal D}_{i} = \left( {\bf x}_{i}, y_{i} \right) $ and $ {\cal D}_{j} = \left( {\bf x}_{j}, y_{j} \right) $ as
# ```{math}
# :label: pairwise_similarity2
# \begin{eqnarray}
# g({\cal D}_{i}, {\cal D}_{j}) &=& y_{i} y_{j} \phi({\bf x}_{i})^{T} \phi({\bf x}_{j}) \\
# &=&  y_{i} y_{j} k({\bf x}_{i}, {\bf x}_{j}).
# \end{eqnarray}
# ```
# Note that for normalized feature vectors $ \phi({\bf x}_{i}) $ and $ \phi({\bf x}_{j}) $, $ k({\bf x}_{i}, {\bf x}_{j}) \geq 0 $. Thus, the sign of the product $ y_{i} y_{j} $ indicates either a label matching ($ +1 $) or mismatching ($ -1 $) in binary classification problems with labels in $ {\cal Y} = \lbrace -1, +1 \rbrace $. In this case, we can redefine the Gram matrix as a $ N \times N $ influence matrix $ {\bf G} = \left[ G_{i,j} \right] $ such that $$ G_{i,j} = g({\cal D}_{i}, {\cal D}_{j}) $$ for all $ i,j \in \lbrace 1, 2, \ldots, N \rbrace $.
# 
# Now, let us rewrite the dual SVM problem in the non-linear feature space $ {\cal Z} $ as
# ```{math}
# :label: svm_form12
# \begin{eqnarray}
# \maximize_{\boldsymbol{\lambda} \geq {\bf 0}}  &&  - \frac{1}{2} \sum_{i=1}^{N} \sum_{j=1}^{N} \lambda_{i} \lambda_{j} y_{i} y_{j} \phi({\bf x}_{i})^{T} \phi({\bf x}_{j}) + \sum_{i=1}^{N} \lambda_{i} \\
# s.t. \,\,\, &&  \sum_{i=1}^{N} \lambda_{i} y_{i} = 0 \\
# &&  0 \leq \lambda_{i} \leq C, \,\,\, \forall i \in \lbrace 1, 2, \ldots, N \rbrace.
# \end{eqnarray}
# ```
# Thus, as far as we are able to compute the similarity metric $ k({\bf x}_{i}, {\bf x}_{j}) $ implicitly, we can efficiently solve the optimization problem in {eq}`svm_form12` without computing the non-linear features $ \lbrace \phi({\bf x}_{i}) \rbrace $, $ i \in \lbrace 1, 2, \ldots, N \rbrace $. Fortunately, this possible as far as the the kernel $ k({\bf x}_{i}, {\bf x}_{j}) $ satisfies some mild conditions.
# 
# Now, we offer without proof a key theorem for the Kernel Trick.
# ```{prf:theorem} Representer theorem
# A kernel function $ k: {\cal X} \times {\cal X} \rightarrow \mathbb{R} $ is positive definite i.f.f. it corresponds to the inner product in some feature space $ {\cal Z} $ defined by the transformation $ \phi: {\cal X} \rightarrow {\cal Z} $, i.e. $$ \exists \phi({\bf x}) \mid k({\bf x}_{i}, {\bf x}_{j}) = \phi({\bf x}_{i})^{T} \phi({\bf x}_{j}) \Leftrightarrow \forall {\bf a} \in \mathbb{R}^{N}  \mid {\bf a}^{T} {\bf K} {\bf a} \geq 0 $$ with $ {\bf K} = \left[ K_{i,j} \right] $ denoting a $ N \times N $ matrix such that $ K_{i,j} = k({\bf x}_{i}, {\bf x}_{j}) $, $ \forall i,j \in \lbrace 1, 2, \ldots, N \rbrace $.
# ```
# As a corollary, if we choose a particular function $ k({\bf x}_{i}, {\bf x}_{j}) $ such that $ \sum_{i=1}^{N} \sum_{j=1}^{N} a_{i} a_{j} {\bf x}_{i}^{T} {\bf x}_{j} \geq 0 $ for all data items $ i,j \in \lbrace 1, 2, \ldots, N \rbrace $ in your training dataset $ {\cal D} $, we can compute the inner product $ \phi({\bf x}_{i})^{T} \phi({\bf x}_{j}) $ implicitly without even knowing the non-linear transformation $ \phi({\bf x}) $. Putting in other words, if the matrix $ {\bf K} $ is positive definite such that {eq}`positive_definite` holds, we can replace the the product $ \phi({\bf x}_{i})^{T} \phi({\bf x}_{j}) $ in {eq}`svm_form12` by the kernel $ k({\bf x}_{i}, {\bf x}_{j}) $ to obtain the kernelized SVM formulation
# ```{math}
# :label: svm_form13
# \begin{eqnarray}
# \maximize_{\boldsymbol{\lambda} \geq {\bf 0}}  &&  - \frac{1}{2} \sum_{i=1}^{N} \sum_{j=1}^{N} \lambda_{i} \lambda_{j} y_{i} y_{j} k({\bf x}_{i}, {\bf x}_{j}) + \sum_{i=1}^{N} \lambda_{i} \\
# s.t. \,\,\, &&  \sum_{i=1}^{N} \lambda_{i} y_{i} = 0 \\
# &&  0 \leq \lambda_{i} \leq C, \,\,\, \forall i \in \lbrace 1, 2, \ldots, N \rbrace
# \end{eqnarray}
# ```
# which is still a quadratic program (convex) and therefore the global optimal solution $ \boldsymbol{\lambda}^{\ast} $ to this dual problem also leads to an optimal solution to the primal problem in {eq}`svm_form_7`. Note that the Kernel Trick allows one to generalize several learning algorithms which rely on inner products (e.g. SVMs, PCAs and LDAs). Moreover, it also allows one to work with *non-vector* data (e.g. strings and graphs) by means of the selection of a proper kernel / similarity measure $ k({\bf x}_{i}, {\bf x}_{j}) $ between the data items / objects within the original object space $ {\cal X} $.
# 
# Typical kernel functions employed
# * **Linear** $ k({\bf x}_{i}, {\bf x}_{j}) = {\bf x}_{i}^{T} {\bf x}_{j} $, which corresponds to working in the original feature space, i.e. $ \phi({\bf x}) = {\bf x} $;
# * **Polynomial** $ k({\bf x}_{i}, {\bf x}_{j}) = (1 + {\bf x}_{i}^{T} {\bf x}_{j})^{p}$, which corresponds to $\phi$ mapping to all polynomials -- non-linear features -- up to degree $p$;
# * **Gaussian** $ k({\bf x}_{i}, {\bf x}_{j}) = \exp(-\gamma || {\bf x}_{i} - {\bf x}_{j} ||^{2}) $, which corresponds in turn to an infinite feature space or functional space with infinite number of dimensions.
# 
# Recall from {eq}`w_optimal` and {eq}`b_optimal` that
# ```{math}
# :label: w_optimal2
# {\bf w}^{\ast} = \sum_{i=1}^{N} \lambda^{\ast}_{i} y_{i} {\bf x}_{i}
# ```
# ```{math}
# :label: b_optimal2
# b^{\ast} = {{\bf w}^{\ast}}^{T} {\bf x}_{i} - y_{i}.
# ```
# Now, by substituting $ {\bf x}_{i} $ by $ \phi({\bf x}_{i} ) $ in {eq}`w_optimal2` and in {eq}`b_optimal2`, we have
# ```{math}
# :label: w_optimal3
# {\bf w}^{\ast} = \sum_{i=1}^{N} \lambda^{\ast}_{i} y_{i} \phi({\bf x}_{i})
# ```
# ```{math}
# :label: b_optimal3
# b^{\ast} = {{\bf w}^{\ast}}^{T} \phi({\bf x}_{i}) - y_{i}.
# ```
# Lastly, by plugging {eq}`w_optimal3` into {eq}`b_optimal3`, we can compute the optimal offset term
# \begin{eqnarray}
# b^{\ast} &=& \left( \sum_{j=1}^{N} \lambda^{\ast}_{j} y_{j} \phi({\bf x}_{j}) \right)^{T} \phi({\bf x}_{i}) - y_{i} \nonumber \\
# &=& \sum_{j=1}^{N} \lambda^{\ast}_{j} y_{j} \underbrace{\phi({\bf x}_{j})^{T} \phi({\bf x}_{i})}_{k({\bf x}_{j}, {\bf x}_{i})} - y_{i} \nonumber \\
# &=& \sum_{j=1}^{N} \lambda^{\ast}_{j} y_{j} k({\bf x}_{j}, {\bf x}_{i}) - y_{i}
# \end{eqnarray}
# using any training example $ {\cal D}_{i} = \left( {\bf x}_{i}, y_{i} \right) $, $ i \in \lbrace 1, 2, \ldots, N \rbrace $, whose corresponding Lagrange multiplier $ \lambda^{\ast}_{i} > 0 $.
# 
# Note however that there is no need to compute the optimal coefficient vector $ {\bf w}^{\ast} $ explicitly. Specifically, for an arbitrary feature vector $ {\bf x} $ in the original feature space $ {\cal X} $, the Kernelized SVM classifier in the non-linear feature space $ {\cal Z} $ -- determined by some mapping function $ \phi:{\cal X} \rightarrow {\cal Z} $ -- can be written as
# \begin{eqnarray}
# h_{kSVM}({\bf x}) &=& \sgn \left( {{\bf w}^{\ast}}^{T} \phi({\bf x}) - b^{\ast} \right) \nonumber \\
# &=& \sgn \left( \left( \sum_{i=1}^{N} \lambda^{\ast}_{i} y_{i} \phi({\bf x}_{i}) \right)^{T} \phi({\bf x}) - b^{\ast} \right) \nonumber \\
# &=& \sgn \left( \sum_{i=1}^{N} \lambda^{\ast}_{i} y_{i} \underbrace{\phi({\bf x}_{i})^{T} \phi({\bf x})}_{k({\bf x}_{i}, {\bf x})} - b^{\ast} \right). \nonumber
# \end{eqnarray}
# Finally leading to
# ```{math}
# :label: kerneliezed_svm
# h_{kSVM}({\bf x}) = \sgn \left( \sum_{i=1}^{N} \lambda^{\ast}_{i} y_{i} k({\bf x}_{i}, {\bf x}) - b^{\ast} \right).
# ```
# 
# ````{prf:example} Kernelized SVM classifier in action using a toy example with *sklearn*
# Figures below illustrate the effect of different Kernels with *sklearn* using a toy binary classification example. {numref}`toyexp_kernel_svm_01_fig` shows a linear kernel failing miserably to separate training data items assigned to <span style="color: blue;">blue</span> and <span style="color: red;">red</span> class labels; {numref}`toyexp_kernel_svm_02_fig` a polynomial kernel with degree $p=4$ is able to circumscribe the data items assigned to the <span style="color: blue;">blue</span> labels; and {numref}`toyexp_kernel_svm_03_fig` a Gaussian kernel -- a.k.a. radial basis function -- is also able to separate both classes, but using kind of more rounded decision boundary between classes.
# 
# ```{figure} /images/classification/toyexp_kernel_svm_01.png
# ---
# height: 200px
# name: toyexp_kernel_svm_01_fig
# align: left
# ---
# classifier = svm.SVC(kernel='linear', C=C)
# ```
# 
# ```{figure} /images/classification/toyexp_kernel_svm_02.png
# ---
# height: 200px
# name: toyexp_kernel_svm_02_fig
# align: left
# ---
# classifier = svm.SVC(kernel='poly', degree=4, C=C)
# ```
# 
# ```{figure} /images/classification/toyexp_kernel_svm_03.png
# ---
# height: 200px
# name: toyexp_kernel_svm_03_fig
# align: left
# ---
# classifier = svm.SVC(kernel='rbf', C=C)
# ```
# ````
# 
# ````{margin}
# ```{note}
# SVMs and Kernels were hot research topics in the 90's and early 2000's. Nevertheless, Kernelized SVMs still one of the strongest classifiers today.
# ```
# ````
