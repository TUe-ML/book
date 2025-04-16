#!/usr/bin/env python
# coding: utf-8

# ## Classification Objective
# 
# ### Feature space
# 
# Let $ {\bf x} $ denote a feature vector residing in a multidimensional space $ {\cal X} $ comprising several  features. We describe the data items -- i.e. the objects we are trying to classify -- by feature vectors in $ {\cal X} $. Note that the vector $ {\bf x} \in {\cal X} $ describing a given data item may comprise either continuous (e.g. length, height, weight) or discrete (e.g. color code) features.
# 
# 
# ### Class labels
# 
# Moreover, let $ y $ be a discrete variable indicating the class label (target) among the set of possible labels $ {\cal Y} $, where $ {\cal Y} \triangleq \lbrace \ell_{1}, \ell_{2}, \ldots, \ell_{L} \rbrace $ is a finite set -- a.k.a. alphabet -- of $ L $ arbitrary labels. The variable $ y \in {\cal Y} $ indicates thus a class value (label) assigned to a data item. A label $ \ell $ can be either the class name or a unique identification number / code representing the class. However, for the sake of simplicity, the class labels are usually normalized as indexes spanning e.g. from $ 0 $ to $ L-1 $. 
# 
# 
# ```{prf:remark}
# We should pick up the set of features according to the particular classification problem we are trying to solve and also make sure that it really encloses sufficient information to help distinguishing data items / objects from different classes. Note therefore that the selection of classes and the selection of features are often entangled.
# ```
# 
# ### Training examples
# 
# Now let
# \begin{eqnarray}
# {\cal D} &\triangleq& \bigcup_{i=1}^{N} \lbrace \left( {\bf x}_{i}, y_{i} \right) \rbrace \\
# &\equiv& \lbrace \left( {\bf x}_{1}, y_{1} \right), \left( {\bf x}_{2}, y_{2} \right), \ldots, \left( {\bf x}_{N}, y_{N} \right) \rbrace \nonumber
# \end{eqnarray}
# denote a dataset of $ N $ training examples, i.e. the data items for which we already know the true classification. Specifically, the $ i $-th _known_ training example is represented by a pair $ {\cal D}_{i} \triangleq \left( {\bf x}_{i}, y_{i} \right) $ including both the feature vector $ {\bf x}_{i} $ and the associated class label $ y_{i} $. 
# 
# ```{prf:remark}
# As we increase the number of features (more dimensions) or the granularity of the existing features (e.g. more values per discrete feature), the classifier might increase its ability to distinguish training examples from different classes. However, besides possibly increasing the computational burden to deal with all input features, the required number of training examples might significantly increase to prevent over fitting.
# ```
# 
# ### Classifier definition
# 
# `````{admonition} Task (Classification)
# :class: tip
# **Given** a dataset consisting of $n$ observations $\vvec{x}_i$ and their corresponding labels $y_i\in\{1,\ldots, c\}$ belonging to one of $c$ classes
#     $$\mathcal{D}=\left\{(\vvec{x}_{i},y_i)\vert \vvec{x}_{i}\in\mathbb{R}^{d}, 1\leq i \leq n\right\}$$
# 
# **Find** a classifier $f:\mathbb{R}^ d\rightarrow \mathbb{R}^c$, that captures the relationship between observations and their class. The classifier predicts the label with the maximum value:  
# $$\hat{y}_i = \argmax_{k\in\{1,\ldots,c\}}f(\vvec{x}_{i})_k.$$ 
# The goal is to find a classifier that predicts the correct labels $\hat{y}_i = y_i$.
# `````
# 
# Classifiers, similar to regression models, are defined by their inference and their training. Inference describes how the model performs prediction of (unseen) data points. The training or learning describes how the model is generated, given the training data. 
# ### Theoretical Optimality of a Classifier
# ````{prf:property} i.i.d. Class Distribution
# Under the i.i.d. class distribution assumption, we assume that the dataset samples are _identically_ distributed and _independently_ drawn from an _unknown_ probability distribution $ p^{\ast}({\bf x}, y) $, i.e. 
# ```{math}
# :label: iid_assumption
# {\bf x}_{i}, y_{i} \sim  p^{\ast}({\bf x}, y), \forall i \in \lbrace 1, \ldots, n \rbrace.
# ```
# ````
# 
# 
# 
# ````{prf:definition} Classifier EPE
# :label: true_classifier_error
# Given a classifier $f_\mathcal{D}:\mathbb{R}^d\rightarrow \mathbb{R}^c$ that has been trained on the dataset $\mathcal{D}$. We define the 0-1 loss as
# $$L_{01}(y,\hat{y}) = \begin{cases}
# 0, & \text{if } y\neq \hat{y}\\
# 1, & \text{if } y=\hat{y}
# \end{cases}$$
# the Expected Prediction Error (EPE) of classifiers is the expected error
# ```{math}
# :label: true_classification_error
# p( y \neq \argmax_{k} f(\vvec{x})_k ) = \mathbb{E}_{\vvec{x},y,\mathcal{D}} [L (y, \argmax_k f_\mathcal{D}(\vvec{x}) )] ,
# ```
# over three random variables:
# * $\vvec{x}$ is the random variable of a feature vector in the test set.
# * $y$ is the random variable of the class of $\vvec{x}$.
# * $\mathcal{D}$ is the random variable of the training data.
# ````
# 
# 
# 
# ```{note}
# The probability of miss-classification $ p( y \neq \argmax_k f(\vvec{x})) $ is also known as the risk $ R(f) $ of the classifier $ f $.
# ```
# 
# ````{prf:definition} Bayes optimal classifier
# :label: bayes_optimal_classifier
# The **Bayes classifier** is the optimal classifier that minimizes the probability of misclassification 
# ```{math}
# :label: final_estimator
# y^\ast(\vvec{x}) = \argmax_{1\leq y\leq c} p^\ast (y\mid \vvec{x})
# ```
# ````
# The Bayes classifier has the lowest EPE possible. 
# `````{toggle}
# ````{prf:proof}
# From {eq}`true_classification_error`, we can write
# ```{math}
# :label: bayes_optimal_classifier
# h^{\ast}({\bf x}) = \argmin_{h} {E}_{{\bf X},Y \sim p^{\ast}} \lbrace \left[ Y \neq h(\bf X) \right] \rbrace
# ```
# where the argument $ h $ resides on the space of arbitrary functions of the type $ {\cal X} \rightarrow {\cal Y} $.
# 
# We offer without proof that the solution to {eq}`bayes_optimal_classifier` is given by the maximum likelihood (ML) estimator 
# ```{math}
# :label: ml_estimator
# h^{\ast}({\bf x}) = \argmax_{y \in {\cal Y}} p^{\ast} ( y \mid {\bf x}),
# ```
# where $ p^{\ast} ( y \mid {\bf x}) $ is the true likelihood function. In this case, it indicates how likely a particular label $ y $ represents the true class value given that a feature vector $ {\bf x} $ was observed.
# 
# Note that we can multiply the right-hand side of {eq}`ml_estimator` by the true marginal distribution $ p^{\ast}({\bf x}) $ without affecting the result
# \begin{eqnarray}
# h^{\ast}({\bf x}) &=& \argmax_{y \in {\cal Y}} p^{\ast} ( y \mid {\bf x}) \, p^{\ast}({\bf x}) \nonumber \\
# &=& \argmax_{y \in {\cal Y}} p^{\ast} ({\bf x}, y) \nonumber
# \end{eqnarray}
# and finally write Eq. {eq}`final_estimator`. $ \blacksquare $
# ````
# `````
# 
# ````{margin}
# ```{note}
# As $ p^{\ast} $ is typically unknown in must practical applications, suitable learning algorithms must rely on different approaches aiming to minimize computable approximations to the true classification error and are therefore sub-optimal in a Bayesian sense.
# ```
# ````
# 
# ```{prf:definition} Excess risk
# The excess risk of a given classifier $ h $ is given by $ R(h) - R(h^{\ast}) $, where $ R(h^{\ast}) $ is the risk of the Bayes optimal classifier $ h^{\ast} $. The excess risk of a consistent classifier $ h $ converges to zero as the number of training examples $ N $ grows unbounded, i.e. $ N \rightarrow \infty $.
# ```

# In[ ]:




