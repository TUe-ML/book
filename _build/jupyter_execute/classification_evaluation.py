#!/usr/bin/env python
# coding: utf-8

# ## Evaluation  
# This sections deals with practical issues regarding the evaluation of classifiers.
# 
# ### Estimating the true classification error
# 
# Recall that the true classification error of a classifier $ h $ is defined as
# :::{math}
# :label: true_classification_error2a
# CE_{true} = Pr \lbrace Y \neq h({\bf X}) \rbrace.
# :::
# By definition $ Pr \lbrace Y \neq h({\bf X}) \rbrace \triangleq {E}_{{\bf X},Y \sim p^{\ast}} \lbrace \left[ Y \neq h(\bf X) \right] \rbrace $. Thus, we can rewrite {eq}`true_classification_error2a` as
# :::{math}
# :label: true_classification_error2b
# CE_{true} = {E}_{{\bf X},Y \sim p^{\ast}} \lbrace \left[ Y \neq h(\bf X) \right].
# :::
# 
# Without loss of generality, let us assume that the multidimensional feature space $ {\cal X} $ includes only continuous-valued features. Therefore, the expected value of the function $ \left[ Y \neq h(\bf X) \right] $ of $ {\bf X},Y $ w.r.t. the *mixed* probability distribution $ p^{\ast} $ can be written as
# :::{math}
# :label: expected_value
# {E}_{{\bf X},Y \sim p^{\ast}} \lbrace \left[ Y \neq h(\bf X) \right] \rbrace = \sum_{y \in {\cal Y}} \int_{{\bf x} \in {\cal X}} \left[ y \neq h({\bf x}) \right] p^{\ast} ({\bf x}, y) d {\bf x}.
# :::
# Note, however, that the underlying distribution $ p^{\ast} $ is *unknown*. Thus, in general, the true classification error can not be computed exactly as in {eq}`true_classification_error2b`. Even if we knew the true distribution $ p^{\ast} $, the integrals / summations in {eq}`expected_value` have often no closed form solution.
# 
# In practice, we approximate the true classification error by computing a Monte Carlo approximation of the integrals /summations on the right-hand side of {eq}`expected_value`. Specifically, let
# \begin{eqnarray}
# {\cal D}' &\triangleq& \bigcup_{i=1}^{M} \lbrace \left( \bar{\bf x}_{i}, \bar{y}_{i} \right) \rbrace \nonumber \\
# &\equiv& \lbrace \left( \bar{\bf x}_{1}, \bar{y}_{1} \right), \left( \bar{\bf x}_{2}, \bar{y}_{2} \right), \ldots, \left( \bar{\bf x}_{M}, \bar{y}_{M} \right) \rbrace \nonumber
# \end{eqnarray}
# be a dataset containing $ M $ *testing* or *validation* examples also drawn from the underlying distribution $ p^{\ast} $. Note that we know the true label $ \bar{y}_{i} $ of each data item $ \bar{\bf x}_{i} $. From the law of large numbers, the integrals/summations on the right-hand side of {eq}`expected_value` can be computed as
# :::{math}
# :label: expected_value_approximation
# \sum_{y \in {\cal Y}} \int_{{\bf x} \in {\cal X}} \left[ y \neq h(\bf x) \right] p^{\ast} ({\bf x}, y) d {\bf x} = \lim_{M \rightarrow \infty} \frac{1}{M} \sum_{i=1}^{M} \left[ \bar{y}_{i} \neq h(\bar{\bf x}_{i}) \right]
# :::
# with $ \bar{\bf x}_{i}, \bar{y}_{i} \sim p^{\ast} ({\bf x}, y) $, $ \forall M \in \lbrace 1, 2, 3, \ldots \rbrace $, since by definition $$ \sum_{y \in {\cal Y}} \int_{{\bf x} \in {\cal X}} p^{\ast} ({\bf x}, y) d {\bf x} = 1. $$
# 
# Thus, for a finite number of samples $ M $, the true classification error is typically approximated in a Monte Carlo sense using a very large *testing* or *validation* dataset $ {\cal D}' $ as
# :::{math}
# CE_{true} \approx \frac{1}{M} \sum_{i=1}^{M} \left[ \bar{y}_{i} \neq \hat{y}_{i} \right]
# :::
# with $ \hat{y}_{i} = h(\bar{\bf x}_{i}) $.
# 
# ::::{tip}
# Using the training dataset to estimate true estimation error can lead to over fitting. Specifically, using the *training* examples to estimate the true classification error is too optimistic. A small training error does not imply good generalization, as the classifier can still perform poorly on samples not seen during training. To avoid over-fitting, we reserve therefore examples in a *testing* dataset to estimate the true classification error after training. This procedure avoids over-fitting since the *testing* examples are new to the classifier, i.e. they were not used in training. In addition to this, we can also reserve examples in a *validation* dataset to estimate the true classification error and tune any hyper-parameters of the classifier during training. In general, we split the set of available samples into training, validating and testing datasets using some proportion (e.g. $ 60 / 20 / 20 $ or $ 70/15/15 $).
# 
# :::{figure} /images/classification/train_val_test_datasets.svg
# ---
# height: 320px
# name: training_testing_fig
# align: left
# ---
# Splitting the available samples into training, validation and testing datasets.
# :::
# 
# Thus, for a given classifier $ h(\cdot) $, we can estimate the training, validation and testing classification errors respectively as 
# \begin{eqnarray}
# CE_{train} &=& \frac{1}{|{\cal D}_{train}|} \sum_{\left(\bar{\bf x}, \bar{y} \right) \in {\cal D}_{train} } \left[ \bar{y} \neq h(\bar{\bf x}) \right] \nonumber \\
# CE_{val} &=& \frac{1}{|{\cal D}_{val}|} \sum_{\left(\bar{\bf x}, \bar{y} \right) \in {\cal D}_{val} } \left[ \bar{y} \neq h(\bar{\bf x}) \right] \nonumber \\
# CE_{test} &=& \frac{1}{|{\cal D}_{test}|} \sum_{\left(\bar{\bf x}, \bar{y} \right) \in {\cal D}_{test} } \left[ \bar{y} \neq h(\bar{\bf x}) \right]. \nonumber 
# \end{eqnarray}
# Note however that the training error $ CE_{train} $ and validation error $ CE_{val} $ are often too optimistic - especially, the training error -- whereas the testing error $ CE_{test} $ is a noisy estimation of the true classification error $ CE_{true} $ such that $ CE_{test} \rightarrow CE_{true} $ as the number of testing examples grows unbounded, i.e. $ | CE_{test} | \rightarrow \infty $.
# ::::
# 
# :::{prf:remark}
# Some works report the classification accuracy -- opposite of error -- instead of the classification error. The true accuracy is defined as 
# \begin{eqnarray}
# ACC_{true} &\triangleq& Pr \lbrace Y = h({\bf X}) \rbrace \label{eq:true_classification_accuracy} \\
# &\equiv&  1 - CE_{true}. \nonumber
# \end{eqnarray}
# On the other hand, the training, validation and testing classification accuracies of a classifier $ h(\cdot) $ can be computed respectively as 
# \begin{eqnarray}
# ACC_{train} &=& \frac{1}{|{\cal D}_{train}|} \sum_{\left(\bar{\bf x}, \bar{y} \right) \in {\cal D}_{train} } \left[ \bar{y} = h(\bar{\bf x}) \right] \nonumber \\
# ACC_{val} &=& \frac{1}{|{\cal D}_{val}|} \sum_{\left(\bar{\bf x}, \bar{y} \right) \in {\cal D}_{val} } \left[ \bar{y} = h(\bar{\bf x}) \right] \nonumber \\
# ACC_{test} &=& \frac{1}{|{\cal D}_{test}|} \sum_{\left(\bar{\bf x}, \bar{y} \right) \in {\cal D}_{test} } \left[ \bar{y} = h(\bar{\bf x}) \right]. \nonumber 
# \end{eqnarray}
# :::
# 
# ### Typical Machine Learning (ML) workflow
# 
# {numref}`train_validate_test_02` illustrates the typical procedure to learn a ML model. The training procedure under the **Repeat** rectangle is repeated over several training epochs considering different selections of the model hyper-parameters. Intermediate statistics -- training and validation errors -- are collected along the training epochs. Final statistics -- the testing error -- are computed for the model with the best validation error. Training and validation datasets may be also shuffled / randomized over the training epochs. The testing dataset in turn shall not play any role on the training procedure.
# 
# :::{figure} /images/classification/train_validate_test_02.png
# ---
# height: 320px
# name: train_validate_test_02
# align: left
# ---
# A block diagram representing a typical ML workflow.
# :::
# 
# ::::{danger}
# ** Avoid human-in-the-loop!** Dot not adjust the model parameters or hyper-parameters by guessing based on the testing accuracy / error. Do not allow the information containing in the testing data set leak to the training procedure.
# 
# :::{figure} /images/classification/train_validate_test_03.png
# ---
# height: 320px
# name: train_validate_test_03
# align: left
# ---
# Human feedback plugging the testing dataset into the training procedure. A naive human -- thirsty for good results -- may change the model parameters / hyper-parameters to improve the training error. The resulting test error will therefore under estimate the true error leading to an over-confident model. Conversely, the model generalization ability to deal with unseen samples will be over-estimated.
# :::
# ::::
# 
# ### Confusion matrix
# 
# Let 
# \begin{equation}
# {\bf CM} = \begin{bmatrix}
# c_{1,1} & c_{1,2} & \ldots & c_{1,L} \\
# c_{2,1} & c_{2,2} & \ldots & c_{2,L} \\
# \vdots &  \vdots & & \vdots \\
# c_{L,1} & c_{L,2} & \ldots & c_{L,L}
# \end{bmatrix}
# \end{equation}
# define a $L \times L$ confusion matrix such that, $ \forall i,j \in \lbrace 1, 2, \ldots, L \rbrace $, $ c_{i,j} \in {\cal Y} \triangleq \lbrace \ell_{1}, \ell_{2}, \ldots, \ell_{L} \rbrace $ and 
# \begin{equation}
# c_{i,j} = \sum_{\left(\bar{\bf x}, \bar{y} \right) \in {\cal D}' } \left[ h(\bar{\bf x}) = \ell_{j} \wedge  \bar{y} = \ell_{i} \right]
# \end{equation}
# for some *validation* or *training* dataset $$ {\cal D}' = \bigcup_{i=1}^{M} \lbrace \left( \bar{\bf x}_{i}, \bar{y}_{i} \right) \rbrace. $$ The matrix element $ c_{i,j} $ indicates therefore how frequently the true class value associated with a data item $ \bar{\bf x} $ in $ {\cal D}' $ is $ \bar{y} = \ell_{i} $ while the classifier predicts $ h(\bar{\bf x}) = \ell_{j} $ with $ \ell_{i}, \ell_{j} \in {\cal Y} $. Note that the diagonal elements $ \lbrace c_{i,j} \rbrace $, $ i = j $, correspond to accurate predictions while off-diagonal elements $ \lbrace c_{i,j} \rbrace $, $ i \neq j $, correspond to erroneous classifier predictions.
# 
# Alternatively, the confusion matrix can be normalized such that, $ \forall i \in \lbrace 1, 2, \ldots, L \rbrace $,
# \begin{equation}
# \sum_{j=1}^{L} c_{i,j} = 1. \nonumber
# \end{equation}
# In this case, each matrix element $ c_{i,j} $ stores an empirical approximation to the probability $ Pr \left\lbrace h({\bf x}) = \ell_{j} \mid {y} = \ell_{i} \right\rbrace $ of the classifier predicting $ \ell_{j} $ given that the true class value is $ \ell_{i} $.
# 
# ::::{prf:example}
# The confusion matrix is a convenient tool for a human to quickly evaluate a classifier performance by comparing its diagonal and off-diagonal elements.
# 
# :::{figure} /images/classification/confusion_matrix.svg
# ---
# height: 320px
# name: confusion_matrix
# align: left
# ---
# A $ 4 \times 4 $ confusion matrix for a classifier $ h:{\cal X} \rightarrow {\cal Y} $ such that $ {\cal Y} = \lbrace \ell_{1}, \ell_{2}, \ell_{3} , \ell_{4} \rbrace $. The $i$-th row stores either frequencies or approximations to the empirical probabilities $ \lbrace Pr \left\lbrace h({\bf x}) = \ell_{j} \mid y = \ell_{i} \right\rbrace \rbrace $, $ j \in \lbrace 1, 2, 3, 4 \rbrace $. Note that $ Pr \left\lbrace h({\bf x}) = \ell_{j} \mid y = \ell_{i} \right\rbrace $ is the probability of the classifier $ h(\cdot) $ returning $ \ell_{j} $ given that the true class value is $ \ell_{i} $ and $ \sum_{j=1}^{4} Pr \left\lbrace h({\bf x}) = \ell_{j} \mid y = \ell_{i} \right\rbrace = 1$, $ \forall i \in \lbrace 1, 2, 3, 4 \rbrace $.
# :::
# ::::
