#!/usr/bin/env python
# coding: utf-8

# ## Classification Objective
# Classification is a supervised learning task where the goal is to predict a discrete class label for each input. A typical toy dataset in classification is the Iris dataset, where the task is to classify flower  species based on physical features of the plant.

# In[1]:


from sklearn.datasets import load_iris
import pandas as pd

# Load the dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target, name="target")

# Preview the dataset
X.head()


# The Iris dataset contains 150 data points, having 4 features: sepal length, sepal width, petal length, and petal width. The target classes are:
# 
# * $y=0$: setosa
# * $y=1$: versicolor
# * $y=2$: virginica
# 
# All features in the Iris dataset are continuous numerical values, as they represent physical measurements of the flower in centimeters. However, classification datasets may also have discrete features such as male/female or color (green, red, blue).     
# 
# ### Classification Challenges
# Even though the Iris dataset is relatively “clean,” it still allows us to discuss core challenges in classification tasks.
# 
# #### Class Imbalance
# In many real-world problems, one class dominates the others. Let's check the class distribution in Iris:

# In[2]:


y.value_counts(normalize=True)


# Here, each class occurs exactly 50 times (1/3 of the data), making the Iris dataset a perfectly balanced classification dataset. Having one class dominate over others may result in classifiers that predict the majority class in most cases and that are also correct most of the time with that prediction. Hence, the evaluation must be performed carefull in those cases, and maybe the classifier needs to be adapted to this case as well.  
# #### Mixed Feature Types
# While the Iris dataset only contains continuous numerical features, many real-world classification problems involve discrete or categorical features, which introduce their own set of challenges.
# In many datasets, features can be discrete — meaning they take on a finite set of values, such as:
# 
# * Categorical/Nominal: e.g., "color" = red, green, blue"
# * Ordinal: e.g., "education level" = high school < college < graduate"
# * Binary: e.g., "has_credit_card" = yes/no"
# 
# Categorical and binary data has no meaningful $\leq$ operation and this requires special treatment. For example, simple statistics such as computing the mean and variance don't have a meaning for categorical data. Not all classifiers can handle all feature types, hence it's important to match the classifier to the feature types in the data. 

# ### Classifier definition
# 
# `````{admonition} Task (Classification)
# :class: tip
# **Given** a dataset consisting of $n$ observations $\vvec{x}_i$ and their corresponding labels $y_i$ indicating one of $c$ classes
#     $$\mathcal{D}=\left\{(\vvec{x}_{i},y_i)\vert \vvec{x}_{i}\in\mathbb{R}^{d},y_i\in\{1,\ldots, c\}, 1\leq i \leq n\right\}$$
# 
# **Find** a classifier $f:\mathbb{R}^ d\rightarrow \mathbb{R}^c$, that captures the relationship between observations and their class. The classifier predicts the label with the maximum value:  
# $$\hat{y}_i = \argmax_{l\in\{1,\ldots,c\}}f(\vvec{x}_{i})_l.$$ 
# The goal is to find a classifier that predicts the correct labels $\hat{y}_i = y_i$.
# `````
# 
# Classifiers, similar to regression models, are defined by their inference and their training. Inference describes how the model performs prediction of (unseen) data points. The training or learning describes how the model is generated, given the training data. 
# ### Evaluation
# We quickly define the most straightforward classification evaluation metrics: the $L_{01}$-loss and the accuracy. Both put into relation how many errors/correct predictions a classifier makes in a dataset. 
# 
# ```{prf:definition} 0-1 Loss
# Given a classifier $\hat{y}(\vvec{x})$ returning the predicted label. We define the **0-1 loss** as
# $$L_{01}(y,\hat{y}) = \begin{cases}
# 1, & \text{if } y\neq \hat{y}\\
# 0, & \text{if } y=\hat{y}
# \end{cases}$$
# ```
# The 0-1 loss indicates whether a classifier makes an error in its prediction. In contrast, the accuracy indicates how much a classifier gets right.
# ```{prf:definition} Accuracy
# Given a classifier $\hat{y}(\vvec{x})$returning the predicted label. the accuracy of the classifier on dataset $\mathcal{D}$ containing $n$ data points is given as
# 
# \begin{align*}
# \mathrm{Acc}(\hat{y},\mathcal{D}) &= \frac{\text{Correct predictions}}{\text{Total predictions}}\\
# &= \frac1n \lvert\{(\vvec{x}_i,y_i)\in\mathcal{D}\mid \hat{y}(\vvec{x}_i)=y_i\}\rvert\\
# &= 1- \frac1n\sum_{i=1}^n L_{01}(\hat{y}(\vvec{x}_i),y_i).
# \end{align*}
# ```
# ### Theoretically Optimal Classifiers
# Similarly to the regression data sampling process according to a true regression function plus some noise, we also have some assumptions about how our classification data is sampled and what the ground truth classifier is that we want to recover.
# ````{prf:property} i.i.d. Class Distribution
# Under the i.i.d. class distribution assumption, we assume that the dataset samples are _identically_ distributed and _independently_ drawn from an _unknown_ probability distribution $ p^{\ast}({\bf x}, y) $, i.e. 
# ```{math}
# :label: iid_assumption
# {\bf x}_{i}, y_{i} \sim  p^{\ast}({\bf x}, y), \forall i \in \lbrace 1, \ldots, n \rbrace.
# ```
# ````
# Likewise, we can define the expected prediction error for a classification problem.
# ````{prf:definition} Classifier EPE
# :label: true_classifier_error
# Given a classifier $f_\mathcal{D}:\mathbb{R}^d\rightarrow \mathbb{R}^c$ that has been trained on dataset $\mathcal{D}$. 
# the Expected Prediction Error (EPE) of classifiers is the expected error
# ```{math}
# :label: true_classification_error
# p( y \neq \argmax_{l} f(\vvec{x})_l ) = \mathbb{E}_{\vvec{x},y,\mathcal{D}} [L_{01} (y, \argmax_l f_\mathcal{D}(\vvec{x}) )_l] ,
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
# The probability of misclassification $ p( y \neq \argmax_l f(\vvec{x})_l) $ is also known as the risk $ R(f) $ of the classifier $ f $.
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
# 
# 

# In[ ]:




