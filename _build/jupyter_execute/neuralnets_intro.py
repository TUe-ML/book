#!/usr/bin/env python
# coding: utf-8

# ## From Linear Models to Neural Networks
# At their core, neural networks are function approximators composed of layers of interconnected nodes that transform input data through weighted connections and nonlinear activations. Originally proposed in the 1950s with the perceptron, neural networks saw early promise but soon hit limitations due to their inability to solve even simple non-linear tasks.
# 
# The resurgence of interest began in the 1980s with the development of backpropagation, enabling multi-layer networks to be efficiently optimized. However, they were soon overshadowed by models like the kernel SVM, which were mathematically elegant, easier to train, and delivered strong performance on small to medium datasets.
# 
# Kernel methods offer strong theoretical guarantees and closed-form training objectives, but scale poorly with large datasets and require careful kernel selection. Neural networks can be seen as models that learn the feature transformation, that is implicitly provided by the kernel, themselves. Adapting both the transformation and the decision boundary simultaneously makes neural networks exceptionally powerful in high-dimensional, unstructured domains like images, audio, and text. The breakthrough came in the 2010s, fueled by larger datasets, faster GPUs, and architectural innovations (e.g., convolutional layers, residual connections), enabling deep neural networks (DNNs) to surpass traditional models on a wide range of tasks.
# 
# We start now with introducing another linear classifier, that is extended to a deep neural network classifier by a preceeding feature transformation.    
# 
# 
# ### Logistic and Softmax Regression
# 
# Logistic regression is a binary classification model where the probability of a sample belonging to the positive class is modeled as:
# 
# $$
# P(y=1 \mid \mathbf{x}) = \sigma(\mathbf{w}^\top \mathbf{x} + b)
# $$
# 
# where $\sigma(z) = \frac{1}{1 + e^{-z}}$ is the sigmoid function. This maps the linear output into a probability between 0 and 1.
# 
# For multiclass classification, the **softmax regression** generalizes logistic regression. For $K$ classes:
# 
# $$
# P(y = k \mid \mathbf{x}) = \frac{\exp(\mathbf{w}_k^\top \mathbf{x} + b_k)}{\sum_{j=1}^{K} \exp(\mathbf{w}_j^\top \mathbf{x} + b_j)}
# $$
# 
# This models class probabilities that sum to 1 and is typically trained using cross-entropy loss.
# 
# 
# ## Representation Learning
# 
# Logistic or Softmax regression rely on linear decision boundaries. In practice, data often lies on complex, nonlinear manifolds. This motivates the need for **representation learning**, where the model learns intermediate, useful transformations of the input data before making predictions.
# 
# Representation learning is central to neural networks. Instead of manually crafting features, networks **learn features jointly** with the classification task.
# 
# 
# ## Neural Networks: Architecture and Building Blocks
# 
# Neural networks stack multiple linear models interleaved with **nonlinear activation functions** to create deep, hierarchical representations.
# 
# ### 4.1 Linear Layers and Activations
# 
# The most basic neural network layer performs a linear transformation:
# 
# $$
# \mathbf{z} = \mathbf{W}\mathbf{x} + \mathbf{b}
# $$
# 
# This is followed by a **nonlinear activation function** $ \phi $, such as:
# 
# - ReLU: $ \phi(z) = \max(0, z) $
# - Sigmoid: $ \phi(z) = \frac{1}{1 + e^{-z}} $
# - Tanh: $ \phi(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}} $
# 
# A simple feedforward neural network (MLP) with one hidden layer:
# 
# $$
# \begin{align*}
# \mathbf{h} &= \phi(\mathbf{W}_1 \mathbf{x} + \mathbf{b}_1) \\
# \hat{\mathbf{y}} &= \text{softmax}(\mathbf{W}_2 \mathbf{h} + \mathbf{b}_2)
# \end{align*}
# $$
# 
# Each layer composes the previous one, enabling complex function approximation.
# 
# 
# ## Universal Approximation and the Representer Theorem
# 
# The **Universal Approximation Theorem** states that a neural network with one hidden layer and enough hidden units can approximate any continuous function on a compact domain to arbitrary accuracy.
# 
# This power comes from **nonlinearity and compositional depth**.
# 
# In contrast, the **Representer Theorem** applies in kernel methods and shows that solutions to certain regularized learning problems lie in the span of training data. While neural networks do not directly satisfy the conditions of the theorem, they similarly perform **implicit regularization** through stochastic gradient descent and architectural constraints, guiding them toward simple, generalizable functions.
# 
