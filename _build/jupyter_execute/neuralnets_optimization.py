#!/usr/bin/env python
# coding: utf-8

# ## Training Neural Networks: Backpropagation and SGD
# 
# Neural networks are trained by minimizing a loss function, often via **stochastic gradient descent (SGD)**.
# 
# ### SGD
# 
# ### Backpropagation
# 
# The **backpropagation algorithm** efficiently computes gradients of the loss with respect to all parameters using the chain rule of calculus. For a network with parameters $ \theta $, inputs $ \mathbf{x} $, and loss $ \mathcal{L} $, we compute:
# 
# $$
# \frac{\partial \mathcal{L}}{\partial \theta}
# $$
# 
# by propagating derivatives from the output layer back to earlier layers.
# 
# 
# 

# In[1]:


# make plot of loss function for logistic regression and then plot sgd trajectory based on various step sizes.
# best use pytorch and autograd otherwise it's a computational nightmare

