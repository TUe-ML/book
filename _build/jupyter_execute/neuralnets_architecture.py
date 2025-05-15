#!/usr/bin/env python
# coding: utf-8

# ## Architecture
# ## Convolutional Neural Networks (CNNs)
# 
# For high-dimensional data like images, **convolutional layers** exploit local spatial structure:
# 
# $$
# \text{Conv}(\mathbf{x}) = \sum_{i,j} \mathbf{K}_{i,j} \cdot \mathbf{x}_{i:i+k, j:j+k}
# $$
# 
# CNNs use:
# 
# - **Local connectivity** (kernels are small spatial patches)
# - **Weight sharing** (same kernel applied across the image)
# - **Pooling** to reduce dimensionality
# 
# These principles allow CNNs to be efficient and translationally invariant.
# 
# 
# ## Skip Connections and Deep Architectures
# 
# Deeper networks can suffer from **vanishing gradients** or degraded performance. **Skip connections** (or residual connections) address this by allowing the gradient to flow more directly:
# 
# $$
# \mathbf{h}_{l+1} = \phi(\mathbf{W}_l \mathbf{h}_l + \mathbf{b}_l) + \mathbf{h}_l
# $$
# 
# This enables **ResNets** and other deep architectures to train effectively, and has become a standard design choice in modern networks.
