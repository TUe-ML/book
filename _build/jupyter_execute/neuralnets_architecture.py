#!/usr/bin/env python
# coding: utf-8

# ## Architecture
# 
# The artificial neuron unit is the building block of ANNs. Thus, let us precisely model its behavior. The output value $ \hat{y} $ of an arbitrary neuron with $ D $ inputs is computed as
# :::{math}
# :label: neuron_output1
# \begin{eqnarray}
# \hat{y} &=& \phi(a) \nonumber \\
# &=& \phi \left( \sum_{i=1}^{D} w_{i} x_{i} + b \right),
# \end{eqnarray}
# :::
# where $ x_{i} $ is the input value (or feature) from the $ i $-th previous neuron, $ w_{i} $ is the weight associated with the $ i $-th input, $ b $ is a bias term, $ \phi(\cdot) $ is a non-linear activation function and $ a = \sum_{i=1}^{D} w_{i} x_{i} + b  $ is the activation. Alternatively, let the vector $ {\bf x} = \begin{bmatrix} x_{1} & \ldots & x_{D} \end{bmatrix}^{T} $ collect the input values from the $ D $ previous neurons and let $ {\bf w} = \begin{bmatrix} w_{1} & \ldots & w_{D} \end{bmatrix}^{T} $ collect the corresponding weights. We can write then the output value as
# :::{math}
# :label: neuron_output2
# \begin{equation}
# \hat{y} = \phi \left( {\bf w}^{T} {\bf x} + b \right).
# \end{equation}
# :::
# Note therefore that $ {\bf w}^{T} {\bf x} + b = 0 $ defines a hyperplane over the $D$-dimensional input space. For convenience, we can further rewrite {eq}`neuron_output1` by making the bias term implicit. More precisely, we absorb the bias into the weights by making $ w_{0} = b $ and create a new dummy input $ x_{0} = 1 $. Thus, we can write
# :::{math}
# :label: neuron_output3
# \begin{equation}
# \hat{y} = \phi \left( \sum_{i=0}^{D} w_{i} x_{i} \right).
# \end{equation}
# :::
# More compactly, we can rewrite {eq}`neuron_output3` in vector notation as
# :::{math}
# :label: neuron_output4
# \begin{equation}
# \hat{y} = \phi \left( {\bf w}^{T} {\bf x} \right),
# \end{equation}
# :::
# where the vectors were redefined such that $ {\bf x} = \begin{bmatrix} x_{0} & x_{1} & \ldots & x_{D} \end{bmatrix}^{T} $ and $ {\bf w} = \begin{bmatrix} w_{0} & w_{1} & \ldots & w_{D} \end{bmatrix}^{T} $. That is, the output value $ \hat{y} $ is an affine function $ {\bf w}^{T} {\bf x} $ of the inputs in $ {\bf x} $ (activation) followed by a non-linearity $ \phi(\cdot) $ (activation function).
# 
# {numref}`computational_graph1a` illustrates the neuron unit computation steps graphically. 
# 
# :::{figure} /images/neuralnets/neuron_unit_computation_steps.png
# ---
# height: 480px
# name: computational_graph1a
# align: left
# ---
# Computation steps of a neuron.
# :::
# 
# {numref}`computational_graph1b` shows the graphical representation of a single neuron unit. Note that the computational graph of a single neuron in {numref}`computational_graph1b` omits the summation in {numref}`computational_graph1a` required to compute the activation $a$. Thus, we assume that the inward weighted input signals in {numref}`computational_graph1b` are all added up before applying the non-linearity $ \phi(\cdot) $.
# 
# :::{figure} /images/neuralnets/neuron_unit_computational_graph.png
# ---
# height: 480px
# name: computational_graph1b
# align: left
# ---
# Computational graph of a neuron.
# :::
# 
# An ANN is simply a computational graph comprising multiple artificial neurons (see {numref}`computational_graph2`). Furthermore, the network topology is designed -- one can play for example with the number hidden layers, number of units per layer and number of connections per unit -- according to the specific task the network must perform e.g. indicate the class of objects on images. The task in turn is mapped into some arbitrarily complex function of the network inputs. Learning in this case consists in adjusting the network parameters (weights) such that the entire network approximates as good as possible the task function.
# 
# :::{figure} /images/neuralnets/example_toy_ANN.png
# ---
# height: 320px
# name: computational_graph2
# align: left
# ---
# Computational graph representing a toy neuron network comprising four neuron units. The illustrated network computes a function $ f: \mathbb{R}^{D} \rightarrow \mathbb{R} $ of its inputs (features) collected by the vector $ {\bf x} $. Note that $ f(\cdot) $ is an approximation to some task-related function that must be learned from data.
# :::
# 
# {numref}`computational_graph3` shows the computational graph of a single hidden layer neural network with an _input_ layer with $D$ input units storing observed features $ \lbrace x_{i} \rbrace $, a _hidden_ layer with $ H $ non-linear units (symbol $ \phi $) computing _unobserved_ (or embedded) features $ \lbrace h_{j} \rbrace $ and an _output_ layer with a single linear unit (symbol $ \sum $) yielding the network output $ \hat{y} = f({\bf x}) $.
# 
# :::{figure} /images/neuralnets/single_layer_MLP.png
# ---
# height: 320px
# name: computational_graph3
# align: left
# ---
# Computational graph representing a single hidden layer neural network. Weights were omitted in the graph for the sake of clarity.
# :::
# 
# The $j$-th hidden unit computes the _unobserved_ feature as
# \begin{equation}
# h_{j} = \phi \left( \sum_{i} w^{1}_{i,j} x_{i} \right). \nonumber
# \end{equation}
# Note that the superscript $ 1 $ indicates that the weight $ w^{1}_{i,j} $ belongs to an unit at the _first_ (hidden) layer. On the other hand, the subscripts $ i,j $ indicates that the weight $ w^{1}_{i,j} $ corresponds to a synaptic connection between the $i$-th input unit and the $j$-th hidden unit. Lastly, the single output unit computes the network output as $$ \hat{y} = \sum_{j} w^{2}_{j,1} h_j, $$ where the superscript $ 2 $ indicates that the weight $ w^{2}_{j,1} $ belongs to the single unit $ 1 $ at the _second_ (output) layer.
# 
# Alternatively, let the vectors $ {\bf x} = \begin{bmatrix} x_{0} = 1 & x_{1} & \ldots & x_{D} \end{bmatrix}^{T} $ and $ {\bf w}_{j}^{1} = \begin{bmatrix} w^{1}_{0,j} & w^{1}_{1,j} & \ldots & w^{1}_{D,j} \end{bmatrix}^{T} $ collect respectively the observed features and the corresponding weights at the $j$-th unit of the single hidden layer $ 1 $. The $j$-th _unobserved_ (or embedded) feature can be rewritten then as 
# \begin{equation}
# h_{j} = \phi \left( ({\bf w}_{j}^{1})^{T} {\bf x} \right). \nonumber
# \end{equation}
# 
# Analogously, let $ {\bf h} = \begin{bmatrix} h_{1} & h_{2} & \ldots & h_{H} \end{bmatrix}^{T} $ collect the embedded features and let $ {\bf w}_{1}^{2} = \begin{bmatrix} w^{2}_{1,1} & w^{2}_{2,1} & \ldots & w^{2}_{H,1} \end{bmatrix}^{T} $ collect the corresponding weights at the single unit of the output layer $ 2 $, we can rewrite the network output as
# \begin{equation}
# \hat{y} = ({\bf w}_{1}^{2})^{T} {\bf h}. \nonumber
# \end{equation}
# 
# Now, let the matrices
# \begin{eqnarray}
# {\bf W}^{1} &=& \begin{bmatrix} {\bf w}_{1}^{1} & \ldots & {\bf w}_{H}^{1} \end{bmatrix} \nonumber \\
# &=& \begin{bmatrix} 
# w^{1}_{0,1} & \ldots & w^{1}_{0,H} \\
# w^{1}_{1,1} & \ldots & w^{1}_{1,H} \\
# \vdots &  & \vdots \\
# w^{1}_{D,1} & \ldots & w^{1}_{D,H} \nonumber
# \end{bmatrix}
# \end{eqnarray}
# and
# \begin{eqnarray}
# {\bf W}^{2} &=& \begin{bmatrix} {\bf w}_{1}^{2} \end{bmatrix} \nonumber \\
# &=& \begin{bmatrix} 
# w^{2}_{1,1} \\
# \vdots \\
# w^{2}_{H,1} \nonumber
# \end{bmatrix}
# \end{eqnarray}
# collect all weights at layers $ 1 $ and $ 2 $, respectively, such that the $j$-th column stores the weights $ \lbrace w^{\ell}_{\bullet,j} \rbrace $ of the $j$-th unit at the corresponding layer $ \ell $. We can further rewrite the network output in compact vector-matrix notation as
# \begin{eqnarray}
# \hat{y} &=& ({\bf W}^{2})^{T} {\bf h} \nonumber \\
# &=& ({\bf W}^{2})^{T} \phi \left( {\bf a} \right) \nonumber \\
# &=& ({\bf W}^{2})^{T} \phi \left( ({\bf W}^{1})^{T} {\bf x} \right), \nonumber
# \end{eqnarray}
# in which the non-linearity is applied element-wise along the activation vector $ {\bf a} = \begin{bmatrix} a_{1} & a_{2} & \ldots & a_{H} \end{bmatrix}^{T} $ such that $ h_{j} = \phi(a_{j}) $, $ \forall j \in \lbrace 1, \ldots, H \rbrace $. Therefore, the output of the single hidden layer network in {numref}`computational_graph3` is equivalent to applying an affine transformation $ ({\bf W}^{1})^{T} {\bf x} $ to the network inputs $ {\bf x} $ followed by a non-linearity $ \phi $ to obtain the embedded features $ {\bf h} $ followed by a new affine transformation $ ({\bf W}^{2})^{T} {\bf h} $ to obtain the final output $ \hat{y} $.
# 
# :::{attention}
# The network capacity is determined by the number of network parameters (weights). As we increase the network capacity, we increase the network flexibility to fit the training examples. Going too far increasing the capacity and the network will overfit the training examples. On the other hand, the network will underfit the training examples with a too small capacity. In both cases, the network loses its ability to properly generalize, i.e. to provide a good approximation of the task-related function for unseen input patterns $ {\bf x} $ belonging, for instance, to a testing dataset.
# :::
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
