#!/usr/bin/env python
# coding: utf-8

# ## Function approximator
# 
# What do we mean exactly by a _complex task_? ANNs build high-dimensional complex functions using simple modules -- artificial neurons. However, like the brain, ANNs are black boxes and are hard to interpret, i.e. to clearly understand how they get their final output. Nevertheless, ANNs are universal approximators, they can approximate any function as indicated in the following theorem shown by Pinkus {cite}`pinkus1999approximation`.
# 
# :::{prf:theorem} Pinkus Theorem
# :label: approximator
# Any continuous function $f^{\ast} \colon \mathbb{R}^{D} \rightarrow \mathbb{R}^{O}$ can be arbitrarily well approximated (on any compact set) by a single layer neural network, for a wide range of non-linearities $\phi$ (non-polynomials). 
# :::
# 
# :::{figure} /images/neuralnets/network_approximation.svg
# ---
# height: 320px
# name: network_approximation
# align: left
# ---
# Network approximation $ f \colon \mathbb{R}^{D} \rightarrow \mathbb{R}^{O} $ to a continuous function $ f^{\ast} \colon \mathbb{R}^{D} \rightarrow \mathbb{R}^{O} $. For a given observed feature vector $ \check{\bf x} \in \mathbb{R}^{D} $, the network estimates the output as $ \hat{\bf y} = f(\check{\bf x}) \in \mathbb{R}^{O} $ as opposed to the true output $ {\bf y} = f^{\ast}(\check{\bf x}) \in \mathbb{R}^{O} $ that would be obtained by the approximated function $ f^{\ast} $. Lastly, note that the network task represented by the function $f^{\ast}$ can be arbitrarily complex e.g. convert speech-to-text.
# :::
# 
# Note that we need to employ a non-linear activation function $\phi \colon \mathbb{R} \rightarrow \mathbb{R} $. {numref}`activation_functions` illustrates some widespread activation functions in machine learning literature. Common activation functions in traditional machine learning are the sigmoid
# \begin{equation}
# \phi(x) = \frac{1}{1+e^{-x}}
# \end{equation}
# and hyperbolic tangent
# \begin{equation}
# \phi(x) = \frac{e^{x}-e^{-x}}{e^{x}+e^{-x}}.
# \end{equation}
# However, modern deep learning methods typically use unbounded, non-linear activation functions such as the rectified linear unit (ReLU)
# \begin{equation}
# \phi(x) = \max(0, x),
# \end{equation}
# the leaky ReLU
# \begin{equation}
# \phi(x) = \max(\alpha x, x)
# \end{equation}
# or the exponential linear unit (LU)
# \begin{equation}
# \phi(x) = \left\lbrace 
# \begin{matrix}
# x, & x \geq 0 \\
# \alpha \left( e^{x}-1 \right), & \mbox{otherwise}
# \end{matrix}
# \right.
# \end{equation}
# in which $ \alpha \geq 0 $ is a small constant e.g. $ \alpha = 0.1 $.
# 
# Activation functions are carefully designed such that their derivatives are available in closed form and can be computed very efficiently. For example, for the sigmoid and ReLU activation functions, the derivatives are straightforwardly computed respectively as
# \begin{equation}
# \phi'(x) =  \frac{\partial \phi(x)}{\partial x} =  \phi(x) \, \phi(-x)
# \end{equation} 
# and
# \begin{equation}
# \phi'(x) =  \frac{\partial \phi(x)}{\partial x} =  \left[ x \geq 0 \right],
# \end{equation}
# where the Iverson bracket $ \left[ \Psi \right] $ yields $ 1 $ if the logical proposition $ \Psi $ is true or $ 0 $, otherwise.
# 
# :::{figure} /images/neuralnets/activation_functions.png
# ---
# height: 480px
# name: activation_functions
# align: left
# ---
# Common non-linear activation functions (borrowed from {cite}`sze2017efficient`). In contrast to traditional ones, modern activation functions are typically unbounded and can be computed very efficiently.
# :::
# 
# ::::{important}
# Modern activation functions are unbounded such that their output values do not saturate and their derivatives do not collapse to zero for larger activation values (see {numref}`sigmoid_function` and {numref}`relu_function`). This is paramount in deep neural networks, since the activation values might become bigger and bigger as the feed input features $ \lbrace x_{i} \rbrace $ are forwarded over the network. Moreover, unbounded activation functions -- with non-zero derivatives -- are crucial for the deep learning itself. As we will see shortly (in the backpropagation procedure), the network weights are updated in order to minimize some loss function $ L(y,\hat{y}) $. Specifically, each weight is updated using the back propagated derivatives of the loss function in respect to the weight itself. To accomplish this, the chain rule of calculus is applied several times over potentially long branches of the associated computational graph linking the parameter to the network outputs. Additionally, the product of chained derivatives might become smaller and smaller as derivatives are back propagated through the network. Note though that, when the activation function saturates, its derivative becomes close to zero. Thus, if that happens in all paths between a network parameter and the network outputs, this parameter will become insensitive to the improvements in the loss function and will not be updated anymore, i.e. the network will not learn to perform its task based on this parameter.
# 
# {numref}`sigmoid_function` and {numref}`relu_function` show respectively the sigmoid and ReLU activation functions and their derivatives.
# 
# :::{figure} /images/neuralnets/sigmoid_d.png
# ---
# height: 320px
# name: sigmoid_function
# align: left
# ---
# Sigmoid activation function. The sigmoid derivative collapses to zero for large absolute values of the activation.
# :::
# 
# :::{figure} /images/neuralnets/relu_d.png
# ---
# height: 320px
# name: relu_function
# align: left
# ---
# ReLU activation function. The ReLU function derivative remains constant for large positive activation values.
# :::
# ::::
# 
# ::::{prf:example}
# Suppose a linear activation function of the type $ \phi(a) = \kappa a $. It can be shown that the resulting network can be actually reduced to a single linear neuron unit with properly adjusted weights $ \tilde{\bf w} = \begin{bmatrix} \tilde{w}_{0} \equiv \tilde{b} & \tilde{w}_{1} & \ldots & \tilde{w}_{D} \end{bmatrix} $, i.e. the network will not be able to approximate any function more complex than an affine transform $ \hat{y} = f({\bf x}) = \tilde{\bf w}^{T} {\bf x} $ of its inputs $ {\bf x} = \begin{bmatrix} {x}_{0} \equiv 1 & {x}_{1} & \ldots & {x}_{D} \end{bmatrix} $. Thus, a non-linear activation function is required for the network to approximate more complex functions.
# 
# {numref}`computational_graph4` shows an example of a linear network with activation function $ \phi(a) = a $.
# :::{figure} /images/neuralnets/example_ANN_linear_02.png
# ---
# height: 320px
# name: computational_graph4
# align: left
# ---
# The need for non-linearity. Computational graph comprising three linear neuron units on the left is equivalently to a single linear unit on the right.
# :::
# 
# Note that in this case we have
# \begin{eqnarray}
# \hat{y} &=& w_{0,1}^{2} + \sum_{j=1}^2 w^{2}_{j,1} \left( \sum_{i=0}^D w^{1}_{i,j} x_{i} \right) \nonumber \\
# &=& \sum_{i=0}^D \tilde{w}_{i} x_{i}. \nonumber
# \end{eqnarray}
# Thus, the network is equivalent to a single unit with weights $$ \tilde{w}_{0} = {w^{2}_{0,1}} + w^{1}_{0,1} \, w^{2}_{1,1} +  w^{1}_{0,2} \, w^{2}_{2,1} $$ and $$ \tilde{w}_{i} = w^{1}_{i,1} \, w^{2}_{1,1} +  w^{1}_{i,2} \, w^{2}_{2,1}, \,\,\, \forall i \in \lbrace 1, \ldots, D \rbrace. $$
# ::::
# 
