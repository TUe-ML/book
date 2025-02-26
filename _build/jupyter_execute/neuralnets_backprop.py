#!/usr/bin/env python
# coding: utf-8

# ## Backpropagation
# 
# Neural networks are powerful function approximators, with thousands, millions or even billions of parameters that must be tuned (learned). But how do they learn these parameters from data?
# 
# In short, we need first to define a loss function indicating what the network should learn. More precisely, the loss function must express how good (or bad) the network is performing the task it is supposed to do during training. Thus, the selection of the loss function often depends on the kind of task the network must perform. For example, we typically use:
# * the mean squared error (MSE) for regression tasks; and
# * the cross entropy for classification tasks.
# 
# In the sequel, we must adjust the network parameters to reduce the training loss. This machine learning problem is commonly stated as an _optimization problem_ and it is frequently solved by means of _gradient descent_ methods. The backpropagation algorithm, in turn, provides a systematic procedure to compute gradients over the computational graph representing computations across an artificial neural network. Specifically, it employs the chain rule of calculus with a lot of _bookkeeping_ to allow one to compute the gradient of the training loss w.r.t. each network parameter. Thus, backpropagation plays a major role to allow one to use gradient descent methods for learning the network parameters. 
# 
# {prf:ref}`learning_alg` summarizes this high-level learning procedure which employs the back propagation of the _prediction quality_ -- encoded in the loss function -- to adjust the weights of the artificial synapses. 
# 
# :::{prf:algorithm} Learning procedure
# :label: learning_alg
# 
# **Inputs**
# * Network structure and hyper-parameters.
# 
# **Output**
# * Trained / tunned network parameters.
# 
# **Function** Learn network parameters
# 1.  Initialize network parameters
# 2. **repeat**
# 	1. Compute the training loss, i.e. assess the network prediction quality.
#     2. Back propagate training loss gradients w.r.t. network parameters.
#     3. Change parameters a bit to reduce the loss. *(Gradient descent step)*
# 3. **until** Reach some stop criteria
# 4. **return** Trained network parameters
# :::
# 
# ### Loss functions
# 
# There are several loss functions we can use. For convenience, let $ {\bf w} $ be a long vector collecting all neural network parameters, i.e. the vector $ {\bf w} $ collect the parameters $ \mathbf{W}^{1}, \mathbf{W}^{2}, \ldots, \mathbf{W}^{L} $ from all network layers. Now, let $ \hat{y} = f({\bf x}; {\bf w}) $ be the network's output for input vector $ {\bf x} $ and network parameters $ {\bf w} $. The loss function $L(y, \hat{y})$ -- a.k.a. risk or cost function -- measures how well the prediction $\hat{y}$ approximates the target $y$. Moreover, the training loss, i.e. the empirical risk, is computed using the training samples $$ {\cal D}_{train} = \bigcup_{i=1}^{N} \lbrace \left( {\bf x}_{i}, y_{i} \right) \rbrace $$ as
# :::{math}
# :label: empirical_risk
# \begin{eqnarray}
# L_{train} &=& \frac{1}{N} \sum_{i=1}^{N} L \left( y_{i}, \hat{y}_{i} \right) \\
# &=& \frac{1}{N} \sum_{i=1}^{N} L \left( y_{i}, f({\bf x}_{i}; {\bf w}) \right).
# \end{eqnarray}
# :::
# 
# As mentioned above, the most common loss for regression is the Mean Square Error (MSE). The squared error is defined as
# :::{math}
# :label: squared_error
# \begin{equation}
# L(y, \hat{y}) = (\hat{y} - y)^2
# \end{equation}
# :::
# and its partial derivative w.r.t. estimate $ \hat{y} $ is given simple by
# \begin{equation}
# {\partial L(y, \hat{y}) \over \partial \hat{y}} = 2 (\hat{y} - y).
# \end{equation}
# The Mean Squared Error (MSE) in turn is obtained by plugging {eq}`squared_error` into the empirical risk {eq}`empirical_risk`
# \begin{eqnarray}
# L_{train} &=& MSE(y, \hat{y}; {\bf w}) \\
# &=& \frac{1}{N} \sum_{i=1}^{N} (\hat{y}_{i} - y_{i})^2  \\
#  &=& \frac{1}{N} \sum_{i=1}^{N} (f({\bf x}_{i}; {\bf w}) - y_{i})^2.
# \end{eqnarray}
# 
# On the other hand, a Cross Entropy (CE) loss function is frequently employed for classification tasks. But how do we setup a neural network as a classifier? Let $C$ denote the number of classes among the network must decide. First, we employ $C$ linear units at the last hidden layer of the network, i.e. the vector $ {\bf h}^{L} $ collecting the outputs of the last hidden layer has $ C $ elements. Then, we use a _softmax_ activation to convert this vector into a probability distribution.
# 
# In the example of {numref}`classification_MLP_fig`, the network must decide among $ C=3 $ classes. Thus, $\mathbf{W}^{4}$ must be a $H_L \times C$ matrix ($C=3$) such that $$ {\bf h}^{4} = {\mathbf{W}^{4}}^{T} \phi_3 \Big( {\mathbf{W}^{3}}^{T} \phi_2 \Big( {\mathbf{W}^{2}}^{T} \phi_1 \Big( {\mathbf{W}^{1}}^{T} {\bf h}^{0} \Big) \Big) \Big) $$ is a $ 3 $-element vector with $ {\bf h}^{0} = \begin{bmatrix} 1 & x_1 & x_2 & \ldots & x_D \end{bmatrix}^{T} $.
# 
# :::{figure} /images/neuralnets/classification_MLP.svg
# ---
# height: 320px
# name: classification_MLP_fig
# align: left
# ---
# A neural network with $ C = 3 $ linear units at its output layer. The $ \sum $ symbol indicates that the corresponding units is linear.
# :::
# 
# The softmax activation converts an arbitrary input vector into a valid probability distribution. In precise mathematical terms, let ${\bf h}=\begin{bmatrix} h_1 & h_2 & \ldots & h_C \end{bmatrix}^{T}$ be a $C$-dimensional vector. Then, the softmax of ${\bf h}$ is defined as
# :::{math}
# :label: softmax
# \begin{equation}
# \softmax({\bf h}) = \begin{bmatrix} \hat{y}_1 & \hat{y}_2 & \ldots & \hat{y}_C \end{bmatrix}^{T},
# \end{equation}
# :::
# in which
# \begin{equation}
# \hat{y}_{i} = {e^{h_{i}} \over \sum_{c=1}^{C} e^{h_{c}}}. 
# \end{equation}
# Note that the softmax {eq}`softmax` is defined such that it returns a valid categorical distribution, i.e. $$ \hat{y}_{i} \geq 0, \,\, \forall i \in \lbrace 1, 2, \ldots, C \rbrace, $$ and $$ \sum_{i=1}^{C} \hat{y}_{i} = 1. $$ Lastly, it is worth noting that the vector ${\bf h}$ is often called the _logits_ of the softmax.
# 
# :::{figure} /images/neuralnets/softmax.png
# ---
# height: 320px
# name: softmax_fig
# align: left
# ---
# Softmax example using _NumPy_ for array data handling and _Matplotlib_ for visualization.
# :::
# 
# Let the vector $ \hat{\bf y} = \begin{bmatrix} \hat{y}_1 & \hat{y}_2 & \ldots & \hat{y}_C \end{bmatrix}^{T} $ store the predicted categorical distribution $ q(c) $ over the $ C $ classes such that $ q(c) = \hat{y}_c $, $ \forall c \in \lbrace 1, \ldots, C \rbrace $. Note that $ \hat{\bf y} $ is the softmax of the last layer, i.e. $ \hat{\bf y} = \softmax({\bf h}^{L}) $, where ${\bf h}^{L}=\begin{bmatrix} h_1^{L} & h_2^{L} & \ldots & h_C^{L} \end{bmatrix}^{T}$ collect the values at the last hidden layer $ L $. The Cross Entropy (CE) of the predicted distribution $ q(c) $ relative to the true distribution $ p(c) = \left[ y = c \right] $ (a distribution with no uncertainty at all) is defined as
# :::{math}
# :label: ce
# \begin{eqnarray}
# L(y, \hat{\bf y}) &=& CE(y, \hat{\bf y}; {\bf w}) \\
# %&=& CE(y, f({\bf x}; {\bf w}))  \\
# &\triangleq& - \sum_{c=1}^{C} p(c) \log q(c)  \\
# &=& - \sum_{c=1}^{C} \left[ y=c \right] \log \left( \hat{y}_{c} \right)  \\
# &=& - \left\lbrace \underbrace{\left[ y=1 \right]}_{0} \log \left( \hat{y}_{1} \right) + \ldots + \underbrace{\left[ y=y \right]}_{1} \log \left( \hat{y}_{y} \right) + \ldots + \underbrace{\left[ y=C \right]}_{0} \log \left( \hat{y}_{C} \right) \right\rbrace  \\
# &=& - \log \left( \hat{y}_{y} \right),
# \end{eqnarray}
# :::
# in which $ y $ corresponds to the index of the true class. In this case, the CE {eq}`ce` is the negative log-probability of the true class. For a perfect prediction ($ \hat{y}_{y} = 1 $ and $ \hat{y}_{c} = 0 $ for $ c \neq y $), the CE loss is zero (minimum). On the other hand, for an uncertain prediction ($ 0 < \hat{y}_{y} < 1 $), CE will quickly approach $ \infty $ as $ \hat{y}_{y} \rightarrow 0^{+} $, i.e. the CE loss strongly penalizes the predicted distribution if the predicted probability of the true-class $ \hat{y}_{y} $ is close to zero.
# 
# Therefore, the mean cross entropy, i.e the training loss, is obtained as 
# \begin{equation}
# L_{train} =  {1 \over N} \sum_{i=1}^{N} CE(y_{i}, \hat{\bf y}_{i}; {\bf w}),
# \end{equation}
# where $ y_{i} $ and $ \hat{\bf y}_{i} = f({\bf x}_{i}; {\bf w})$ are respectively the true class and the network output (probability) associated with the $i$-th training example $ \left({\bf x}_{i}, y_{i} \right) \in {\cal D}_{train} $.
# 
# :::{figure} /images/neuralnets/MLP_CE_04.png
# ---
# height: 320px
# name: ce_fig
# align: left
# ---
# Computational graph representing a MLP with $4$ hidden layers designed for classification ($C=3$). The last hidden layer comprises $3$ linear units whose outputs are collected by the vector $ {\bf h}^{4} $. The $\softmax$ is applied to $ {\bf h}^{4} $ to build a valid categorical distribution stored in $ \hat{\bf y} = \begin{bmatrix} \hat{y}_{1} & \hat{y}_{2} & \hat{y}_{3} \end{bmatrix}^{T} $ which can be evaluated in turn by computing the cross entropy $ CE(y, \hat{\bf y}; {\bf w}) $ using the true class $ y $ as in {eq}`ce`.
# :::
# 
# Let $ {\bf h} = \begin{bmatrix} h_1 & h_2 & \ldots & h_C \end{bmatrix}^{T} $ collect the values at the last hidden layer, i.e. the logits of the softmax. To be able to learn, we need to compute the derivative of the cross-entropy. In the following, we use $ CE = CE(y, \hat{\bf y}; {\bf w}) $ for the sake of clarity. The derivative of the cross-entropy w.r.t. the value $ h_j $ of the $j$-th unit is given by
# \begin{align*}
# {\partial CE \over \partial h_j} =
# \begin{cases}
#  \hat{y}_j - 1   & \text{if } j=y \\
#  \hat{y}_j  & \text{if } j\not=y.
# \end{cases}
# \end{align*}
# 
# ### Gradient descent
# {prf:ref}`gradient_descent_alg` summarizes the gradiend descent procedure which optimizes the training loss w.r.t. the network parameters collected by the vector ${\bf w}$.
# 
# :::{prf:algorithm} Training Loss Optimization procedure
# :label: gradient_descent_alg
# 
# **Inputs**
# * Neural network structure with parameters ${\bf w}$.
# * Training data $ {\cal D}_{train} = \bigcup_{i=1}^{N} \lbrace \left( {\bf x}_i, y_i \right) \rbrace $.
# * Maximum number of training epochs $max\_epochs$.
# * Step-sizes $\lbrace \eta_k \rbrace $.
# 
# **Output**
# * Trained neural network parameters ${\bf w}$.
# 
# **Function** GradientDescent (${\bf w}$, ${\cal D}_{train}$, $max\_epochs$, $\eta_k$})
# 1. Randomly initialize the weigths ${\bf w}$.
# 2. **forall** $k \in \lbrace 1, \ldots, max\_epochs \rbrace $
#     1. Compute network predictions for all $i \in \lbrace 1, \ldots, N \rbrace $:
#     $$ \triangleright \,\,\,\,\,\, \hat{\bf y}_{i} \gets f({\bf x}_{i}; {\bf w})$$ 
#     2. Compute the training loss:
#     $$ \triangleright \,\,\,\,\,\, L_{train} = {1 \over N} \sum_{i=1}^N L(y_i, \hat{\bf y}_{i})$$
#     3. Compute the gradient of the loss w.r.t. ${\bf w}$:
#     $$ \triangleright \,\,\,\,\,\, \nabla_{\bf w} L_{train}$$ *(Requires backpropagation)*
#     4. Update the weights:
#     $$ \triangleright \,\,\,\,\,\, {\bf w} \gets {\bf w} - \eta_{k} \, \nabla_{\bf w} L_{train}$$
# 3. **return** Trained neural network parameters ${\bf w}$.
# :::
# 
# {numref}`loss_ladscape_fig` illustrates in turn how challenging could be optimizing the loss function for two parameters.
# 
# :::{figure} /images/neuralnets/loss_landscape.png
# ---
# height: 320px
# name: loss_ladscape_fig
# align: left
# ---
# Loss landscape of neural networks for a bi-dimensional parameter space (borrowed from {cite}`li2017visualizing`). There are several local minima and one global minima. Ordinary gradient descent implementations could be easily captured by a local minima.
# :::
# 
# Note that computing the gradient $ \nabla_{\bf w} L_{train} $ is key for {prf:ref}`gradient_descent_alg`. But how do we compute this gradient? First, let us defined it precisely. The gradient of the training loss $L_{train}$ w.r.t. to the parameter vector $ {\bf w} $ is defined as
# \begin{equation}
#  \nabla_{\bf w} L_{train} = \begin{bmatrix} \frac{\partial L_{train}}{\partial w_1} & \frac{\partial L_{train}}{\partial w_2} & \cdots \end{bmatrix}^{T}. 
# \end{equation}
# Note though that we can rewrite the overall gradient as a sum of sample-wise gradients
# \begin{eqnarray}
# \nabla_{\bf w} L_{train} &=& \nabla_{\bf w}  \left( {1 \over N}  \sum_{i=1}^N L(y_{i}, \hat{\bf y}_{i}) \right)  \\
# &=& {1 \over N}  \sum_{i=1}^N \nabla_{\bf w} L(y_{i}, \hat{\bf y}_{i}),
# \end{eqnarray}
# in which the gradient $ \nabla_{\bf w} L(y_{i}, \hat{\bf y}_{i}) $ of the $i$-th sample in $ {\cal D}_{train} $ is computed independently.
# 
# The gradient of a single sample $$ \nabla_{\bf w} L(y_{i}, \hat{\bf y}_{i}) $$ in turn can be computed using the backpropagation of error (backprop). {numref}`backprop_fig` illustrates the backpropagation procedure. First the input features in $ {\bf x} $ are feedforwarded to compute the network prediction $\hat{\bf y} $ using the current network parameters stored in ${\bf w}$. Then, the loss function $ L(y, \hat{\bf y}) $ is computed to evaluate the prediction quality against the true value $ y $. In this sense, the loss function encodes the prediction error. Lastly, the prediction error is backpropagated. Specifically, the derivatives of the loss function are backpropagated to allow one to compute the loss function derivative $\frac{\partial L_{train}}{\partial w_j}$ w.r.t any network parameter $ w_j $ in $ {\bf w} $.
# 
# :::{figure} /images/neuralnets/backprop_scheme_04.png
# ---
# height: 480px
# name: backprop_fig
# align: left
# ---
# Backpropagation procedure overview.
# :::
# 
# But how do we compute the derivatives $ \frac{\partial L_{train}}{\partial w_j} $ of the loss function w.r.t. to each network parameters $ w_j $? We employ the chain rule of calculus plus some _bookkeeping_ to store partial derivatives along the paths linking this parameter to the network outputs.
# 
# ::::{prf:definition} Chain rule of calculus
# Let the function $ f(x) = f(g(x), h(x)) $ be the composition of two functions $ g(x) $ and $ h(x) $. Thus, the derivative of $ f(\cdot) $ w.r.t. the argument $ x $ can be computed as $$ {\partial f \over \partial x } = {\partial f \over \partial g } {\partial g \over \partial x } + {\partial f \over \partial h } {\partial h \over \partial x }. $$
# 
# {numref}`backprop_detail_fig` shows how the derivatives are combined over the computational graph corresponding to function $ f(x) = f(g(x), h(x)) $. Partial derivatives along the same backpropagation path between two node are multiplied to obtain $ {\partial f \over \partial g } {\partial g \over \partial x } $ and $ {\partial f \over \partial h } {\partial h \over \partial x } $. Backpropagated derivatives arriving at the node $ {\bf x} $ are added up to obtain the final derivative $ {\partial f \over \partial x } $.
# 
# :::{figure} /images/neuralnets/backprop_detail_simple.svg
# ---
# height: 320px
# name: backprop_detail_fig
# align: left
# ---
# Backpropagation of partial derivatives.
# :::
# ::::
# 
# {numref}`backprop_detail_small_fig1` through {numref}`backprop_detail_small_fig4` detail the backpropagation procedure for a small network. It shows the backpropagated messages over the network. Note that the computational graph was expanded in the figures to turn the computation of each unit activation explicit.
# 
# :::{figure} /images/neuralnets/backprop_detail_small_01.png
# ---
# height: 320px
# name: backprop_detail_small_fig1
# align: left
# ---
# A single hidden layer network.
# :::
# 
# :::{figure} /images/neuralnets/backprop_detail_small_02.png
# ---
# height: 320px
# name: backprop_detail_small_fig2
# align: left
# ---
# Backprop message $ {\color{blue} B_{1} } $.
# :::
# 
# :::{figure} /images/neuralnets/backprop_detail_small_03.png
# ---
# height: 320px
# name: backprop_detail_small_fig3
# align: left
# ---
# Backprop messages $ {\color{green} B_{2,1} } $ and ${\color{green} B_{2,2} }$.
# :::
# 
# :::{figure} /images/neuralnets/backprop_detail_small_04.png
# ---
# height: 320px
# name: backprop_detail_small_fig4
# align: left
# ---
# Backprop messagess ${\color{darkred} B_{3,1} } $ and ${\color{darkred} B_{3,2} }$.
# :::
# 
# We detail the bookkeeping of backpropagated messages below.
# 
# I. Compute the message $ {\color{blue} {B_1} } $ as the derivative of the loss function (MSE or CE):
# \begin{equation}
#  {\color{blue} \partial L \over \partial \hat{y}} \triangleq {\color{blue} {B_1} }.
# \end{equation}
# II. Compute the messages $ {\color{green} B_{2,1} } $ and $ {\color{green} B_{2,2} } $:
# \begin{eqnarray}
#  {\color{green} \partial L \over \partial h_i} &=& {\color{blue} \partial L \over \partial \hat{y}}{\partial \hat{y} \over \partial h_i}  \\
#  &=& {\color{blue} B_1 } w^{2}_{i,1}  \\
#  &\triangleq& {\color{green} B_{2,i}}.
# \end{eqnarray}
# III. Compute the messages ${\color{darkred} B_{3,1} } $ and ${\color{darkred} B_{3,2} }$ using the activation function derivative:
# \begin{eqnarray}
#  {\color{darkred} \partial L \over \partial a_i} &=& {\color{green} \partial L \over \partial h_i}{\partial h_i \over \partial a_i}  \\
#  &=& {\color{green} B_{2,i} } \, \phi_1'(a_i)  \\
#  &\triangleq& {\color{darkred} B_{3,i} }.
# \end{eqnarray}
# IV. Then, compute the partial derivatives of the loss w.r.t. the network parameters at the output layer:
# \begin{eqnarray}
#  {\partial L \over \partial w^2_{i,1}} &=& {\color{blue} \partial L \over \partial \hat{y}}{\partial \hat{y} \over \partial w^2_{i,1}}   \\
# &\equiv& {\color{blue} {B_1} } {h_i}.
# \end{eqnarray}
# VI. Finally, compute the partial derivatives of the loss w.r.t. the network parameters at the hidden layer:
# \begin{eqnarray}
# {\partial L \over \partial w^1_{i,j}} &=& {\color{darkred} \partial L \over \partial a_j}{\partial a_j \over \partial w^1_{i,j}}  \\
# &\equiv& {\color{darkred} {B_{3,j}} } {x_i}.
# \end{eqnarray}
# 
# :::{prf:definition} Jacobian matrix
# Let $y = f(x)$ be a single input, single output (SISO) function $f \colon \mathbb{R} \rightarrow \mathbb{R}$. The derivative of $f$ is written as $$ f' \triangleq {\partial f \over \partial x}. $$ Now, let $f \colon \mathbb{R}^n \rightarrow \mathbb{R}^m$ denote a multiple input, multiple output (MIMO) function, i.e.
# \begin{eqnarray}
# {\bf y} &=& f({\bf x})  \\
# &=& \begin{bmatrix} f_1({\bf x}) & \ldots & f_m({\bf x})\end{bmatrix}^{T}.  
# \end{eqnarray}
# The _derivative_ of $f$ w.r.t. to its argument $ {\bf x} = \begin{bmatrix} x_{1} & \ldots & x_{n} \end{bmatrix}^{T} $ is defined by the **Jacobian matrix**. More precisely, the Jacobian matrix ${\bf J}$ is a $m \times n$ matrix containing all _partial_ derivatives
# \begin{equation}
# {\bf J} = 
# \begin{bmatrix}
# {\partial f_1 \over \partial x_1} & \ldots & {\partial f_1 \over \partial x_n} \\
# \vdots &  \ddots & \vdots \\
# {\partial f_m \over \partial x_1} & \ldots & {\partial f_m \over \partial x_n}
# \end{bmatrix}.
# \end{equation}
# :::
# 
# ### Vectorized Backprop
# 
# Now, let ${\color{green}\mathbf{v}}$ and ${\color{darkred} \mathbf{v'}}$ be two sets containing respectively $n$ and $m$ units of a neural network (possibly from different layers). Moreover, let  $f({\color{green}\mathbf{v}}) = {\color{darkred} \mathbf{v'}}$ be a function $f \colon \mathbb{R}^n \mapsto \mathbb{R}^m$ mapping values from ${\color{green}\mathbf{v}}$ into values at  ${\color{darkred} \mathbf{v'}}$. If all computational paths from the units in ${\color{green}\mathbf{v}}$ to the loss function $L$ (at the output) go over the units in ${\color{darkred}\mathbf{v}'}$, then we can write
# :::{math}
# :label: jacob_magic
# \begin{equation}
# {\color{green} \nabla_{\mathbf{v}} L} =  {\color{blue} \mathbf{J}^{T}} {\color{darkred}\nabla_{\mathbf{v}'} L}.
# \end{equation}
# :::
# That is, the Gradient of the loss $L$ w.r.t. ${\color{green}\mathbf{v}}$ is the Gradient of the loss $L$ w.r.t. ${\color{darkred}\mathbf{v}'}$ pre-multiplied by the Jacobian of $f$ transposed. Thus, it suffices to know the Jacobian of $f$ to backpropagate the Gradients from units ahead (${\color{darkred} \mathbf{v}'}$) to units backwards (${\color{green}\mathbf{v}}$) in the network. Note that this is similar to the $1D$ case in which $L = L(f({\color{green}v})) = L({\color{darkred}v'})$ and
# \begin{eqnarray}
#  {\color{green} \partial L \over \partial v} &=& {\color{darkred} \partial L \over \partial v'}{\color{blue} \partial v' \over \partial v}  \\
#  &=& {\color{blue} f'} {\color{darkred} \partial L \over \partial v'}.
# \end{eqnarray}
# 
# {numref}`vectorized_backprop_fig` compares the $1D$ backpropagation procedure with the vectorized backpropagation over a simplified computational graph in which grouped units (circles) in ${\color{darkred} \mathbf{v}'}$ and ${\color{green}\mathbf{v}}$ are enclosed by boxes. In the former case, we compute the Gradient at a single unit $ {\color{green}v} $ using the gradient $ {\color{blue} f'} $ of the SISO function$ f: \mathbb{R} \rightarrow \mathbb{R} $ mapping the values at this unit to the values at the unit ahead ${\color{darkred}v'}$. In the later case, we compute the Gradient at the units ${\color{green}\mathbf{v}}$ using the Jacobian $ {\color{blue} \mathbf{J}^{T}} $ of the MIMO function $ f: \mathbb{R}^{n} \rightarrow \mathbb{R}^{m} $ summarizing all computational paths of these units to the units ahead ${\color{darkred} \mathbf{v}'}$.
# 
# :::{figure} /images/neuralnets/jacobian.svg
# ---
# height: 640px
# name: vectorized_backprop_fig
# align: left
# ---
# Vectorized backpropagation of the Gradients.
# :::
# 
# ### Vectorized Backprop in MLPs
# 
# Note that MLPs have **only two** vectorized operations. Specifically, at a given layer $\ell$, we have
# * A matrix multiplication ${\bf a}^{\ell} = {{\bf W}^{\ell}}^{T} {\bf h}^{\ell-1}$; and
# * An element-wise nonlinearity ${\bf h}^{\ell} = \phi_{\ell}({\bf a}^{\ell})$.
# 
# Note though that the Jacobian of a **matrix multiplication** of the type $ {\bf y} = f({\bf x}) = {\bf W} {\bf x}$ is just the matrix $ {\bf W} $ defining the affine transform, whereas the Jacobian of an **element-wise nonlinearity** of the type ${\bf y} = f({\bf x})$ is a diagonal matrix $ {\bf J} = \mathrm{diag}(f'({\bf x})) $.
# 
# More explicitly, let the matrix multiplication
# \begin{equation}
# \underbrace{\begin{bmatrix}
# y_1 \\
# y_2 \\
# \vdots \\
# y_n
# \end{bmatrix}}_{\bf y}
# =
# \underbrace{\begin{bmatrix}
# w_{11} & w_{12} & \dots  & w_{1m} \\
# w_{21} & w_{22} & \dots  & w_{2m} \\
# \vdots & \vdots & \ddots & \vdots \\
# w_{n1} & w_{n2} & \dots  & w_{nm}
# \end{bmatrix}}_{\bf W}
# \underbrace{\begin{bmatrix}
# x_1 \\
# x_2 \\
# \vdots \\
# x_m
# \end{bmatrix}}_{\bf x}
# \end{equation}
# define a linear function of the type ${\bf y} = f({\bf x}) = \mathbf{W} {\bf x}$. The Jacobian of this matrix multiplication operation is given by
# \begin{eqnarray}
# {\bf J}
# &=&
# \begin{bmatrix}
# {\partial y_1 \over \partial x_1} & {\partial y_1 \over \partial x_2} & \dots  & {\partial y_1 \over \partial x_m} \\
# {\partial y_2 \over \partial x_1} & {\partial y_2 \over \partial x_2} & \dots  & {\partial y_2 \over \partial x_m} \\
# \vdots & \vdots & \ddots & \vdots \\
# {\partial y_n \over \partial x_1} & {\partial y_n \over \partial x_2} & \dots  & {\partial y_n \over \partial x_m} \\
# \end{bmatrix}  \\
# &=&
# \underbrace{\begin{bmatrix}
# w_{11} & w_{12} & \dots  & w_{1m} \\
# w_{21} & w_{22} & \dots  & w_{2m} \\
# \vdots & \vdots & \ddots & \vdots \\
# w_{n1} & w_{n2} & \dots  & w_{nm}
# \end{bmatrix}}_{\bf W}. 
# \end{eqnarray}
# 
# On the other hand, let the function ${\bf y} = f({\bf x}) = \begin{bmatrix} f(x_1) & f(x_2) & \ldots & f(x_n) \end{bmatrix}^{T} $ represent an element-wise nonlinearity applied to the input vector $ {\bf x} = \begin{bmatrix} x_1 & x_2 & \ldots & x_n \end{bmatrix}^{T} $. The Jacobian of this element-wise nonlinearity is given
# :::{math}
# :label: jacobian_magic2
# \begin{eqnarray}
# {\bf J}
# &=&
# \begin{bmatrix}
# {\partial y_1 \over \partial x_1} & {\partial y_1 \over \partial x_2} & \dots  & {\partial y_1 \over \partial x_n} \\
# {\partial y_2 \over \partial x_1} & {\partial y_2 \over \partial x_2} & \dots  & {\partial y_2 \over \partial x_n} \\
# \vdots & \vdots & \ddots & \vdots \\
# {\partial y_n \over \partial x_1} & {\partial y_n \over \partial x_2} & \dots  & {\partial y_n \over \partial x_n} \\
# \end{bmatrix}  \\
# &=&
# \underbrace{\begin{bmatrix}
# f'(x_1) & 0       & \dots  & 0 \\
# 0       & f'(x_2) & \dots  & 0 \\
# \vdots  & \vdots  & \ddots & \vdots \\
# 0       & 0       & 0      & f'(x_n)
# \end{bmatrix}}_{\mathrm{diag}(f'({\bf x}))}. 
# \end{eqnarray}
# :::
# 
# ::::{prf:example} Vectorized backprop over a small MLP
# {numref}`vectorized_backprop2_fig` shows a MLP with $3$ non-linear hidden layers and one linear output layer.
# 
# :::{figure} /images/neuralnets/multi_layer_MLP_07.png
# ---
# height: 320px
# name: vectorized_backprop2_fig
# align: left
# ---
# Example of a small MLPs to illustrate the vectorized backpropagation procedure.
# :::
# 
# In the sequel, we detail the feedforward evaluation and the backpropagation of error procedures for the small MLP shown in {numref}`vectorized_backprop2_fig`. First, remember that the network output $ \hat{y} = f({\bf x}) $ for a given input vector ${\bf x} = \begin{bmatrix} x_1 & x_2 & \ldots & x_D \end{bmatrix}^{T} $ is recursively computed -- from the inputs to the output -- by applying a linear transformation $ \mathbf{a}^\ell = {\mathbf{W}^{\ell}}^{T} \mathbf{h}^{\ell-1} $ followed by an element-wise nonlinearity $ \mathbf{h}^\ell = \phi_\ell(\mathbf{a}^\ell) $ at each layer $ \ell $. For the MLP shown in {numref}`vectorized_backprop2_fig`, we have the following feedforward computation steps
# 1. $ \mathbf{a}^1 = {\mathbf{W}^{1}}^{T} \mathbf{h}^0 $ with $ \mathbf{h}^0 \triangleq \begin{bmatrix} 1 & {\bf x}^{T} \end{bmatrix}^{T} \equiv \begin{bmatrix} 1 & x_1 & x_2 & \ldots & x_D \end{bmatrix}^{T} $;
# 2. $ \mathbf{h}^1 = \phi_1(\mathbf{a}^1) $;
# 3. $ \mathbf{a}^2 = {\mathbf{W}^2}^{T} \mathbf{h}^1 $;
# 4. $ \mathbf{h}^2 = \phi_2(\mathbf{a}^2) $;
# 5. $ \mathbf{a}^3 = {\mathbf{W}^3}^{T} \mathbf{h}^2 $;
# 6. $ \mathbf{h}^3 = \phi_3(\mathbf{a}^3) $;
# 7. $ \hat{y}={\mathbf{W}^4}^{T} \mathbf{h}^3 $.
# 
# We can summarize the feedforward network evaluation in the following nested expression
# $$
# \hat{y} = \underbrace{{\mathbf{W}^{4}}^{T} 
# \underbrace{\phi_3 \Big(
# \underbrace{{\mathbf{W}^{3}}^{T}
# \underbrace{\phi_2 \Big(
# \underbrace{{\mathbf{W}^{2}}^{T} 
# \underbrace{\phi_1 \Big(
# \underbrace{{\mathbf{W}^{1}}^{T} {\bf h}^0}_{\mathbf{a}^1}
# \Big)}_{\mathbf{h}^1=\phi_1(\mathbf{a}^1)}}_{\mathbf{a}^2= {\mathbf{W}^2}^{T} \mathbf{h}^1}
# \Big)}_{\mathbf{h}^2=\phi_2(\mathbf{a}^2)}}_{\mathbf{a}^3={\mathbf{W}^3}^{T} \mathbf{h}^2}
# \Big)}_{\mathbf{h}^3=\phi_3(\mathbf{a}^3)}}_{\hat{y}={\mathbf{W}^4}^{T} \mathbf{h}^3}
# $$
# in which the layer outputs $ {\bf h}^{1} $, $ {\bf h}^{2} $ and $ {\bf h}^{3} $ contain the intermediate results of the feedforward evaluation procedure. For convenience, the activations $ {\bf a}^{1} $, $ {\bf a}^{2} $ and $ {\bf a}^{3} $ are also stored for the backpropagation procedure.
# 
# Now, let the operator $\odot$ denote the element-wise multiplication such that
# :::{math}
# :label: odot
# \begin{equation}
# {\bf A} \odot {\bf B} \triangleq 
# \begin{bmatrix} 
# a_{1,1} b_{1,1} & a_{1,2} b_{1,2} & \ldots & a_{1,n} b_{1,n} \\
# a_{2,1} b_{2,1} & a_{2,2} b_{2,2} & \ldots & a_{2,n} b_{2,n} \\ 
# \vdots & \vdots & \ddots & \vdots \\
# a_{m,1} b_{m,1} & a_{m,2} b_{m,2} & \ldots & a_{m,n} b_{m,n} \\ 
# \end{bmatrix}^{T} 
# \end{equation}
# :::
# for any pair of $m \times n$ matrices $ {\bf A} $ and $ {\bf B} $. Furthermore, let us assume that the gradient $\nabla_{\hat{y}} L$ of the loss function w.r.t. to the network output $ \hat{y} $ is given in closed form and can be easily computed. Then, we can recursively compute the gradients from the output to the inputs -- backpropagation of error -- using the following steps
# 1. $\nabla_{{\bf h}^3} L = \left( {\mathbf{W}^4}^{T} \right)^{^T} \nabla_{\hat{y}} L \therefore \nabla_{{\bf h}^3} L = \mathbf{W}^4 \nabla_{\hat{y}} L$ in which $ \mathbf{W}^4 $ is the Jacobian of the affine transform (see {eq}`jacob_magic`);
# 2. $\nabla_{\mathbf{a}^3} L = \mathrm{diag}(\phi_{3}'(\mathbf{a}^3)) \nabla_{{\bf h}^3} L \therefore \nabla_{\mathbf{a}^3} L = \phi_{3}'(\mathbf{a}^3) \odot \nabla_{{\bf h}^3} L $ by combining {eq}`jacobian_magic2` and {eq}`odot`;
# 3. $\nabla_{{\bf h}^2} L = \mathbf{W}^3 \nabla_{\mathbf{a}^3} L$;
# 4. $\nabla_{\mathbf{a}^2} L = \phi_{2}'(\mathbf{a}^2) \odot \nabla_{{\bf h}^2} L$;
# 5. $\nabla_{{\bf h}^1} L = \mathbf{W}^2 \nabla_{\mathbf{a}^2} L$;
# 6. $\nabla_{\mathbf{a}^1} L = \phi_{1}'(\mathbf{a}^1) \odot \nabla_{{\bf h}^1} L$;
# 7. $\nabla_{{\bf h}^0} L = \mathbf{W}^1 \nabla_{\mathbf{a}^1} L$.
# 
# Alternatively, we can write
# $$
# \nabla_{{\bf h}^0} L = \underbrace{\mathbf{W}^1 \Big( 
# \underbrace{\phi_{1}'(\mathbf{a}^1) \odot \Big(
# \underbrace{\mathbf{W}^2 \Big( 
# \underbrace{\phi_{2}'(\mathbf{a}^2) \odot \Big(
# \underbrace{\mathbf{W}^3 \Big(
# \underbrace{\phi_{3}'(\mathbf{a}^3) \odot \Big(
# \underbrace{\mathbf{W}^4 \nabla_{\hat{y}} L}_{\nabla_{{\bf h}^3} L}
# \Big)}_{\nabla_{\mathbf{a}^3} L = \phi_{3}'(\mathbf{a}^3) \odot \nabla_{{\bf h}^3} L}
# \Big)}_{\nabla_{{\bf h}^2} L = \mathbf{W}^3 \nabla_{\mathbf{a}^3} L}
# \Big)}_{\nabla_{\mathbf{a}^2} L = \phi_{2}'(\mathbf{a}^2) \odot \nabla_{{\bf h}^2} L}
# \Big)}_{\nabla_{{\bf h}^1} L = \mathbf{W}^2 \nabla_{\mathbf{a}^2} L}
# \Big)}_{\nabla_{\mathbf{a}^1} L = \phi_{1}'(\mathbf{a}^1) \odot \nabla_{{\bf h}^1} L}
# \Big)}_{\nabla_{{\bf h}^0} L = \mathbf{W}^1 \nabla_{\mathbf{a}^1} L}
# $$
# to stress how gradients of the loss function $L$ are backpropagated to compute the intermediate gradients $ \nabla_{{\bf h}^3} L $, $ \nabla_{{\bf h}^2} L$, $ \nabla_{{\bf h}^1} L $ and $ \nabla_{{\bf h}^0} L $ from the gradient $\nabla_{\hat{y}} L$. Note that the activation vectors $ {\bf a}^{1} $, $ {\bf a}^{2} $ and $ {\bf a}^{3} $ obtained in the feedforward evaluation procedure are required to compute the gradients $ \phi_{1}'({\bf a}^{1}) $, $ \phi_{2}'({\bf a}^{2}) $ and $ \phi_{3}'({\bf a}^{3}) $.
# 
# In the sequel, we compute the gradient of the loss $L$ w.r.t. the network parameters $ {\bf W}^{1} $, $ {\bf W}^{2} $, $ {\bf W}^{3} $ and $ {\bf W}^{4} $ are computed as
# * $ {\partial L \over \partial w_{i,1}^{4}} = {\partial L \over \partial \hat{y}} {\partial \hat{y} \over \partial w_{i,1}^{4}} = {\partial L \over \partial \hat{y}} h_{i}^{3} \,\,\,\,\,\therefore\,\, \nabla_{{\bf W}^{4}} L = {\bf h}^{3} \, \nabla_{\hat{y}} L$;
# * $ {\partial L \over \partial w_{i,j}^{3}} = {\partial L \over \partial a_{j}^{3}} {\partial a_{j}^{3} \over \partial w_{i,j}^{3}} = {\partial L \over \partial a_{j}^{3}} h_{i}^{2} \,\,\therefore\,\, \nabla_{{\bf W}^{3}} L = {\bf h}^{2} \left( \nabla_{\mathbf{a}^{3}} L \right)^{T}$; (outer product, check this as an exercise)
# * $ {\partial L \over \partial w_{i,j}^{2}} = {\partial L \over \partial a_{j}^{2}} {\partial a_{j}^{2} \over \partial w_{i,j}^{2}} = {\partial L \over \partial a_{j}^{2}} h_{i}^{1} \,\,\therefore\,\, \nabla_{{\bf W}^{2}} L = {\bf h}^{1} \left( \nabla_{\mathbf{a}^{2}} L \right)^{T}$;
# * $ {\partial L \over \partial w_{i,j}^{1}} = {\partial L \over \partial a_{j}^{1}} {\partial a_{j}^{1} \over \partial w_{i,j}^{1}} = {\partial L \over \partial a_{j}^{1}} h_{i}^{0} \,\,\therefore\,\, \nabla_{{\bf W}^{1}} L = {\bf h}^{0} \left( \nabla_{\mathbf{a}^{1}} L \right)^{T}$.
# Note therefore that we also need to bookkeep the hidden layers' outputs $ {\bf h}^{1} $, $ {\bf h}^{2} $ and $ {\bf h}^{3} $ from the feedforward evaluation procedure as weel as the gradients $ \nabla_{\mathbf{a}^{1}} L $, $ \nabla_{\mathbf{a}^{2}} L $, $ \nabla_{\mathbf{a}^{3}} L$ and $\nabla_{\hat{y}} L$ from the backpropagation procedure. 
# 
# The feedforward evaluation and backpropagation procedures are performed for each training example $ \left( {\bf x}_{i}, y_{i} \right) \in {\cal D}_{train} $. Finally, we compute the gradient of the empirical training loss $ L_{train} $ w.r.t. parameters $ {\bf W}^{\ell} $ as
# \begin{equation}
# \nabla_{{\bf W}^{\ell}} L_{train} = \frac{1}{N} \sum_{i=1}^{N} \nabla_{{\bf W}^{\ell}} L (y_{i}, \hat{y}_{i})
# \end{equation}
# and update the parameters at the $\ell$-th layer as
# \begin{equation}
# {\bf W}^{\ell} \gets {\bf W}^{\ell} - \eta_{k} \nabla_{{\bf W}^{\ell}} L_{train},
# \end{equation}
# where $ \eta_{k} $ is the learning rate used in the $ k $-th step of the standard gradient descent method, a.k.a. batch gradient descent as it employs the whole training dataset $ {\cal D}_{train} $ at each step $ k $ to update the network parameters.
# ::::
# 
# :::{tip}
# We can empirically check the correctness of our analytical derivations of the derivatives employed by the backprop procedure. More precisely, let $f({\bf x}) = f(x_1, x_2, \dots, x_D)$ be a $D$-dimensional function. Its gradient
# \begin{equation}
# \nabla_{{\bf x}} f = \begin{bmatrix} {\partial f \over \partial x_1} & {\partial f \over \partial x_2} & \ldots & {\partial f \over \partial x_D} \end{bmatrix}^{T}
# \end{equation}
# w.r.t. $ {\bf x} $ can be approximated -- at each dimension $ i $ -- using finite differences as
# \begin{equation}
# \left. {\partial f \over \partial x_i} \right|_{{\bf x}} \approx  {f(x_1, \dots, x_i + \delta, \dots, x_D) - f(x_1, \dots, x_i, \dots, x_D) \over \delta}
# \end{equation}
# and then compare this approximation with results obtained by the closed form expression for the partial derivatives $ \lbrace  {\partial f \over \partial x_i} \rbrace $. Note though that the approximations are computed around a given $ {\bf x} $, i.e. the aproximation to $ \nabla_{{\bf x}} f $ is valid for a particular input vector $ {\bf x} $. For $\delta \rightarrow 0$, the approximation becomes (by definition) the partial derivative. Besides being usefull for double checking analytical derivations using some data samples, this approximation is too expensive to use it in the backprop procedure itself. Thus, activation functions with closed form expressions for their derivatives are still required by the network to efficiently learn. 
# :::
