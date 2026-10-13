(nn_impl_exercises)=
# Exercises

These exercises teach you how a neural network works and how to implement one yourself, in Numpy. Doing so should give the a firm grasp of the theory and the implementation details. First, you do the computations by hand on very small networks. Then, you implement a full network in numpyAt the end you can find some related practice questions.

Some notation first. A network with $L$ hidden layers computes
\begin{align*}
\vvec{h}^{(0)} &= \vvec{x},\\
\vvec{h}^{(\ell)} &= \phi_\ell(W^{(\ell)}\vvec{h}^{(\ell-1)}+\vvec{b}^{(\ell)}), \quad 1\leq \ell\leq L,\\
f_\theta(\vvec{x}) &= \softmax(W\vvec{h}^{(L)}+\vvec{b}),
\end{align*}
where $W^{(\ell)}\in\mathbb{R}^{d_\ell\times d_{\ell-1}}$, $\vvec{b}^{(\ell)}\in\mathbb{R}^{d_\ell}$, $d_0=d$, and $W\in\mathbb{R}^{c\times d_L}$, $\vvec{b}\in\mathbb{R}^c$ define the softmax regression head. The vector $\vvec{z}=W\vvec{h}^{(L)}+\vvec{b}$ holds the logits. The loss is the average cross entropy
$$\mathcal{L}(\mathcal{D},\theta) = \frac{1}{n}\sum_{i=1}^n CE(y_i,f_\theta(\vvec{x}_i)), \qquad CE(y,\vvec{p}) = -\log(p_y).$$
For the backward pass, we give a name to the input of each activation function:
$$\vvec{a}^{(\ell)} = W^{(\ell)}\vvec{h}^{(\ell-1)}+\vvec{b}^{(\ell)}, \qquad \vvec{h}^{(\ell)} = \phi_\ell(\vvec{a}^{(\ell)}).$$

## Softmax and Cross Entropy
A softmax regression classifier has $c=3$ classes and the parameters
$$W = \begin{pmatrix}2 & -1\\ 0 & 1\\ 1 & 1\end{pmatrix},\qquad \vvec{b} = \begin{pmatrix}0\\1\\-2\end{pmatrix}.$$
Consider the point $\vvec{x}=(1,2)$.

### Logits, Probabilities and Prediction
Compute the logits $\vvec{z}=W\vvec{x}+\vvec{b}$ and the softmax probabilities $\softmax(\vvec{z})$. Which class does the classifier predict? Compute the cross entropy $CE(y,\softmax(\vvec{z}))$ for the true label $y=3$ and for the true label $y=2$.

````{dropdown} Solution
The logits are
$$\vvec{z} = W\vvec{x}+\vvec{b} = \begin{pmatrix}2\cdot 1-1\cdot 2\\ 0\cdot 1+1\cdot 2\\ 1\cdot 1+1\cdot 2\end{pmatrix}+\begin{pmatrix}0\\1\\-2\end{pmatrix} = \begin{pmatrix}0\\3\\1\end{pmatrix}.$$
The softmax divides the exponential of each logit by the sum of all exponentials:
$$\exp(0)+\exp(3)+\exp(1) \approx 1 + 20.09 + 2.72 = 23.80,$$
$$\softmax(\vvec{z}) \approx \begin{pmatrix}1/23.80\\ 20.09/23.80\\ 2.72/23.80\end{pmatrix} \approx \begin{pmatrix}0.042\\0.844\\0.114\end{pmatrix}.$$
The classifier predicts the class with the highest probability, which is also the class with the highest logit: class $2$.

The cross entropy is the negative logarithm of the probability of the true class:
\begin{align*}
y=3:&\quad CE(3,\softmax(\vvec{z})) = -\log(0.114) \approx 2.170,\\
y=2:&\quad CE(2,\softmax(\vvec{z})) = -\log(0.844) \approx 0.170.
\end{align*}
The loss is small if the classifier gives a high probability to the true class. The loss is large if the classifier gives a low probability to the true class.
````

### Shifting the Logits
Show that $\softmax(\vvec{z}+a\vvec{1}) = \softmax(\vvec{z})$ for every $a\in\mathbb{R}$. Then explain why implementations compute $\softmax(\vvec{z}-\max_j z_j\vvec{1})$ instead of $\softmax(\vvec{z})$. Use the logits $\vvec{z}=(1000, 1001)$ as an example.

````{dropdown} Solution
For every class $y$, we have
$$\softmax(\vvec{z}+a\vvec{1})_y = \frac{\exp(z_y+a)}{\sum_{j=1}^c\exp(z_j+a)} = \frac{\exp(a)\exp(z_y)}{\exp(a)\sum_{j=1}^c\exp(z_j)} = \softmax(\vvec{z})_y.$$
Thus, we can subtract any constant from all logits and the probabilities do not change.

A computer stores numbers with a finite range. In 64-bit floating point, $\exp(1000)$ is too large and becomes `inf`. The direct computation gives `inf/inf = nan` for both classes. If we subtract the maximum $1001$ first, we get the logits $(-1,0)$ and
$$\softmax(-1,0) = \left(\frac{\exp(-1)}{\exp(-1)+1},\frac{1}{\exp(-1)+1}\right)\approx(0.269, 0.731).$$
After the shift, all exponents are at most zero. Thus, no exponential becomes too large. The largest term in the sum is $\exp(0)=1$, so we never divide by zero.
````

## Activation Functions
Consider a network with one hidden layer and the parameters
$$W^{(1)} = \begin{pmatrix}1 & 2\\0 & 1\end{pmatrix},\quad \vvec{b}^{(1)} = \begin{pmatrix}1\\-1\end{pmatrix},\quad W^{(2)} = \begin{pmatrix}1 & -1\end{pmatrix},\quad b^{(2)} = 2.$$

### Affine Layers
Assume that the network has no activation function: $f(\vvec{x}) = W^{(2)}(W^{(1)}\vvec{x}+\vvec{b}^{(1)})+b^{(2)}$. Compute $\tilde{W}$ and $\tilde{b}$ such that $f(\vvec{x}) = \tilde{W}\vvec{x}+\tilde{b}$.

````{dropdown} Solution
As in the chapter, we have $\tilde{W} = W^{(2)}W^{(1)}$ and $\tilde{b} = W^{(2)}\vvec{b}^{(1)}+b^{(2)}$:
$$\tilde{W} = \begin{pmatrix}1 & -1\end{pmatrix}\begin{pmatrix}1 & 2\\0 & 1\end{pmatrix} = \begin{pmatrix}1 & 1\end{pmatrix},\qquad \tilde{b} = 1\cdot 1 + (-1)\cdot(-1) + 2 = 4.$$
Thus, $f(\vvec{x}) = x_1+x_2+4$. The two layers compute the same function as one affine layer.
````

### The Effect of ReLU
Now apply the ReLU activation after the first layer: $f_{relu}(\vvec{x}) = W^{(2)}ReLU(W^{(1)}\vvec{x}+\vvec{b}^{(1)})+b^{(2)}$. Compute $f(\vvec{x})$ and $f_{relu}(\vvec{x})$ for $\vvec{x}=(0,2)$ and for $\vvec{x}=(1,-2)$. Explain the result. Why can a softmax regression on top of affine layers without activations not separate the points inside a circle from the points on a ring around the circle?

````{dropdown} Solution
For $\vvec{x}=(0,2)$, we have $W^{(1)}\vvec{x}+\vvec{b}^{(1)} = (4+1, 2-1) = (5,1)$. Both entries are positive, so the ReLU does not change them:
$$f(\vvec{x}) = f_{relu}(\vvec{x}) = 5-1+2 = 6.$$
For $\vvec{x}=(1,-2)$, we have $W^{(1)}\vvec{x}+\vvec{b}^{(1)} = (1-4+1, -2-1) = (-2,-3)$. The ReLU sets both entries to zero:
$$f(\vvec{x}) = -2+3+2 = 3,\qquad f_{relu}(\vvec{x}) = 0-0+2 = 2.$$
The two functions are equal where all hidden neurons are active, and different elsewhere. The ReLU network is piecewise affine: each neuron changes the function where its affine function crosses zero.

Without activations, all layers together compute one affine function of $\vvec{x}$. A softmax regression on an affine function of $\vvec{x}$ has decision boundaries that are hyperplanes in the space of $\vvec{x}$. Thus, the region of each class is an intersection of half-spaces, which is a convex set. The ring is not convex, so it cannot be the region of one class. The nonlinear activations make it possible to learn a feature transformation in which the classes are linearly separable.
````

## Shapes and Parameter Counts
Consider two networks with ReLU activations:
1. A network for points in the plane with $d=2$, two hidden layers with $d_1=d_2=32$, and $c=3$ classes.
2. A network for grayscale images of $16\times 16$ pixels with one hidden layer with $d_1=64$ and $c=10$ classes. The network gets the image as a vector with $256$ entries.

### Shapes of the Parameters
Give the shape of every weight matrix and bias vector, and the number of parameters of each network.

````{dropdown} Solution
A layer from $d_{\ell-1}$ to $d_\ell$ dimensions has a weight matrix $W^{(\ell)}\in\mathbb{R}^{d_\ell\times d_{\ell-1}}$ and a bias $\vvec{b}^{(\ell)}\in\mathbb{R}^{d_\ell}$. Thus, it has $d_\ell d_{\ell-1}+d_\ell$ parameters.

**Network 1.**

| Layer | Weights | Bias | Parameters |
|---|---|---|---|
| hidden 1 | $W^{(1)}\in\mathbb{R}^{32\times 2}$ | $\vvec{b}^{(1)}\in\mathbb{R}^{32}$ | $64+32=96$ |
| hidden 2 | $W^{(2)}\in\mathbb{R}^{32\times 32}$ | $\vvec{b}^{(2)}\in\mathbb{R}^{32}$ | $1024+32=1056$ |
| head | $W\in\mathbb{R}^{3\times 32}$ | $\vvec{b}\in\mathbb{R}^{3}$ | $96+3=99$ |

In total, network 1 has $1251$ parameters.

**Network 2.**

| Layer | Weights | Bias | Parameters |
|---|---|---|---|
| hidden 1 | $W^{(1)}\in\mathbb{R}^{64\times 256}$ | $\vvec{b}^{(1)}\in\mathbb{R}^{64}$ | $16384+64=16448$ |
| head | $W\in\mathbb{R}^{10\times 64}$ | $\vvec{b}\in\mathbb{R}^{10}$ | $640+10=650$ |

In total, network 2 has $17098$ parameters. Most parameters are in the first layer, because the input has many dimensions.
````

### Shapes for a Batch of Points
In an implementation, we process $n$ points at the same time. The data matrix $D\in\mathbb{R}^{n\times d}$ holds one point per row, $D_{i\cdot}=\vvec{x}_i^\top$. We write the hidden layer outputs of all points as rows of a matrix $H^{(\ell)}$:
$$H^{(0)}=D,\qquad A^{(\ell)} = H^{(\ell-1)}W^{(\ell)\top}+\vvec{1}\vvec{b}^{(\ell)\top},\qquad H^{(\ell)} = \phi_\ell(A^{(\ell)}).$$
The logits are $Z = H^{(L)}W^\top+\vvec{1}\vvec{b}^\top$, and $P$ applies the softmax to each row of $Z$. Show that row $i$ of $H^{(1)}$ is $\vvec{h}^{(1)}(\vvec{x}_i)^\top$. Then give the shapes of $A^{(1)}$, $H^{(1)}$, $H^{(2)}$, $Z$ and $P$ for network 1 and $n=600$.

````{dropdown} Solution
Row $i$ of $DW^{(1)\top}$ is $D_{i\cdot}W^{(1)\top} = \vvec{x}_i^\top W^{(1)\top} = (W^{(1)}\vvec{x}_i)^\top$. Row $i$ of $\vvec{1}\vvec{b}^{(1)\top}$ is $\vvec{b}^{(1)\top}$. Thus, row $i$ of $A^{(1)}$ is $(W^{(1)}\vvec{x}_i+\vvec{b}^{(1)})^\top = \vvec{a}^{(1)}(\vvec{x}_i)^\top$. The activation is applied element-wise, so row $i$ of $H^{(1)}$ is $\vvec{h}^{(1)}(\vvec{x}_i)^\top$.

For network 1 and $n=600$:
$$A^{(1)},H^{(1)},H^{(2)}\in\mathbb{R}^{600\times 32},\qquad Z,P\in\mathbb{R}^{600\times 3}.$$
In numpy, the term $\vvec{1}\vvec{b}^{(\ell)\top}$ is broadcasting: `H @ W.T + b` adds `b` to every row.
````

## A Forward Pass by Hand
Consider a network with one hidden layer, ReLU activation, $d=2$, $d_1=2$ and $c=2$. The parameters are
$$W^{(1)} = \begin{pmatrix}1 & -1\\2 & 1\end{pmatrix},\quad \vvec{b}^{(1)} = \begin{pmatrix}0\\-2\end{pmatrix},\quad W = \begin{pmatrix}1 & 0\\-1 & 1\end{pmatrix},\quad \vvec{b} = \begin{pmatrix}0\\1\end{pmatrix}.$$
Compute $\vvec{a}^{(1)}$, $\vvec{h}^{(1)}$, the logits $\vvec{z}$, the probabilities $f_\theta(\vvec{x})$ and the cross entropy for the point $\vvec{x}=(1,-1)$ with the label $y=2$.

````{dropdown} Solution
\begin{align*}
\vvec{a}^{(1)} &= W^{(1)}\vvec{x}+\vvec{b}^{(1)} = \begin{pmatrix}1+1\\2-1\end{pmatrix}+\begin{pmatrix}0\\-2\end{pmatrix} = \begin{pmatrix}2\\-1\end{pmatrix},\\
\vvec{h}^{(1)} &= ReLU(\vvec{a}^{(1)}) = \begin{pmatrix}2\\0\end{pmatrix},\\
\vvec{z} &= W\vvec{h}^{(1)}+\vvec{b} = \begin{pmatrix}2\\-2\end{pmatrix}+\begin{pmatrix}0\\1\end{pmatrix} = \begin{pmatrix}2\\-1\end{pmatrix},\\
f_\theta(\vvec{x}) &= \softmax(\vvec{z}) = \frac{1}{\exp(2)+\exp(-1)}\begin{pmatrix}\exp(2)\\\exp(-1)\end{pmatrix} \approx \begin{pmatrix}0.953\\0.047\end{pmatrix}.
\end{align*}
The true label is $y=2$, so
$$CE(2,f_\theta(\vvec{x})) = -\log(0.047) \approx 3.049.$$
The network gives a probability of only $4.7\%$ to the true class. Thus, the loss is high. In the next exercises, we compute the gradient that tells us how to change the parameters.
````

## The Gradient of the Cross Entropy
The chapter states that for $f(\vvec{x})=\softmax(\vvec{z})$
$$\frac{\partial CE(y,f(\vvec{x}))}{\partial \vvec{z}} = f(\vvec{x})^\top-\vvec{e}_y^\top.$$
Prove this statement. Use that $CE(y,\softmax(\vvec{z})) = -z_y+\log\sum_{j=1}^c\exp(z_j)$. Then show that the entries of this Jacobian sum to zero.

````{dropdown} Solution
From the definition of the softmax, we get
$$CE(y,\softmax(\vvec{z})) = -\log\frac{\exp(z_y)}{\sum_{j=1}^c\exp(z_j)} = -z_y+\log\sum_{j=1}^c\exp(z_j).$$
We compute the partial derivative subject to $z_k$. The first term gives $-1$ if $k=y$ and $0$ otherwise. For the second term, we apply the chain rule:
$$\frac{\partial}{\partial z_k}\log\sum_{j=1}^c\exp(z_j) = \frac{\exp(z_k)}{\sum_{j=1}^c\exp(z_j)} = \softmax(\vvec{z})_k = f(\vvec{x})_k.$$
Thus,
$$\frac{\partial CE(y,f(\vvec{x}))}{\partial z_k} = \begin{cases}f(\vvec{x})_k-1 & \text{ if } k=y\\ f(\vvec{x})_k & \text{ otherwise}\end{cases}$$
and the Jacobian is the row vector $f(\vvec{x})^\top-\vvec{e}_y^\top$.

The probabilities sum to one, so the entries sum to $\sum_{k=1}^c f(\vvec{x})_k - 1 = 0$. The gradient increases the logit of the true class and decreases the other logits by the same total amount. The gradient is "prediction minus target". This is the same pattern as the residual in the gradient of the RSS in regression.
````

## A Backward Pass by Hand
Use the network, the point and the label from the forward pass exercise above.

### Backpropagation Through the Network
Compute the gradients of $CE(y,f_\theta(\vvec{x}))$ subject to $W$, $\vvec{b}$, $W^{(1)}$ and $\vvec{b}^{(1)}$. Use the chain rule with the Jacobians from the chapter:
$$\frac{\partial W\vvec{h}+\vvec{b}}{\partial \vvec{h}} = W,\qquad \frac{\partial W\vvec{h}+\vvec{b}}{\partial W_{jk}} = \vvec{e}_jh_k,\qquad \frac{\partial ReLU(\vvec{a})}{\partial \vvec{a}} = \diag(ReLU'(a_1),\ldots,ReLU'(a_{d_1})).$$
Which entries of $W^{(1)}$ and $\vvec{b}^{(1)}$ get a zero gradient? Why?

```{tip}
:class: dropdown
The Jacobian of $\vvec{z}=W\vvec{h}+\vvec{b}$ subject to $\vvec{h}$ has the shape $c\times d_1$, as for the function $\vvec{b}-A\vvec{x}$ in the optimization exercises. Use the dimensions to see the order of the factors in the chain rule.
```

````{dropdown} Solution
**Output layer.** From the cross entropy exercise, the Jacobian subject to the logits is
$$\vvec{g}^\top := \frac{\partial CE}{\partial \vvec{z}} = f_\theta(\vvec{x})^\top-\vvec{e}_2^\top \approx (0.953, -0.953).$$
With $\frac{\partial \vvec{z}}{\partial W_{jk}} = \vvec{e}_jh^{(1)}_k$ we get $\frac{\partial CE}{\partial W_{jk}} = \vvec{g}^\top\vvec{e}_jh^{(1)}_k = g_jh^{(1)}_k$. Thus, the gradient is the outer product
$$\nabla_W CE = \vvec{g}\vvec{h}^{(1)\top} = \begin{pmatrix}0.953\\-0.953\end{pmatrix}\begin{pmatrix}2 & 0\end{pmatrix} \approx \begin{pmatrix}1.905 & 0\\-1.905 & 0\end{pmatrix},\qquad \nabla_{\vvec{b}} CE = \vvec{g} \approx \begin{pmatrix}0.953\\-0.953\end{pmatrix}.$$

**Hidden layer.** We go back through the head and the ReLU:
$$\frac{\partial CE}{\partial \vvec{a}^{(1)}} = \underbrace{\frac{\partial CE}{\partial \vvec{z}}}_{1\times 2}\underbrace{\frac{\partial \vvec{z}}{\partial \vvec{h}^{(1)}}}_{2\times 2}\underbrace{\frac{\partial \vvec{h}^{(1)}}{\partial \vvec{a}^{(1)}}}_{2\times 2} = \vvec{g}^\top W\diag(ReLU'(\vvec{a}^{(1)})).$$
We have
$$\vvec{g}^\top W \approx (0.953, -0.953)\begin{pmatrix}1 & 0\\-1 & 1\end{pmatrix} = (1.905, -0.953).$$
Since $\vvec{a}^{(1)}=(2,-1)$, we have $\diag(ReLU'(\vvec{a}^{(1)})) = \diag(1,0)$. Thus,
$$\vvec{\delta}^\top := \frac{\partial CE}{\partial \vvec{a}^{(1)}} \approx (1.905, 0).$$
The same argument as for the output layer gives
$$\nabla_{W^{(1)}} CE = \vvec{\delta}\vvec{x}^\top \approx \begin{pmatrix}1.905\\0\end{pmatrix}\begin{pmatrix}1 & -1\end{pmatrix} = \begin{pmatrix}1.905 & -1.905\\0 & 0\end{pmatrix},\qquad \nabla_{\vvec{b}^{(1)}} CE = \vvec{\delta} \approx \begin{pmatrix}1.905\\0\end{pmatrix}.$$

**Zero gradients.** The second hidden neuron has $a^{(1)}_2=-1<0$, so its output is zero and $ReLU'(a^{(1)}_2)=0$. Thus, the second row of $W^{(1)}$ and the second entry of $\vvec{b}^{(1)}$ get a zero gradient. The second column of $W$ also gets a zero gradient, because it multiplies $h^{(1)}_2=0$. A neuron that is inactive for all training points never gets a gradient. This is the dead neuron effect of the chapter.
````

### One Gradient Descent Step
Do one gradient descent step with step size $\eta=0.1$ on all four parameters. Compute the new probabilities and the new cross entropy. Did the loss decrease?

````{dropdown} Solution
The new parameters are
$$W^{(1)} \gets \begin{pmatrix}0.810 & -0.810\\2 & 1\end{pmatrix},\quad \vvec{b}^{(1)}\gets\begin{pmatrix}-0.191\\-2\end{pmatrix},\quad W\gets\begin{pmatrix}0.810 & 0\\-0.810 & 1\end{pmatrix},\quad \vvec{b}\gets\begin{pmatrix}-0.095\\1.095\end{pmatrix}.$$
The new forward pass gives $\vvec{a}^{(1)}\approx(1.429,-1)$, $\vvec{h}^{(1)}\approx(1.429,0)$ and
$$\vvec{z} \approx \begin{pmatrix}0.810\cdot 1.429-0.095\\-0.810\cdot 1.429+1.095\end{pmatrix}\approx\begin{pmatrix}1.061\\-0.061\end{pmatrix},\qquad f_\theta(\vvec{x})\approx\begin{pmatrix}0.754\\0.246\end{pmatrix}.$$
The new loss is $-\log(0.246)\approx 1.404$. The loss decreased from $3.049$ to $1.404$ in one step. The probability of the true class increased from $0.047$ to $0.246$.
````

## Batch Training
An implementation computes the gradient of the average loss over $n$ points at the same time. Use the batch notation from the shapes exercise and the one-hot label matrix $Y\in\mathbb{1}^{n\times c}$ with $Y_{i\cdot}=\vvec{e}_{y_i}^\top$. Let
$$G = \frac{1}{n}(P-Y)\in\mathbb{R}^{n\times c}.$$
Show that
$$\nabla_W\mathcal{L}(\mathcal{D},\theta) = G^\top H^{(L)},\qquad \nabla_{\vvec{b}}\mathcal{L}(\mathcal{D},\theta) = G^\top\vvec{1}.$$
Then show that the gradient subject to the hidden layer $L$ follows from
$$G^{(L)} = (GW)\circ\phi_L'(A^{(L)}),\qquad \nabla_{W^{(L)}}\mathcal{L}(\mathcal{D},\theta) = G^{(L)\top}H^{(L-1)},\qquad \nabla_{\vvec{b}^{(L)}}\mathcal{L}(\mathcal{D},\theta) = G^{(L)\top}\vvec{1},$$
where $\circ$ is the element-wise product. Compare $\nabla_W\mathcal{L}$ with the gradient of the RSS in regression, $\nabla_{\bm\beta}RSS(\bm\beta) = 2X^\top(X\bm\beta-\vvec{y})$.

````{dropdown} Solution
The loss is the average of the point losses. The gradient is linear, so
$$\nabla_W\mathcal{L}(\mathcal{D},\theta) = \frac{1}{n}\sum_{i=1}^n\nabla_W CE(y_i,f_\theta(\vvec{x}_i)) = \frac{1}{n}\sum_{i=1}^n\left(f_\theta(\vvec{x}_i)-\vvec{e}_{y_i}\right)\vvec{h}^{(L)}(\vvec{x}_i)^\top,$$
where we use the outer product from the backward pass exercise. Row $i$ of $G$ is $\frac{1}{n}(f_\theta(\vvec{x}_i)-\vvec{e}_{y_i})^\top$ and row $i$ of $H^{(L)}$ is $\vvec{h}^{(L)}(\vvec{x}_i)^\top$. A matrix product $G^\top H^{(L)}$ is the sum of the outer products of the rows of $G$ and $H^{(L)}$. Thus, $\nabla_W\mathcal{L} = G^\top H^{(L)}$. In the same way, $\nabla_{\vvec{b}}\mathcal{L} = \sum_i G_{i\cdot}^\top = G^\top\vvec{1}$.

For one point, the backward pass exercise gives $\vvec{\delta}^\top = \vvec{g}^\top W\diag(\phi_L'(\vvec{a}^{(L)}))$. Multiplication with a diagonal matrix from the right multiplies entry $k$ with $\phi_L'(a^{(L)}_k)$. Thus, $\vvec{\delta}^\top = (\vvec{g}^\top W)\circ\phi_L'(\vvec{a}^{(L)})^\top$. If we stack these rows for all points, we get $G^{(L)} = (GW)\circ\phi_L'(A^{(L)})$. The gradients of $W^{(L)}$ and $\vvec{b}^{(L)}$ follow as for the head, with $G^{(L)}$ in place of $G$ and $H^{(L-1)}$ in place of $H^{(L)}$. The same step repeats for every earlier layer:
$$G^{(\ell-1)} = (G^{(\ell)}W^{(\ell)})\circ\phi_{\ell-1}'(A^{(\ell-1)}).$$

**Comparison with regression.** The transpose of the head gradient is $\frac1n H^{(L)\top}(P-Y)$. The RSS gradient is $2X^\top(X\bm\beta-\vvec{y})$. Both have the form "the input of the layer, transposed, times (prediction minus target)". The head of a neural network is a linear model on the learned features $H^{(L)}$. The difference is that the network also learns $H^{(L)}$ with the hidden layers.
````

## The Vanishing Gradient Problem
The sigmoid function is $\sigma(x) = \frac{1}{1+\exp(-x)}$.

### The Derivative of the Sigmoid
Show that $\sigma'(x) = \sigma(x)(1-\sigma(x))$. What is the maximum value of $\sigma'$?

````{dropdown} Solution
With the chain rule, we get
$$\sigma'(x) = \frac{\exp(-x)}{(1+\exp(-x))^2} = \frac{1}{1+\exp(-x)}\cdot\frac{\exp(-x)}{1+\exp(-x)} = \sigma(x)(1-\sigma(x)),$$
because $\frac{\exp(-x)}{1+\exp(-x)} = 1-\frac{1}{1+\exp(-x)}$. The function $s(1-s)$ has its maximum at $s=\frac12$. Since $\sigma(0)=\frac12$, the maximum is $\sigma'(0)=\frac14$.
````

### A Deep Chain of Sigmoids
Consider a network with one neuron per layer: $h^{(\ell)} = \sigma(w_\ell h^{(\ell-1)}+b_\ell)$ for $1\leq\ell\leq L$. Give an upper bound for $\left\lvert\frac{\partial h^{(L)}}{\partial h^{(0)}}\right\rvert$. Compute the bound for $L=10$ and $\lvert w_\ell\rvert = 1$ and for $L=10$ and $\lvert w_\ell\rvert = 2$. What is the corresponding derivative for ReLU activations if all neurons are active? What does this mean for the gradient of the weights in the first layers?

````{dropdown} Solution
With the chain rule, we get
$$\frac{\partial h^{(L)}}{\partial h^{(0)}} = \prod_{\ell=1}^L\frac{\partial h^{(\ell)}}{\partial h^{(\ell-1)}} = \prod_{\ell=1}^L w_\ell\,\sigma'(a^{(\ell)}),\qquad\text{thus}\qquad \left\lvert\frac{\partial h^{(L)}}{\partial h^{(0)}}\right\rvert\leq\prod_{\ell=1}^L\frac{\lvert w_\ell\rvert}{4}.$$
For $L=10$ and $\lvert w_\ell\rvert=1$, the bound is $4^{-10}\approx 9.5\cdot10^{-7}$. For $\lvert w_\ell\rvert=2$, the bound is $2^{-10}\approx 9.8\cdot10^{-4}$. The bound is reached only if all $a^{(\ell)}=0$. Large weights push $a^{(\ell)}$ to the flat parts of the sigmoid, where $\sigma'$ is much smaller than $\frac14$.

For ReLU and active neurons, $ReLU'(a^{(\ell)})=1$ and the derivative is $\prod_\ell w_\ell$. The activations do not make it smaller.

The gradient of a weight in layer $\ell$ contains the product of the derivatives of all layers after layer $\ell$. With sigmoid activations, this product becomes very small for the first layers. Thus, gradient descent changes the first layers very slowly. This is the vanishing gradient problem. The practical exercise below shows it in a real network.
````

## SGD: Batches and Epochs
A training set has $n=1200$ points. We train with SGD, batch size $\lvert\mathcal{B}\rvert=32$ and $50$ epochs.

### Counting the Updates
How many parameter updates does SGD do in one epoch, and in total? How many updates does gradient descent do in $50$ epochs? How often does each method use each training point?

````{dropdown} Solution
We have $1200 = 37\cdot 32+16$. Thus, one epoch has $37$ full batches and one batch with $16$ points, which is $38$ updates. In $50$ epochs, SGD does $50\cdot 38=1900$ updates. Gradient descent uses all points for one update, so it does $50$ updates in $50$ epochs. Both methods use each training point once per epoch, so the computational cost per epoch is approximately the same. SGD does many more, but noisier, updates for the same cost.
````

### The Batch Gradient
Assume that SGD uses a batch with one point $\mathcal{B}=\{(\vvec{x}_i,y_i)\}$, where $i$ is drawn uniformly at random from $\{1,\ldots,n\}$. Show that the expected value of $\nabla\mathcal{L}(\mathcal{B},\theta)$ is $\nabla\mathcal{L}(\mathcal{D},\theta)$. Explain why SGD needs a decreasing step size to converge.

````{dropdown} Solution
Each index has the probability $\frac1n$. Thus,
$$\mathbb{E}_i\left[\nabla\mathcal{L}(\{(\vvec{x}_i,y_i)\},\theta)\right] = \sum_{i=1}^n\frac{1}{n}\nabla CE(y_i,f_\theta(\vvec{x}_i)) = \nabla\mathcal{L}(\mathcal{D},\theta).$$
On average, an SGD step goes in the direction of a gradient descent step. But each single batch gradient is different from the full gradient. At a minimizer of $\mathcal{L}(\mathcal{D},\theta)$, the full gradient is zero, but the batch gradients are usually not zero. With a constant step size, SGD continues to move around the minimizer. A decreasing step size makes these movements smaller, so that SGD can converge.
````

## Weight Decay
With weight decay, we minimize $\mathcal{L}(\mathcal{D},\theta)+\lambda\lVert\theta\rVert^2$.

### The Update Rule
Derive the gradient descent update with step size $\eta$. Explain the name "weight decay". Assume that $\eta=0.1$ and $\lambda=0.01$. If the loss gradient is zero, by which factor do the weights change in $1000$ steps? What do you expect if $\lambda$ is very large?

````{dropdown} Solution
The gradient of $\lambda\lVert\theta\rVert^2$ is $2\lambda\theta$. Thus,
$$\theta \gets \theta-\eta\left(\nabla\mathcal{L}(\mathcal{D},\theta)+2\lambda\theta\right) = (1-2\eta\lambda)\,\theta-\eta\nabla\mathcal{L}(\mathcal{D},\theta).$$
Each step first multiplies the weights with $1-2\eta\lambda<1$. The weights "decay" to zero if the loss gradient does not push them back. For $\eta=0.1$ and $\lambda=0.01$, the factor is $0.998$ per step and $0.998^{1000}\approx 0.135$ after $1000$ steps.

If $\lambda$ is very large, the penalty is more important than the loss. Then all weights go to approximately zero. The network then returns approximately the same logits for every input, and the predicted probabilities are approximately uniform. The loss is then approximately $\log c$, and the network is underfitting.
````

## Dropout by Hand
A hidden layer has the output $\vvec{h}^{(\ell)} = (2, 0.5, 1, 3)$. A dropout layer with $p=0.5$ samples $\vvec{u} = (0.7, 0.2, 0.9, 0.4)$.

### The Dropout Output
Follow the dropout algorithm of the chapter. Compute the output in training mode and in inference mode. Show that the expected output of each neuron in training mode equals its input. What happens at inference if we do not scale with $\frac{1}{1-p}$ in training?

````{dropdown} Solution
The mask is $1_u = (\vvec{u}>p) = (1,0,1,0)$. In training mode, the output is
$$\frac{1}{1-p}1_u\circ\vvec{h}^{(\ell)} = 2\cdot(2,0,1,0) = (4,0,2,0).$$
In inference mode, the dropout layer returns $\vvec{h}^{(\ell)} = (2,0.5,1,3)$.

A neuron with input $x$ gives $0$ with probability $p$ and $\frac{x}{1-p}$ with probability $1-p$. Thus,
$$\mathbb{E}[\mathrm{dropout}(x)] = 0\cdot p+\frac{x}{1-p}(1-p) = x.$$
Without the scaling, the expected output in training is $(1-p)x$. The next layer learns its weights for inputs of this size. At inference, all neurons are active and the inputs of the next layer are $\frac{1}{1-p}$ times larger than in training, which is two times larger for $p=0.5$. The outputs of the network then change in a way that the training did not see.
````

## Convolution by Hand
In this exercise, we index all matrices from zero, as in the definition of the 2D convolution in the chapter:
$$O_{ij} = (X\star K)_{ij} = \sum_{i_k=0}^{k-1}\sum_{j_k=0}^{k-1}X_{i+i_k,j+j_k}K_{i_k,j_k}.$$

### A 2D Convolution
Compute $X\star K$ for
$$X = \begin{pmatrix}1 & 2 & 0 & 1\\0 & 1 & 3 & 1\\2 & 0 & 1 & 0\\1 & 1 & 0 & 2\end{pmatrix},\qquad K = \begin{pmatrix}1 & 0\\0 & -1\end{pmatrix}.$$

````{dropdown} Solution
The output has the shape $(4-2+1)\times(4-2+1) = 3\times 3$. Each entry is $O_{ij} = X_{i,j}-X_{i+1,j+1}$. For example, $O_{00} = X_{00}-X_{11} = 1-1 = 0$ and $O_{12} = X_{12}-X_{23} = 3-0 = 3$. All entries are
$$X\star K = \begin{pmatrix}0 & -1 & -1\\0 & 0 & 3\\1 & 0 & -1\end{pmatrix}.$$
````

### Output Sizes
An input has the size $32\times 32$ and the kernel has the size $5\times 5$. Compute the output size for (a) stride $1$ and padding $0$, (b) stride $1$ and padding $2$, (c) stride $2$ and padding $2$.

````{dropdown} Solution
We use $H_{out} = \left\lfloor\frac{H-k+2p}{s}\right\rfloor+1$:
1. $\left\lfloor\frac{32-5}{1}\right\rfloor+1 = 28$, thus $28\times 28$.
2. $\left\lfloor\frac{32-5+4}{1}\right\rfloor+1 = 32$, thus $32\times 32$. Padding $p=\frac{k-1}{2}$ with stride $1$ keeps the size.
3. $\left\lfloor\frac{31}{2}\right\rfloor+1 = 16$, thus $16\times 16$. Stride $2$ approximately halves the size.
````

### Convolution as a Matrix Multiplication
Consider
$$X = \begin{pmatrix}1 & 2 & 0\\0 & 1 & 3\\2 & 0 & 1\end{pmatrix},\qquad K = \begin{pmatrix}1 & 2\\-1 & 0\end{pmatrix}.$$
Compute $O = X\star K$. Then write the matrix $A$ such that $A\,\mathrm{vec}(X) = \mathrm{vec}(O)$, as in the chapter, and use $A$ to compute $\mathrm{vec}(O)$ again.

````{dropdown} Solution
$$O_{00} = 1\cdot1+2\cdot2-1\cdot0+0\cdot1 = 5,\quad O_{01} = 2+0-1+0 = 1,\quad O_{10} = 0+2-2+0 = 0,\quad O_{11} = 1+6-0+0 = 7.$$
Each row of $A$ holds the kernel entries at the positions of the corresponding image patch in $\mathrm{vec}(X) = (X_{00},X_{01},X_{02},X_{10},\ldots,X_{22})$:
$$A = \begin{pmatrix}
1 & 2 & 0 & -1 & 0 & 0 & 0 & 0 & 0\\
0 & 1 & 2 & 0 & -1 & 0 & 0 & 0 & 0\\
0 & 0 & 0 & 1 & 2 & 0 & -1 & 0 & 0\\
0 & 0 & 0 & 0 & 1 & 2 & 0 & -1 & 0
\end{pmatrix},\qquad A\,\mathrm{vec}(X) = A\begin{pmatrix}1\\2\\0\\0\\1\\3\\2\\0\\1\end{pmatrix} = \begin{pmatrix}5\\1\\0\\7\end{pmatrix}.$$
The convolution is a linear function of $X$. The matrix $A$ has many zeros and the same four kernel entries in every row. This is the parameter sharing of a convolutional layer.
````

### The Gradient Subject to the Kernel
Let $\mathcal{L}$ be a loss with $\frac{\partial\mathcal{L}}{\partial O_{ij}} = G_{ij}$. Use $\frac{\partial O_{ij}}{\partial K_{pq}} = X_{i+p,j+q}$ from the chapter to show that $\frac{\partial\mathcal{L}}{\partial K_{pq}} = (X\star G)_{pq}$. Compute this gradient for $X$ from the previous exercise and
$$G = \begin{pmatrix}1 & 0\\2 & -1\end{pmatrix}.$$

````{dropdown} Solution
With the chain rule, we sum over all output entries:
$$\frac{\partial\mathcal{L}}{\partial K_{pq}} = \sum_{i}\sum_{j}\frac{\partial\mathcal{L}}{\partial O_{ij}}\frac{\partial O_{ij}}{\partial K_{pq}} = \sum_{i}\sum_{j}G_{ij}X_{p+i,q+j} = (X\star G)_{pq}.$$
The last step is the definition of the convolution with the kernel $G$. Thus, the gradient subject to the kernel is a convolution of the input with the output gradient. For the given matrices, we get
$$X\star G = \begin{pmatrix}1+0+0-1 & 2+0+2-3\\0+0+4+0 & 1+0+0-1\end{pmatrix} = \begin{pmatrix}0 & 1\\4 & 0\end{pmatrix}.$$
````

## Pooling and Inductive Bias

### Max Pooling and Average Pooling
Apply max pooling and average pooling with a $2\times 2$ window and stride $2$ to
$$X = \begin{pmatrix}1 & 3 & 2 & 0\\4 & 2 & 1 & 1\\0 & 1 & 5 & 2\\2 & 2 & 3 & 4\end{pmatrix}.$$
How many parameters does a pooling layer have? For max pooling, which input entries get a nonzero gradient in backpropagation?

````{dropdown} Solution
The windows are the four $2\times 2$ blocks of $X$:
$$\text{max pooling: }\begin{pmatrix}4 & 2\\2 & 5\end{pmatrix},\qquad\text{average pooling: }\begin{pmatrix}2.5 & 1\\1.25 & 3.5\end{pmatrix}.$$
A pooling layer applies a fixed function. It has no parameters.

The partial derivative of $\max(x_1,\ldots,x_m)$ subject to $x_j$ is $1$ if $x_j$ is the maximum, and $0$ otherwise. Thus, max pooling sends the gradient of each output only to the input entry with the maximum value: $X_{10}=4$, $X_{02}=2$, $X_{22}=5$, and $X_{30}=2$ or $X_{31}=2$. The bottom left window has two maximum entries. Implementations select one of them, for example the first one. Average pooling sends $\frac14$ of the gradient to every entry of the window.
````

### Parameter Counts of a Convolutional Layer and a Dense Layer
A convolutional layer maps an RGB image of shape $(3,32,32)$ with $16$ kernels of size $3\times3$, stride $1$ and padding $0$ to an output tensor. Give the shape of the output tensor and the number of parameters. How many parameters does a dense layer need that maps the same input to an output of the same size? Which inductive biases explain the difference?

````{dropdown} Solution
The output has the shape $(16, 30, 30)$. Each kernel $K^{(s)}$ has the shape $(3,3,3)$, so $27$ weights, and there is one bias $b_s$ per kernel. Thus, the layer has $16\cdot(27+1) = 448$ parameters.

A dense layer from $3\cdot32\cdot32 = 3072$ inputs to $16\cdot30\cdot30 = 14400$ outputs has
$$3072\cdot14400+14400 = 44\,251\,200$$
parameters, which is approximately $10^5$ times more.

The difference comes from two inductive biases. **Locality**: each output depends only on a $3\times3$ patch, not on all pixels. **Parameter sharing**: the same kernel is used at all positions in the image. Parameter sharing also gives **translation equivariance**: if the input shifts, the feature map shifts in the same way.
````

## Practical Exercise: A Neural Network in numpy
You work for SpiralSense, a company that builds small sensor devices. A new device must sort measurements into three classes. The device can run numpy, but it cannot run PyTorch. Your team lead asks you to implement and train a neural network classifier from scratch.

In this exercise, you implement the forward pass, the backward pass and the training loop of an MLP. You use the formulas from the theory exercises above. The code uses the batch notation from the shapes exercise. In the code, `params` is a list of `[W, b]` pairs. The pair `params[l-1]` holds $W^{(l)}$ and $\vvec{b}^{(l)}$, and the last pair `params[-1]` holds the head $W$ and $\vvec{b}$. Each weight matrix has the shape of the book: `(d_out, d_in)`.

Run the setup code below. It generates the training data and the test data, and it gives helper functions.

```python
import numpy as np
import matplotlib.pyplot as plt

def make_spirals(n_per_class, noise, rng, turns=1.5):
    """Three interleaved spirals in the plane. Returns D (n x 2) and labels y in {0, 1, 2}."""
    D, y = [], []
    for l in range(3):
        r = np.linspace(0.05, 1.0, n_per_class)
        t = (np.linspace(0, turns * 2 * np.pi, n_per_class) + 2 * np.pi * l / 3
             + rng.normal(scale=noise, size=n_per_class))
        D.append(np.column_stack([r * np.sin(t), r * np.cos(t)]))
        y.append(np.full(n_per_class, l))
    return np.vstack(D), np.concatenate(y)

def one_hot(y, c):
    """The one-hot matrix Y with Y[i, y[i]] = 1."""
    Y = np.zeros((len(y), c))
    Y[np.arange(len(y)), y] = 1
    return Y

def init_params(layer_sizes, rng):
    """Random weights with the shape (d_out, d_in) and zero biases, for every layer."""
    params = []
    for d_in, d_out in zip(layer_sizes[:-1], layer_sizes[1:]):
        W = rng.normal(scale=np.sqrt(2 / d_in), size=(d_out, d_in))
        params.append([W, np.zeros(d_out)])
    return params

def plot_decision_boundary(D, y, params, activation="relu"):
    """Show the predicted class on a grid, together with the points."""
    grid = np.linspace(-1.2, 1.2, 300)
    xx, yy = np.meshgrid(grid, grid)
    P, _ = forward(np.column_stack([xx.ravel(), yy.ravel()]), params, activation)
    plt.contourf(xx, yy, P.argmax(axis=1).reshape(xx.shape), alpha=0.3, levels=[-0.5, 0.5, 1.5, 2.5])
    plt.scatter(D[:, 0], D[:, 1], c=y, s=8)
    plt.axis("scaled")
    plt.show()

D_train, y_train = make_spirals(200, noise=0.2, rng=np.random.default_rng(0))
D_test, y_test = make_spirals(100, noise=0.2, rng=np.random.default_rng(1))
Y_train, Y_test = one_hot(y_train, 3), one_hot(y_test, 3)

plt.scatter(D_train[:, 0], D_train[:, 1], c=y_train, s=8)
plt.axis("scaled")
plt.show()
```

**Question 1.** Look at the plot of the training data. Can a softmax regression on the two coordinates $(x_1,x_2)$ classify these points well? Why or why not?

````{dropdown} Solution
No. A softmax regression has linear decision boundaries in the space of $\vvec{x}$. Each spiral arm goes around the origin, so every straight line cuts through all three classes. A softmax regression that we train on these points reaches only approximately $45\%$ training accuracy. The network must learn a feature transformation $\vvec{h}^{(L)}$ in which the three classes are linearly separable.
````

**Task 1.** Implement the activation function `relu`, its derivative `relu_grad`, the row-wise `softmax` and the average `cross_entropy`. Subtract the maximum of each row in the softmax, as in the theory exercise. Use $ReLU'(0)=0$, as the chapter suggests.

```python
def relu(A):
    # Your implementation here
    return A

def relu_grad(A):
    # Your implementation here
    return np.ones_like(A)

def softmax(Z):
    """Apply the softmax to every row of Z."""
    # Your implementation here
    return Z

def cross_entropy(P, Y):
    """The average cross entropy of the probabilities P for the one-hot labels Y."""
    # Your implementation here
    return 0.0

ACTIVATIONS = {"relu": (relu, relu_grad)}

print(softmax(np.array([[0.0, 3.0, 1.0], [1000.0, 1001.0, 0.0]])))
print(cross_entropy(softmax(np.array([[0.0, 3.0, 1.0]])), np.array([[0, 0, 1]])))
```

````{dropdown} Solution
```python
def relu(A):
    return np.maximum(A, 0)

def relu_grad(A):
    return (A > 0).astype(float)

def softmax(Z):
    """Apply the softmax to every row of Z."""
    Z = Z - Z.max(axis=1, keepdims=True)  # shift every row, see the theory exercise
    E = np.exp(Z)
    return E / E.sum(axis=1, keepdims=True)

def cross_entropy(P, Y):
    """The average cross entropy of the probabilities P for the one-hot labels Y."""
    return -np.mean(np.log(np.sum(P * Y, axis=1)))

ACTIVATIONS = {"relu": (relu, relu_grad)}
```
The first row of the output is $(0.042, 0.844, 0.114)$, as in the theory exercise. The second row is $(0.269, 0.731, 0.000)$ and has no `nan`. The cross entropy is $2.170$.

The term `np.sum(P * Y, axis=1)` selects the probability of the true class in every row, because each row of `Y` has exactly one `1`.
````

**Task 2.** Implement `forward`. The function computes $A^{(\ell)}$ and $H^{(\ell)}$ for all hidden layers, and then $P=\softmax(H^{(L)}W^\top+\vvec{1}\vvec{b}^\top)$. Store the lists `H` $=[H^{(0)},\ldots,H^{(L)}]$ and `A` $=[A^{(1)},\ldots,A^{(L)}]$ and the matrix `P` in a dictionary `cache`. The backward pass needs these values.

```python
def forward(D, params, activation="relu"):
    """Return the probabilities P and the cache with the lists H and A and the matrix P."""
    phi, _ = ACTIVATIONS[activation]
    # Your implementation here
    P = np.full((D.shape[0], params[-1][0].shape[0]), 1 / params[-1][0].shape[0])
    cache = {"H": [D], "A": [], "P": P}
    return P, cache

params = init_params([2, 32, 32, 3], np.random.default_rng(42))
P, cache = forward(D_train, params)
print("shapes:", [H.shape for H in cache["H"]], P.shape)
print("initial loss:", cross_entropy(P, Y_train), " log(3):", np.log(3))
```

````{dropdown} Solution
```python
def forward(D, params, activation="relu"):
    """Return the probabilities P and the cache with the lists H and A and the matrix P."""
    phi, _ = ACTIVATIONS[activation]
    H, A = [D], []
    for W, b in params[:-1]:          # the hidden layers
        A.append(H[-1] @ W.T + b)
        H.append(phi(A[-1]))
    W, b = params[-1]                 # the softmax regression head
    P = softmax(H[-1] @ W.T + b)
    return P, {"H": H, "A": A, "P": P}
```
The shapes are `(600, 2)`, `(600, 32)`, `(600, 32)` and `(600, 3)`. The initial loss is approximately $1.19$.
````

**Question 2.** The initial loss is close to $\log 3\approx 1.10$. Why do you expect this value for a network with random weights? What would you conclude from an initial loss of $20$?

````{dropdown} Solution
The random weights do not depend on the labels, and they give logits of a moderate size. Then the softmax gives approximately the uniform distribution $(\frac13,\frac13,\frac13)$, and the cross entropy of every point is approximately $-\log\frac13 = \log 3$. An initial loss of $20$ means that the network gives a probability of approximately $\exp(-20)$ to the true classes. This shows an error in the implementation or an initialization with much too large weights.
````

**Question 3.** Assume that you initialize all weights with zero instead of random values. What is the gradient of the hidden layers? Can gradient descent train this network?

````{dropdown} Solution
With zero weights and zero biases, all $A^{(\ell)}$ are zero, so all ReLU outputs and all $ReLU'$ values are zero. Thus, the gradients of all hidden layers are zero, and gradient descent changes only the bias of the head. The network cannot learn.

The problem also occurs with other activations if all weights of a layer are equal. Then all neurons of the layer compute the same function and get the same gradient. They stay equal after each update, so the layer acts as one neuron. Random initialization breaks this symmetry.
````

**Task 3.** Implement `backward` with the formulas from the batch exercise. Start with $G=\frac1n(P-Y)$. Go through the layers from the head back to the first hidden layer. Return a list `grads` that has the same structure as `params`.

```python
def backward(Y, params, cache, activation="relu"):
    """Return the gradients [[grad_W, grad_b], ...] of the average cross entropy."""
    _, phi_grad = ACTIVATIONS[activation]
    H, A, P = cache["H"], cache["A"], cache["P"]
    # Your implementation here
    grads = [[np.zeros_like(W), np.zeros_like(b)] for W, b in params]
    return grads
```

````{dropdown} Solution
```python
def backward(Y, params, cache, activation="relu"):
    """Return the gradients [[grad_W, grad_b], ...] of the average cross entropy."""
    _, phi_grad = ACTIVATIONS[activation]
    H, A, P = cache["H"], cache["A"], cache["P"]
    G = (P - Y) / Y.shape[0]                       # the gradient subject to the logits Z
    grads = [None] * len(params)
    for l in reversed(range(len(params))):
        W = params[l][0]
        grads[l] = [G.T @ H[l], G.sum(axis=0)]     # G^T H and G^T 1
        if l > 0:
            G = (G @ W) * phi_grad(A[l - 1])       # go back through W and the activation
    return grads
```
The Python index `l` starts at zero: `params[l]` maps `H[l]` to `A[l]` (for the head, to the logits $Z$). The activation that produced `H[l]` has the input `A[l - 1]`. Thus, the loop uses `A[l - 1]` to go back from `H[l]` to the layer before it.
````

**Task 4.** Make sure that your backward pass is correct before you train with it. Implement `numerical_partial`. The function approximates one partial derivative with the central difference
$$\frac{\partial\mathcal{L}}{\partial\theta_k}\approx\frac{\mathcal{L}(\theta+\epsilon\vvec{e}_k)-\mathcal{L}(\theta-\epsilon\vvec{e}_k)}{2\epsilon}.$$
Then run the cell below. The relative error must be smaller than $10^{-6}$.

```python
def numerical_partial(D, Y, params, l, j, idx, eps=1e-5):
    """Approximate the partial derivative of the loss subject to params[l][j][idx]."""
    # Your implementation here
    return 0.0

params = init_params([2, 32, 32, 3], np.random.default_rng(42))
P, cache = forward(D_train, params)
grads = backward(Y_train, params, cache)
rng = np.random.default_rng(5)
worst = 0.0
for l in range(len(params)):
    for j in range(2):
        for _ in range(5):
            idx = tuple(rng.integers(0, s) for s in params[l][j].shape)
            num = numerical_partial(D_train, Y_train, params, l, j, idx)
            ana = grads[l][j][idx]
            worst = max(worst, abs(num - ana) / max(abs(num) + abs(ana), 1e-12))
print("largest relative error:", worst)
```

````{dropdown} Solution
```python
def numerical_partial(D, Y, params, l, j, idx, eps=1e-5):
    """Approximate the partial derivative of the loss subject to params[l][j][idx]."""
    theta = params[l][j]
    old = theta[idx]
    theta[idx] = old + eps
    loss_plus = cross_entropy(forward(D, params)[0], Y)
    theta[idx] = old - eps
    loss_minus = cross_entropy(forward(D, params)[0], Y)
    theta[idx] = old                     # restore the parameter
    return (loss_plus - loss_minus) / (2 * eps)
```
The largest relative error is approximately $3\cdot10^{-8}$. An error in the backward pass, for example a missing transpose or a missing factor $\frac1n$, gives a relative error close to $1$.

The numerical gradient needs two forward passes per parameter. For $1251$ parameters, this is $2502$ forward passes for one gradient. Backpropagation computes all partial derivatives with one forward pass and one backward pass. Thus, we use the numerical gradient only to test the implementation.
````

**Task 5.** Implement `train_gd`. The function does `n_steps` steps of gradient descent with step size `eta` and records the loss before each step. Train the network $2{,}000$ steps with each step size in `etas`. Show the loss curves. For each step size, print the final loss and the training and test accuracy.

```python
def accuracy(D, y, params, activation="relu"):
    P, _ = forward(D, params, activation)
    return np.mean(P.argmax(axis=1) == y)

def train_gd(D, Y, params, eta, n_steps, activation="relu"):
    """Train params in place with gradient descent. Return the list of losses."""
    losses = []
    # Your implementation here
    return losses

etas = [0.03, 0.3, 1.0, 3.0]
# Your implementation here: for every eta, initialize the parameters with
# init_params([2, 32, 32, 3], np.random.default_rng(42)), train them, plot the
# losses, and print the final loss and the accuracies.
```

````{dropdown} Solution
```python
def train_gd(D, Y, params, eta, n_steps, activation="relu"):
    """Train params in place with gradient descent. Return the list of losses."""
    losses = []
    for _ in range(n_steps):
        P, cache = forward(D, params, activation)
        losses.append(cross_entropy(P, Y))
        grads = backward(Y, params, cache, activation)
        for (W, b), (grad_W, grad_b) in zip(params, grads):
            W -= eta * grad_W
            b -= eta * grad_b
    return losses

results = {}
for eta in etas:
    params = init_params([2, 32, 32, 3], np.random.default_rng(42))
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        losses = train_gd(D_train, Y_train, params, eta, 2000)
    results[eta] = params
    plt.plot(losses, label=f"eta = {eta}")
    print(f"eta {eta}: loss {losses[-1]:.4f}, train acc {accuracy(D_train, y_train, params):.3f}, "
          f"test acc {accuracy(D_test, y_test, params):.3f}")
plt.xlabel("step")
plt.ylabel("training loss")
plt.yscale("log")
plt.legend()
plt.show()

plot_decision_boundary(D_test, y_test, results[1.0])
```
The output is approximately:

| $\eta$ | final loss | train accuracy | test accuracy |
|---|---|---|---|
| $0.03$ | $1.03$ | $0.50$ | $0.51$ |
| $0.3$ | $0.02$ | $1.00$ | $1.00$ |
| $1.0$ | $0.002$ | $1.00$ | $1.00$ |
| $3.0$ | $0.87$ | $0.54$ | $0.54$ |

The update `W -= eta * grad_W` changes the array in the list `params`. Thus, the function trains the parameters in place.
````

**Question 4.** Describe the four loss curves. Which step size would you choose, and why? Compare with the step size problems of gradient descent for regression.

````{dropdown} Solution
- $\eta=0.03$ is too small. The loss decreases very slowly, from $1.19$ to approximately $1.03$. After $2{,}000$ steps, the network is still underfitting.
- $\eta=0.3$ decreases slowly for approximately $500$ steps (the loss is $0.92$ at step $500$). Then the loss decreases faster and reaches approximately $0.02$.
- $\eta=1.0$ decreases fastest (the loss is $0.13$ at step $500$) and reaches the lowest loss. The curve has some short spikes, but the loss comes back down after each spike.
- $\eta=3.0$ is too large. The loss decreases to approximately $0.8$, but then it jumps up many times, to values above $5$. The training does not reach a low loss.

We choose $\eta=1.0$, or a value between $0.3$ and $1.0$ for more safety. The spikes show that $\eta=1.0$ is close to the limit of stable training. As in regression, a too small step size makes gradient descent slow, and a too large step size makes it unstable. The difference is that the loss of a neural network is not convex. It has flat regions and steep regions, and we find a good step size only by experiments. The decision boundary of the trained network follows the three spiral arms.
````

**Task 6.** Implement `train_sgd`. In every epoch, the function shuffles the training points with `rng.permutation`, divides them into batches of `batch_size` points and does one update per batch. After each epoch, the function records the loss on the full training set. Train with $\eta=0.3$ for $100$ epochs, once with `batch_size=600` (this is gradient descent) and once with `batch_size=32`. Show both loss curves per epoch.

```python
def train_sgd(D, Y, params, eta, n_epochs, batch_size, rng, activation="relu"):
    """Train params in place with SGD. Return the training loss after each epoch."""
    losses = []
    # Your implementation here
    return losses

# Your implementation here: train two networks that start from
# init_params([2, 32, 32, 3], np.random.default_rng(42)), use
# rng=np.random.default_rng(7) for each, and plot both loss curves.
```

````{dropdown} Solution
```python
def train_sgd(D, Y, params, eta, n_epochs, batch_size, rng, activation="relu"):
    """Train params in place with SGD. Return the training loss after each epoch."""
    losses = []
    for _ in range(n_epochs):
        order = rng.permutation(D.shape[0])
        for start in range(0, D.shape[0], batch_size):
            batch = order[start:start + batch_size]
            P, cache = forward(D[batch], params, activation)
            grads = backward(Y[batch], params, cache, activation)
            for (W, b), (grad_W, grad_b) in zip(params, grads):
                W -= eta * grad_W
                b -= eta * grad_b
        losses.append(cross_entropy(forward(D, params, activation)[0], Y))
    return losses

for batch_size in [600, 32]:
    params = init_params([2, 32, 32, 3], np.random.default_rng(42))
    losses = train_sgd(D_train, Y_train, params, 0.3, 100, batch_size, np.random.default_rng(7))
    plt.plot(losses, label=f"batch size {batch_size}")
    print(f"batch size {batch_size}: loss {losses[-1]:.3f}, train acc {accuracy(D_train, y_train, params):.3f}, "
          f"test acc {accuracy(D_test, y_test, params):.3f}")
plt.xlabel("epoch")
plt.ylabel("training loss")
plt.legend()
plt.show()
```
After $100$ epochs, gradient descent has a loss of approximately $1.04$ and an accuracy of approximately $0.50$. SGD with batch size $32$ has a loss of approximately $0.03$ and a test accuracy of approximately $0.99$.
````

**Question 5.** Why is SGD much better than gradient descent after the same number of epochs? Why is the SGD loss curve less smooth?

````{dropdown} Solution
One epoch costs the same for both methods, because both use every point once. But gradient descent does $1$ update per epoch, and SGD with batch size $32$ does $19$ updates per epoch. After $100$ epochs, gradient descent did $100$ updates and is still in the slow first phase of Task 5. SGD did $1{,}900$ updates.

Each SGD update uses the gradient of a small random batch. This gradient is only an estimate of the full gradient (see the theory exercise). Thus, some updates increase the loss on the full training set, and the curve is noisy. This noise can also help SGD to leave plateaus and small local minima.
````

**Task 7.** Study the vanishing gradient. Implement `sigmoid` and `sigmoid_grad` and add them to `ACTIVATIONS`. Initialize a deep network with six hidden layers of $32$ neurons: `init_params([2] + [32] * 6 + [3], np.random.default_rng(42))`. For both activations, compute the norm $\lVert\nabla_{W^{(\ell)}}\mathcal{L}\rVert$ of the weight gradient of every layer at the initialization. Show the norms per layer on a logarithmic scale. Then train both networks with `train_gd` for $2{,}000$ steps with $\eta=0.3$ and print the final loss.

```python
def sigmoid(A):
    # Your implementation here
    return A

def sigmoid_grad(A):
    # Your implementation here
    return np.ones_like(A)

ACTIVATIONS["sigmoid"] = (sigmoid, sigmoid_grad)

# Your implementation here
```

````{dropdown} Solution
```python
def sigmoid(A):
    return 1 / (1 + np.exp(-A))

def sigmoid_grad(A):
    S = sigmoid(A)
    return S * (1 - S)

ACTIVATIONS["sigmoid"] = (sigmoid, sigmoid_grad)

layer_sizes = [2] + [32] * 6 + [3]
for activation in ["sigmoid", "relu"]:
    params = init_params(layer_sizes, np.random.default_rng(42))
    P, cache = forward(D_train, params, activation)
    grads = backward(Y_train, params, cache, activation)
    norms = [np.linalg.norm(grad_W) for grad_W, _ in grads]
    plt.semilogy(range(1, len(norms) + 1), norms, marker="o", label=activation)
    losses = train_gd(D_train, Y_train, params, 0.3, 2000, activation)
    print(f"{activation}: gradient norms {np.round(norms, 5)}, final loss {losses[-1]:.3f}")
plt.xlabel("layer (7 = head)")
plt.ylabel("norm of the weight gradient")
plt.legend()
plt.show()
```
The gradient norms at the initialization are approximately:

| layer | 1 | 2 | 3 | 4 | 5 | 6 | head |
|---|---|---|---|---|---|---|---|
| sigmoid | $5\cdot10^{-5}$ | $3\cdot10^{-3}$ | $1\cdot10^{-2}$ | $3\cdot10^{-2}$ | $0.1$ | $0.4$ | $1.3$ |
| ReLU | $0.1$ | $0.3$ | $0.4$ | $0.3$ | $0.4$ | $0.4$ | $0.4$ |

After training, the ReLU network has a loss close to $0$. The sigmoid network stays at a loss of approximately $\log 3\approx 1.10$.
````

**Question 6.** Explain the gradient norms with the theory exercise on the vanishing gradient. Why does the sigmoid network not learn?

````{dropdown} Solution
With sigmoid activations, the gradient of layer $\ell$ contains one factor $\sigma'\leq\frac14$ for each later hidden layer. Thus, the gradient norm decreases by a large factor from each layer to the layer before it. The first layer gets a gradient that is approximately $25{,}000$ times smaller than the gradient of the head. With $\eta=0.3$, the first layers practically do not change. The head then sees almost random features, and the best that it can do is to predict approximately uniform probabilities.

With ReLU, the derivative of an active neuron is $1$, so the gradient norms are approximately equal for all layers. All layers learn at the same time.

The initialization of `init_params` is designed for ReLU. Another initialization for sigmoid networks can make the problem smaller, but the factor $\frac14$ per layer stays.
````

**Task 8.** Prevent overfitting with weight decay and dropout. The cell below generates a small training set with more noise and a large test set. Implement `forward_dropout`: it is `forward` with a dropout layer after every hidden activation in training mode. Store the scaled masks $\frac{1}{1-p}1_u$ in `cache["M"]`. Implement `backward_dropout`: it multiplies $GW$ with the mask before the activation derivative. Then implement `train_regularized` with gradient descent, weight decay $\lambda$ and dropout probability $p$. Train a network with two hidden layers of $128$ neurons for $5{,}000$ steps with $\eta=0.3$ for each configuration in `configs`. Print the training and test accuracy.

```python
D_small, y_small = make_spirals(25, noise=0.5, rng=np.random.default_rng(3))
D_large, y_large = make_spirals(500, noise=0.5, rng=np.random.default_rng(4))
Y_small = one_hot(y_small, 3)

def forward_dropout(D, params, p_drop, rng, activation="relu"):
    """forward with dropout after every hidden layer. The cache also holds the masks M."""
    # Your implementation here
    return forward(D, params, activation)

def backward_dropout(Y, params, cache, activation="relu"):
    """backward for forward_dropout."""
    # Your implementation here
    return backward(Y, params, cache, activation)

def train_regularized(D, Y, params, eta, n_steps, lam, p_drop, rng, activation="relu"):
    """Gradient descent on the loss plus lam * ||theta||^2, with dropout probability p_drop."""
    # Your implementation here
    pass

configs = [(0.0, 0.0), (1e-3, 0.0), (1e-2, 0.0), (0.0, 0.3), (0.0, 0.5)]   # (lam, p_drop)
# Your implementation here: for every configuration, initialize
# init_params([2, 128, 128, 3], np.random.default_rng(42)), train with
# rng=np.random.default_rng(42), and print the accuracy on D_small and D_large.
```

````{dropdown} Solution
```python
def forward_dropout(D, params, p_drop, rng, activation="relu"):
    """forward with dropout after every hidden layer. The cache also holds the masks M."""
    phi, _ = ACTIVATIONS[activation]
    H, A, M = [D], [], []
    for W, b in params[:-1]:
        A.append(H[-1] @ W.T + b)
        mask = (rng.uniform(size=A[-1].shape) > p_drop) / (1 - p_drop)   # scaled mask
        M.append(mask)
        H.append(mask * phi(A[-1]))
    W, b = params[-1]
    P = softmax(H[-1] @ W.T + b)
    return P, {"H": H, "A": A, "M": M, "P": P}

def backward_dropout(Y, params, cache, activation="relu"):
    """backward for forward_dropout."""
    _, phi_grad = ACTIVATIONS[activation]
    H, A, M, P = cache["H"], cache["A"], cache["M"], cache["P"]
    G = (P - Y) / Y.shape[0]
    grads = [None] * len(params)
    for l in reversed(range(len(params))):
        W = params[l][0]
        grads[l] = [G.T @ H[l], G.sum(axis=0)]
        if l > 0:
            G = (G @ W) * M[l - 1] * phi_grad(A[l - 1])
    return grads

def train_regularized(D, Y, params, eta, n_steps, lam, p_drop, rng, activation="relu"):
    """Gradient descent on the loss plus lam * ||theta||^2, with dropout probability p_drop."""
    for _ in range(n_steps):
        P, cache = forward_dropout(D, params, p_drop, rng, activation)
        grads = backward_dropout(Y, params, cache, activation)
        for (W, b), (grad_W, grad_b) in zip(params, grads):
            W -= eta * (grad_W + 2 * lam * W)
            b -= eta * (grad_b + 2 * lam * b)

for lam, p_drop in configs:
    params = init_params([2, 128, 128, 3], np.random.default_rng(42))
    train_regularized(D_small, Y_small, params, 0.3, 5000, lam, p_drop, np.random.default_rng(42))
    print(f"lambda {lam}, p {p_drop}: train acc {accuracy(D_small, y_small, params):.3f}, "
          f"test acc {accuracy(D_large, y_large, params):.3f}")
```
For the evaluation, `accuracy` calls `forward`, which has no dropout. This is the inference mode of the dropout layer. With $p=0$, the mask is $1$ everywhere and the functions give the same result as `forward` and `backward`.

The output is approximately:

| $\lambda$ | $p$ | train accuracy | test accuracy |
|---|---|---|---|
| $0$ | $0$ | $1.00$ | $0.68$ |
| $0.001$ | $0$ | $0.97$ | $0.73$ |
| $0.01$ | $0$ | $0.33$ | $0.33$ |
| $0$ | $0.3$ | $1.00$ | $0.71$ |
| $0$ | $0.5$ | $0.93$ | $0.72$ |
````

**Question 7.** Explain the results of Task 8 with the bias-variance tradeoff. Why does $\lambda=0.01$ fail? Do you expect the same numbers if you change the seeds?

````{dropdown} Solution
The network has approximately $17{,}000$ parameters and only $75$ training points. Without regularization, it classifies all training points correctly, also the noisy points. It has a low bias and a high variance, and the test accuracy is only $0.68$.

A small weight decay ($\lambda=0.001$) and dropout reduce the variance. The training accuracy decreases a little, and the test accuracy increases by approximately $3$ to $5$ percentage points.

With $\lambda=0.01$, the penalty dominates the loss. As in the theory exercise, the weights go to approximately zero, the network predicts approximately uniform probabilities, and it predicts one class for all points. Its accuracy is $\frac13$. This network is underfitting: it has a high bias.

With other seeds, the numbers change by a few percentage points, because the training set is very small. To compare regularization methods reliably, repeat the experiment with several seeds, or use cross-validation to select $\lambda$ and $p$.
````

## Practical Exercise: Edge Detection with Convolution and Pooling
Your team at SpiralSense now wants to add a small camera to the device. Before you train a convolutional network, you implement the two basic operations in numpy and look at what they do.

**Task 1.** Implement `conv2d` with stride and zero padding, and `pool2d` for max pooling and average pooling. Use the output size formula of the chapter. Then run the cell. It applies the horizontal edge kernel of the chapter to an image with a white square.

```python
def conv2d(X, K, stride=1, padding=0):
    """The 2D convolution X * K of the chapter, with stride and zero padding."""
    # Your implementation here
    return X

def pool2d(X, k, stride, op=np.max):
    """Apply op (np.max or np.mean) to every k x k window."""
    # Your implementation here
    return X

X = np.zeros((8, 8))
X[2:6, 2:6] = 1.0                                   # a white square on a black background
K = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1.0]])  # the horizontal edge kernel

O = conv2d(X, K)
print(O)
print(conv2d(X, K, padding=1).shape, conv2d(X, K, stride=2).shape, conv2d(X, K, stride=2, padding=1).shape)
print(pool2d(np.maximum(O, 0), 2, 2))

fig, axes = plt.subplots(1, 3, figsize=(9, 3))
for ax, image, title in zip(axes, [X, O, conv2d(X, K.T)], ["input", "kernel K", "kernel K^T"]):
    ax.imshow(image, cmap="gray")
    ax.set_title(title)
plt.show()
```

````{dropdown} Solution
```python
def conv2d(X, K, stride=1, padding=0):
    """The 2D convolution X * K of the chapter, with stride and zero padding."""
    X = np.pad(X, padding)
    k = K.shape[0]
    H_out = (X.shape[0] - k) // stride + 1
    W_out = (X.shape[1] - k) // stride + 1
    O = np.zeros((H_out, W_out))
    for i in range(H_out):
        for j in range(W_out):
            patch = X[i * stride:i * stride + k, j * stride:j * stride + k]
            O[i, j] = np.sum(patch * K)            # inner product of the patch and the kernel
    return O

def pool2d(X, k, stride, op=np.max):
    """Apply op (np.max or np.mean) to every k x k window."""
    H_out = (X.shape[0] - k) // stride + 1
    W_out = (X.shape[1] - k) // stride + 1
    O = np.zeros((H_out, W_out))
    for i in range(H_out):
        for j in range(W_out):
            O[i, j] = op(X[i * stride:i * stride + k, j * stride:j * stride + k])
    return O
```
The output of the convolution is
$$O = \begin{pmatrix}
1 & 2 & 3 & 3 & 2 & 1\\
1 & 2 & 3 & 3 & 2 & 1\\
0 & 0 & 0 & 0 & 0 & 0\\
0 & 0 & 0 & 0 & 0 & 0\\
-1 & -2 & -3 & -3 & -2 & -1\\
-1 & -2 & -3 & -3 & -2 & -1
\end{pmatrix}.$$
The shapes are `(8, 8)`, `(3, 3)` and `(4, 4)`, as the output size formula predicts. The max pooling of the ReLU output is
$$\begin{pmatrix}2 & 3 & 2\\0 & 0 & 0\\0 & 0 & 0\end{pmatrix}.$$
````

**Question 1.** Explain the output $O$. Where is it positive, where is it negative, and where is it zero? What does the transposed kernel $K^\top$ detect? What information does the max pooling keep, and what information does it lose?

````{dropdown} Solution
The kernel subtracts the row above the center from the row below the center. At the top edge of the square, the image is dark above and light below. Thus, the output is positive there (a light line). At the bottom edge, the image is light above and dark below, so the output is negative (a dark line). Inside the square and in the background, the image is constant and the output is zero. The values at the corners are smaller, because only a part of the kernel covers the square.

The transposed kernel $K^\top$ subtracts the left column from the right column, so it detects vertical edges.

After the ReLU, the feature map shows only the top edge. The max pooling keeps the information that there is a top edge in the upper part of the image, and its strength. It loses the exact position of the edge inside each $2\times2$ window. Small shifts of the input do not change the pooled output much.
````

**Task 2.** The chapter shows that a convolution is a matrix multiplication $A\,\mathrm{vec}(X) = \mathrm{vec}(O)$. Implement `conv_matrix`, which builds $A$ for a kernel $K$ and an input of size $H\times W$ (stride $1$, no padding). Make sure that `A @ X.ravel()` equals `conv2d(X, K).ravel()`. Then make sure with a numerical gradient that the gradient of $\mathcal{L}(K) = \sum_{ij}G_{ij}(X\star K)_{ij}$ subject to $K$ is $X\star G$.

```python
def conv_matrix(K, H, W):
    """The matrix A with A @ X.ravel() == conv2d(X, K).ravel() for X of shape (H, W)."""
    # Your implementation here
    return np.zeros(((H - K.shape[0] + 1) * (W - K.shape[0] + 1), H * W))

A = conv_matrix(K, 8, 8)
print(A.shape, np.allclose(A @ X.ravel(), conv2d(X, K).ravel()))

# Your implementation here: draw random X (6 x 6), K (3 x 3) and G (4 x 4) with
# np.random.default_rng(0), compute the numerical gradient of L(K) for every
# kernel entry, and compare it with conv2d(X, G).
```

````{dropdown} Solution
```python
def conv_matrix(K, H, W):
    """The matrix A with A @ X.ravel() == conv2d(X, K).ravel() for X of shape (H, W)."""
    k = K.shape[0]
    H_out, W_out = H - k + 1, W - k + 1
    A = np.zeros((H_out * W_out, H * W))
    for i in range(H_out):
        for j in range(W_out):
            for p in range(k):
                for q in range(k):
                    A[i * W_out + j, (i + p) * W + (j + q)] = K[p, q]
    return A

rng = np.random.default_rng(0)
X_r, K_r, G = rng.normal(size=(6, 6)), rng.normal(size=(3, 3)), rng.normal(size=(4, 4))
loss = lambda K: np.sum(G * conv2d(X_r, K))
eps = 1e-6
numerical = np.zeros((3, 3))
for p in range(3):
    for q in range(3):
        E = np.zeros((3, 3))
        E[p, q] = eps
        numerical[p, q] = (loss(K_r + E) - loss(K_r - E)) / (2 * eps)
print("largest difference:", np.abs(numerical - conv2d(X_r, G)).max())
```
The matrix $A$ has the shape `(36, 64)`, and the products are equal. The largest difference of the gradients is approximately $10^{-9}$.

Row `i * W_out + j` of $A$ computes $O_{ij}$. It holds $K_{pq}$ at the column of $X_{i+p,j+q}$ in `X.ravel()`. Each row has only $9$ nonzero entries of $64$, and all rows hold the same $9$ values. A convolutional layer is a dense layer with this restricted weight matrix. This restriction is the inductive bias of the chapter.
````

## Universal Approximation: True or False
Select all correct statements.

Group of answer choices
- The universal approximation theorem shows that gradient descent finds a network that approximates any continuous function on a compact set.
- An MLP with one hidden layer and ReLU activation can approximate any continuous function on a compact set to any accuracy, if the hidden layer has enough neurons.
- The logits of an MLP with ReLU activations are a piecewise affine function of the input.
- An MLP with ten hidden layers and no activation functions can approximate any continuous function.

````{dropdown} Solution
**False: The universal approximation theorem shows that gradient descent finds a network that approximates any continuous function on a compact set.** The theorem is non-constructive. It states that such a network exists, but it does not say how to find its weights.

**True: An MLP with one hidden layer and ReLU activation can approximate any continuous function on a compact set to any accuracy, if the hidden layer has enough neurons.** This is the statement of the theorem in the chapter.

**True: The logits of an MLP with ReLU activations are a piecewise affine function of the input.** Each neuron is affine where it is active and zero where it is inactive. Compositions and sums of piecewise affine functions are piecewise affine. The softmax after the logits is not affine.

**False: An MLP with ten hidden layers and no activation functions can approximate any continuous function.** Without activations, all layers together are one affine function (see the theory exercise). An affine function cannot approximate, for example, $f(x)=x^2$ on $[-1,1]$.
````

## Activations and Training: True or False
Select all correct statements.

Group of answer choices
- The derivative of the sigmoid function is at most $\frac14$, so deep sigmoid networks can have vanishing gradients.
- A ReLU neuron whose input is negative for all training points does not change during gradient descent.
- SGD with a constant step size converges to a minimizer of the training loss.
- The gradient of the cross entropy subject to the logits is the vector of predicted probabilities minus the one-hot vector of the true class.
- In inference mode, a dropout layer sets a random subset of the neurons to zero.
- Weight decay multiplies the weights with a factor smaller than one in each gradient descent step.

````{dropdown} Solution
**True: The derivative of the sigmoid function is at most $\frac14$, so deep sigmoid networks can have vanishing gradients.** The gradient of an early layer contains one factor $\sigma'\leq\frac14$ per later layer.

**True: A ReLU neuron whose input is negative for all training points does not change during gradient descent.** Its derivative $ReLU'$ is zero for all points, so the weights and the bias of the neuron get a zero gradient. This is the dead neuron effect.

**False: SGD with a constant step size converges to a minimizer of the training loss.** The batch gradients are not zero at the minimizer, so SGD continues to move. SGD needs a decreasing step size to converge.

**True: The gradient of the cross entropy subject to the logits is the vector of predicted probabilities minus the one-hot vector of the true class.** This is $f(\vvec{x})-\vvec{e}_y$ from the chapter.

**False: In inference mode, a dropout layer sets a random subset of the neurons to zero.** In inference mode, dropout returns its input unchanged. It drops neurons only in training mode.

**True: Weight decay multiplies the weights with a factor smaller than one in each gradient descent step.** The update is $\theta\gets(1-2\eta\lambda)\theta-\eta\nabla\mathcal{L}(\mathcal{D},\theta)$.
````

## Convolutional Layers: Fill in the Blanks
An input tensor has the shape $(3,64,64)$. A convolutional layer has $8$ kernels of size $5\times5$, stride $1$ and padding $2$. A max pooling layer with a $2\times2$ window and stride $2$ follows it.

- The output tensor of the convolutional layer has the shape ___.
- The convolutional layer has ___ parameters.
- The output tensor of the pooling layer has the shape ___.
- The pooling layer has ___ parameters.

````{dropdown} Solution
- The output size is $\left\lfloor\frac{64-5+2\cdot2}{1}\right\rfloor+1 = 64$, and there is one output channel per kernel. Thus, the shape is $(8,64,64)$.
- Each kernel has the shape $(3,5,5)$, so $75$ weights and one bias. Thus, the layer has $8\cdot(75+1) = 608$ parameters.
- The pooling halves the height and the width and keeps the channels: $\left\lfloor\frac{64-2}{2}\right\rfloor+1 = 32$. Thus, the shape is $(8,32,32)$.
- A pooling layer has $0$ parameters.
````
