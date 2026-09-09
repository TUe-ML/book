# Exercises

## Vanishing Gradient
In this exercise you will take the two-moons PyTorch example from the lecture, adapt the network into a deeper version and analyze the vanishing gradient effect. Specifically, compare the sigmoid and ReLU versions of the same network structure recording and visualizing gradients to determine magnitude of first layer updates.

1. Take the training of a neural network in Pytorch from [Backpropagation](https://sibylse.github.io/TUEML/neuralnets_backprop.html) topic. Execute it locally and verify the results are as expected.

2. Replace the original network by a deeper network with at least 3 hidden layers. Create two versions, one with ReLU activation, one with Sigmoid, train a few times observing the loss and accuracy results. Is one architecture always performing better than the other? Why?
```{tip}
:class: dropdown
Extend the class by adding more `nn.Linear(...)` hidden layers of the same width, add them in the forward pass and latent computation methods. Create one version with `torch.sigmoid(...)` and one with `torch.relu(...)`
```
3. During training record the norm of the weight gradient in each hidden layer. Plot the gradients comparing the architectures and explain whether the sigmoid network shows vanishing gradient effect more clearly than the ReLU network.
```{tip}
:class: dropdown
The gradients are computed with `loss.backward()` step in the training, so they can be stored after it is performed. To extract the gradient norm at a e.g. `hidden1` layer you can use `hidden1grad = model.hidden1.weight.grad.norm().item()`.
```

## Backpropagation Implementation
Design a simple model on paper consisting of linear and ReLU layers, manually calculating and implementing forward and backward passes. Afterward, the training is also implemented - single descent step and visualization of progress. For this exercise you may use PyTorch tensors and tensor operations, but **not** `nn.Linear()`, `torch.optim`, or automated gradient computation.

1. Using the course notation, for the network: $$x \to Linear \to ReLU \to Linear \to SoftMax \to CE$$ write out the forward pass and calculate the backward pass using backpropagation.

2. Implement the forward pass function taking as input the weight matrices and bias vectors and outputing the logits and values of neurons at intermediate layers (these will be needed for the backward pass). Do so using matrix multiplication and the `torch.softmax()`.

3. Implement the backward pass without the use of automatic differentiation like `loss.backward()`.
```{tip}
:class: dropdown
For the vectors $e_y$, you can use `F.one_hot(y, num_classes=a2.shape[1]).float()` from `torch.nn.functional as F`)
```

4. Take again the two moons dataset with the model visualization. Implement one gradient descent training step, with a configurable learning rate. Perform the training multiple times visualizing the progress in between - save the loss for the loss curve, and reuse the decision boundary visualization from lecture example.
