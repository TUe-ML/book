#!/usr/bin/env python
# coding: utf-8

# ## SGD
# 
# We have now discussed how to derive the gradient of the cross-entropy loss, that we can use to optimize the weights of a neural network by gradient descent. The only problem is that neural networks typically require (comparably) vast amounts of data to learn a suitable feature transformation that is able to discriminate between the classes. The computation of the gradient, which requires to add the gradients of the cross-entropy for each data point is typically too costly if the dataset is big.
# 
# **Stochastic Gradient Descent (SGD)** offers here a solution by updating the weights using only a small batch of training examples at a time. That is, in every update step, SGD performs a gradient descent step on the loss function
# $$\mathcal{L}(\mathcal{B},\theta)=\frac{1}{\lvert\mathcal{B}\rvert} \sum_{(\vvec{x},y)\in\mathcal{B}} CE(y,f_\theta(\vvec{x})),$$
# where $\mathcal{B}\subseteq \mathcal{D}$ is a **batch**. Typically, the batches are chosen as disjunctive subsets of the dataset: $\mathcal{B}_1\cup\ldots\cup\mathcal{B}_m=\mathcal{D}$ and $\mathcal{B}_b\cap \mathcal{B}_j =\emptyset$. Performing the updates on all batches of this disjunctive decomposition is then called an **epoch**.
# 
# ```{prf:algorithm} SGD
# 
# **Input**: training data $\mathcal{D}$, network $f_\theta$, step-size $\eta$
# 1. Initialize weights $\theta$ randomly
# 2. **for** epoch $t\in\{1,\ldots,t_{max}\}$
#     1. Divide the data into mini-batches $\mathcal{B}_1\cup\ldots\cup\mathcal{B}_m=\mathcal{D}$
#     2. **for** each batch $\mathcal{B}$
#         1. $\theta\gets \theta -\eta\nabla \mathcal{L}(\mathcal{B},\theta)$
# ```
# Likewise, we can integrate momentum in this optimization scheme, by performing the updates on a batch only.
# 
# #### Properties of SGD
# SGD is not only more efficient in its computation of the gradients, it also has many theoretical advantages to gradient descent. The computation of the gradients on batches only introduces noise into the gradient descent updates, that are often useful in navigating complex loss landscapes and escaping local minima. 
# 
# {numref}`loss_ladscape_fig` gives an impression of the landscape of a neural network with only two parameters. We see how there are multiple local optima and only one very narrow valley that forms the global optimum. SGD can help in these cases, because it's randomness helps to survey the loss landscape without getting stuck in smaller local minima.  
# 
# :::{figure} /images/neuralnets/loss_landscape.png
# ---
# height: 320px
# name: loss_ladscape_fig
# align: center
# ---
# Loss landscape of neural networks for a bi-dimensional parameter space (borrowed from {cite}`li2017visualizing`). There are several local minima and one global minimum.
# :::
# 
# Generally, sharp minima are considered as less desirable since they are associated with poor generalization (overfitting). Flat minima often generalize better and there are some works that support that these are found with smaller batch sizes (more noisy SGD steps). Due to these properties, SGD is also considered to perform an **implicit regularization**. That is, although the global optimum of the loss would return a vastly overfitting model, SGD is capable to find a minimum that corresponds to a regularizing model. 
# 
# However, the randomness of SGD might also be a problem when it comes to the convergence of this method. In fact, SGD requires a decreasing step-size to converge. Hence it requires a learning rate scheduler, and a good learning rate scheduler is in practice often not trivial to find. Additionally, SGD can greatly benefit from additional optimization techniques like momentum.
# 
# ### Training with SGD in Pytorch
# We have used the Pytorch SGD optimizer already in the previous example, but there we computed gradient descent because we didn't split the data into batches. In order to perform SGD, we load the data into a `DataLoader`. In this example, we set the `batch_size=32`, meaning that each update is performed on 32 randomly sampled examples. The parameter `shuffle=True` reshuffles the dataset in every epoch, such that we get a new split in every epoch.     
# We train here a simple shallow network with a single hidden layer of 16 neurons and ReLU activation. When we perform the evaluation to track the loss and accuracy on the train and test-data, we use the command `with torch.no_grad():` to perform the computations without simultaneous gradient tracking.     
# 
# Feel free to try playing with the parameters yourself. Note in particular how we need now only 100 epochs to train this simple model, while we needed 1000 gradient descent steps and a multiple-layer network to achieve a similar accuracy.

# In[1]:


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Generate and preprocess data
X, y = make_moons(n_samples=1000, noise=0.2, random_state=42)
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)

# Create DataLoader for mini-batch SGD
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Define a simple neural network model
model = nn.Sequential(
    nn.Linear(2, 16),
    nn.ReLU(),
    nn.Linear(16, 2)
)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

# Store metrics
train_accs, test_accs = [], []
train_losses, test_losses = [], []

# Training loop
for epoch in range(100):
    model.train()
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

    # Evaluate on full train/test sets
    model.eval()
    with torch.no_grad():
        train_outputs = model(X_train)
        test_outputs = model(X_test)

        train_loss = criterion(train_outputs, y_train).item()
        test_loss = criterion(test_outputs, y_test).item()
        train_losses.append(train_loss)
        test_losses.append(test_loss)

        train_preds = train_outputs.argmax(dim=1)
        test_preds = test_outputs.argmax(dim=1)

        train_acc = (train_preds == y_train).float().mean().item()
        test_acc = (test_preds == y_test).float().mean().item()

        train_accs.append(train_acc)
        test_accs.append(test_acc)

# After this, train_accs and test_accs contain accuracy trajectories,
# and train_losses and test_losses contain loss trajectories.


# We plot the accuracy and the loss of the training and test data. Quite uncommonly, the test accuracy is here most often higher than the train accuracy. This is usually not the case and due to the low complexity of the data and a test set that probably doesn't contain many points near the decision boundary. We observe the rather noisy progress of the trajectories.

# In[2]:


import matplotlib.pyplot as plt

# Plot accuracy and loss side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Accuracy plot
ax1.plot(train_accs, label='Train Accuracy')
ax1.plot(test_accs, label='Test Accuracy')
ax1.set_title('Accuracy over Epochs')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.legend()
ax1.grid(True)

# Loss plot
ax2.plot(train_losses, label='Train Loss')
ax2.plot(test_losses, label='Test Loss')
ax2.set_title('Loss over Epochs')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Cross-Entropy Loss')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()


# In[ ]:




