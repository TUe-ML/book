#!/usr/bin/env python
# coding: utf-8

# # Decision Trees
# In this section, we first introduce some preliminary definitions on tuples and directed binary trees. Then, we formally introduce Decision Trees using these computational structures.
# 
# ### A word on tuples
# Let $ \boldsymbol{\alpha} = \left( \alpha_{1}, \alpha_{2}, \ldots, \alpha_{n} \right) $ denote[^footnote1] a tuple with $ | \boldsymbol{\alpha} | = n $ elements. For now, we assume that the tuple $ \boldsymbol{\alpha} $ is homogeneous in the sense that all elements belong to the same set $ \Omega $, i.e. $$ \boldsymbol{\alpha}[k] \triangleq \alpha_{k} \in \Omega , \forall k \in \lbrace 1, \ldots, n \rbrace. $$ 
# 
# ```{prf:definition}
# Now, let $ \boldsymbol{\beta} = \left( \beta_{1}, \beta_{2}, \ldots, \beta_{m} \right) $ be another tuple with $ | \boldsymbol{\beta} | = m $ elements with $ \beta_{k} \in \Omega $, $ \forall k \in \lbrace 1, \ldots, m \rbrace $, we define the *concatenation* operator as
# \begin{equation}
# \boldsymbol{\alpha} \concat \boldsymbol{\beta} \triangleq \left( \alpha_{1}, \alpha_{2}, \ldots, \alpha_{n}, \beta_{1}, \beta_{2}, \ldots, \beta_{m} \right)
# \end{equation}
# such that $ | \boldsymbol{\alpha} \concat \boldsymbol{\beta} | = n + m $.
# ```
# 
# Note that the null tuple $ \boldsymbol{\varnothing} $[^footnote2] is such that $ | \boldsymbol{\varnothing} | = 0 $, $ \boldsymbol{\alpha} \concat \boldsymbol{\varnothing} \equiv \boldsymbol{\alpha} $ and $  \boldsymbol{\varnothing} \concat \boldsymbol{\alpha} \equiv \boldsymbol{\alpha} $.
# 
# [^footnote1]: Whenever possible we use boldface Greek letters to denote tuples to distinguish them from vectors. However, the distinction might be implied in the context.
# 
# [^footnote2]: We use the bold symbol $ \boldsymbol{\varnothing} $ to distinguish the null tuple from the empty set $ \emptyset $.
# 
# ```{prf:definition}
# For any $ a \in \Omega $, we can define then the *append* $ \blacktriangleright $ and *prepend* $ \blacktriangleleft $ operators respectively as
# \begin{eqnarray}
# \boldsymbol{\alpha} \blacktriangleright a &\triangleq& \left( \alpha_{1}, \alpha_{2}, \ldots, \alpha_{n} \right) \concat \left( a \right)  \nonumber \\
# &\equiv& \left( \alpha_{1}, \alpha_{2}, \ldots, \alpha_{n}, a \right) \nonumber
# \end{eqnarray}
# and
# \begin{eqnarray}
# a \blacktriangleleft \boldsymbol{\alpha} &\triangleq& \left( a \right) \concat \left( \alpha_{1}, \alpha_{2}, \ldots, \alpha_{n} \right) \nonumber \\
# &\equiv& \left( a, \alpha_{1}, \alpha_{2}, \ldots, \alpha_{n} \right). \nonumber
# \end{eqnarray}
# ```
# 
# Finally, for convenience, we define the following notation to get the first $ k $ elements of a tuple $ \boldsymbol{\alpha} $
# \begin{equation}
# \boldsymbol{\alpha}[:\! k] \triangleq \left( \alpha_{1}, \alpha_{2}, \ldots, \alpha_{k} \right)
# \end{equation}
# with $ k \in \lbrace 1, \ldots, | \boldsymbol{\alpha} | \rbrace $. By convention, we also assume that $ \boldsymbol{\alpha}[0] \triangleq \boldsymbol{\varnothing} $.
# 
# ```{prf:remark}
# We can also design an heterogeneous tuple such that it stores different types of objects e.g. other tuples, numbers and sets. In this case, $ \boldsymbol{\alpha}[k] \in \Omega_{k} $, $ \forall k \in \lbrace 1, \ldots, n \rbrace $, and $ \exists k, k' \in \lbrace 1, \ldots, n \rbrace \mid \Omega_{k} \neq \Omega_{k'} $ with $ n = | \boldsymbol{\alpha} | $.
# ```
# 
# ### Directed binary trees
# 
# ```{prf:definition}
# Let the triplet
# \begin{eqnarray}
# {\cal T}_{\boldsymbol{\eta}} &\triangleq& \left( {\cal C}_{\boldsymbol{\eta}}, {\cal T}_{\boldsymbol{\eta} \blacktriangleright 0}, {\cal T}_{\boldsymbol{\eta} \blacktriangleright 1} \right) \nonumber \\
# &\equiv & {\cal C}_{\boldsymbol{\eta}} \blacktriangleleft \left( {\cal T}_{\boldsymbol{\eta} \blacktriangleright 0}, {\cal T}_{\boldsymbol{\eta} \blacktriangleright 1} \right) \nonumber
# \end{eqnarray}
# define an arbitrary node of a binary tree which is uniquely addressed by a sequence of $ 0 $'s and $ 1 $'s stored in another tuple $ \boldsymbol{\eta} $.
# ```
# 
# According to this notation, $ {\cal C}_{\boldsymbol{\eta}} $ stores the content of the node $ {\cal T}_{\boldsymbol{\eta}} $ whereas the tuples $ {\cal T}_{\boldsymbol{\eta} \blacktriangleright 0} $ and  $ {\cal T}_{\boldsymbol{\eta} \blacktriangleright 1} $ denote respectively its left and right branches, or sub-trees. Alternatively, the numbers being appended to $ \boldsymbol{\eta} $ can also be thought as the indexes of the left ($0$) and right ($1$) edges emanating from ${\cal T}_{\boldsymbol{\eta}} $.
# 
# ```{figure} /images/classification/binary_tree_node.svg
# ---
# height: 320px
# name: binary_tree_node_fig
# align: left
# ---
# A binary tree node $ {\cal T}_{\boldsymbol{\eta}} $ with content $ {\cal C}_{\boldsymbol{\eta}} $ -- e.g. another tuple, a set or a number -- and its left and right sub-trees, respectively $ {\cal T}_{\boldsymbol{\eta} \blacktriangleright 0} $ and  $ {\cal T}_{\boldsymbol{\eta} \blacktriangleright 1} $, illustrated by triangles. Note that $ {\cal T}_{\boldsymbol{\eta}} $ is also a tree by itself (dotted triangle).
# ```
# 
# Additionally, the root node is defined as 
# \begin{eqnarray}
# {\cal T}_{\boldsymbol{\varnothing}} &=& \left( {\cal C}_{\boldsymbol{\varnothing}}, {\cal T}_{\left( 0 \right)}, {\cal T}_{\left( 1 \right)} \right) \nonumber \\
# &\equiv& {\cal C}_{\boldsymbol{\varnothing}} \blacktriangleleft \left( {\cal T}_{\left( 0 \right)}, {\cal T}_{\left( 1 \right)} \right) \nonumber .
# \end{eqnarray}
# On the other hand, a leaf node addressed by the sequence $ {\boldsymbol{\ell}} $ must satisfy the following restriction
# \begin{eqnarray}
# {\cal T}_{\boldsymbol{\ell}} &=& \left( {\cal C}_{\boldsymbol{\ell}}, \boldsymbol{\varnothing}, \boldsymbol{\varnothing} \right) \nonumber \\
# &\equiv& {\cal C}_{\boldsymbol{\ell}} \blacktriangleleft \left( \boldsymbol{\varnothing}, \boldsymbol{\varnothing} \right). \nonumber
# \end{eqnarray}
# Note that we can simple denote a tree $ {\cal T} $ by referring to its root node $ {\cal T}_{\boldsymbol{\varnothing}} $, i.e. $ {\cal T} \equiv {\cal T}_{\boldsymbol{\varnothing}} $. Finally, it is worth mentioning that the sequence of $ 0 $'s and $ 1 $'s in $ \boldsymbol{\eta} $ uniquely identifies the edges along the path from the root $ {\cal T}_{\boldsymbol{\varnothing}} $ to node $ {\cal T}_{\boldsymbol{\eta}} $.
# 
# ````{prf:example}
# The adopted notation to represent binary trees is useful for theorem proofing and writing concise pseudo-codes which employ recursion. However, as illustrated in {numref}`binary_tree1_fig`, this notation is not suitable to explicitly express binary trees as it is not human friendly at all. Note though that there are alternative notations. Choose whichever notation you prefer, but make sure it is correct and you use it consistently, i.e. you use the same notation in our proofs and pseudo-codes.
# 
# ```{figure} /images/classification/binary_tree1.svg
# ---
# height: 320px
# name: binary_tree1_fig
# align: left
# ---
# A binary tree $ {\cal T} = \left( 100, \left( 40, \boldsymbol{\varnothing}, \boldsymbol{\varnothing} \right), \left(60, \left( 25, \boldsymbol{\varnothing}, \boldsymbol{\varnothing} \right), \left( 35, \left( 15, \boldsymbol{\varnothing}, \boldsymbol{\varnothing} \right), \left(20, \boldsymbol{\varnothing}, \boldsymbol{\varnothing} \right) \right) \right) \right) $. Alternatively, $ {\cal T} = 100 \blacktriangleleft \left(40 \blacktriangleleft \left( \boldsymbol{\varnothing}, \boldsymbol{\varnothing} \right), 60 \blacktriangleleft \left(  25 \blacktriangleleft \left( \boldsymbol{\varnothing}, \boldsymbol{\varnothing} \right), 35 \blacktriangleleft \left(  15 \blacktriangleleft \left( \boldsymbol{\varnothing}, \boldsymbol{\varnothing} \right), 20 \blacktriangleleft \left( \boldsymbol{\varnothing}, \boldsymbol{\varnothing} \right) \right) \right) \right) $. Note that the sequence $ \boldsymbol{\ell} = \left( 1, 1, 0 \right) $ uniquely identifies the path from root $ {\cal T}_{\boldsymbol{\varnothing}} $ to the leaf $ {\cal T}_{\boldsymbol{\ell}} = 15 \blacktriangleleft \left( \boldsymbol{\varnothing}, \boldsymbol{\varnothing} \right) $.
# ```
# ````
# 
# ### Decision tree definition
# 
# Let us consider again a data item $ {\bf x} $ in a high-dimensional feature space $$ {\cal X} \triangleq {\cal X}_{1} \times {\cal X}_{2} \times \ldots \times {\cal X}_{D} $$ such that each feature $ x_{d} \in {\cal X}_{d} $, $ d \in \lbrace 1, \ldots, D \rbrace $, is either discrete or continuous.
# 
# ```{prf:definition}
# A Decision Tree (DT) is a directed binary tree with two types of nodes
# * **Decision nodes:** internal (non-leaf) nodes apply decision rules to split the data based on feature values. More precisely, each decision node $ \boldsymbol{\eta} $ is associated with some feature $ d_{\boldsymbol{\eta}} $ and some boolean function $ f_{\boldsymbol{\eta}} $ which maps values along the $ d_{\boldsymbol{\eta}} $-dimension of the feature space $ {\cal X} $ into false ($ 0 $) or true ($ 1 $). More concisely, $ f_{\boldsymbol{\eta}} : {\cal X}_{d_{\boldsymbol{\eta}}} \rightarrow \lbrace 0, 1 \rbrace $.
# * **Prediction nodes:** leaves store in turn what is needed for the prediction itself. In particular, each leaf node $ \boldsymbol{\ell} $ is associated with either a fixed label $ y_{\boldsymbol{\ell}} \in {\cal Y} $ or, more generically, a distribution $ P_{\boldsymbol{\ell}} (y) $ over possible labels $ y \in {\cal Y} $, in this case a p.m.f. assigning probabilities to labels in $ {\cal Y} $.
# ```
# 
# Therefore, we can write a decision node as
# \begin{equation}
# {\cal T}_{\boldsymbol{\eta}} = \left( f_{\boldsymbol{\eta}}(\cdot), d_{\boldsymbol{\eta}} \right) \blacktriangleleft \left( {\cal T}_{\boldsymbol{\eta} \blacktriangleright 0}, {\cal T}_{\boldsymbol{\eta} \blacktriangleright 1} \right), \nonumber
# \end{equation}
# whereas a prediction node can be written as either
# \begin{equation}
# {\cal T}_{\boldsymbol{\ell}} = y_{\boldsymbol{\ell}}  \blacktriangleleft \left( \boldsymbol{\varnothing}, \boldsymbol{\varnothing} \right), \nonumber
# \end{equation}
# or, when the leaves store distributions over the class values, as
# \begin{equation}
# {\cal T}_{\boldsymbol{\ell}} = P_{\boldsymbol{\ell}}(\cdot) \blacktriangleleft \left( \boldsymbol{\varnothing}, \boldsymbol{\varnothing} \right). \nonumber
# \end{equation}
# 
# Lastly, note that the root node has the form
# \begin{equation}
# {\cal T}_{\boldsymbol{\varnothing}} = \left( f_{\boldsymbol{\varnothing}}(\cdot), d_{\boldsymbol{\varnothing}} \right) \blacktriangleleft \left( {\cal T}_{\left(0\right)}, {\cal T}_{\left(1\right)} \right). \nonumber
# \end{equation}
# 
# ### Decision tree prediction
# Let $ {\bf x} = \begin{bmatrix} x_{1} & \ldots & x_{d} & \ldots & x_{D} \end{bmatrix}^{T} $ be an arbitrary feature vector in $ {\cal X} $ with $ x_{d} $ denoting the feature value at its $ d $-dimension and $ {\cal T} \equiv {\cal T}_{\boldsymbol{\varnothing}} $ be a trained decision tree, the DT classifier can be written simple as
# \begin{equation}
# h_{DT}({\bf x}; {\cal T}) = y_{\boldsymbol{\ell}}, \nonumber 
# \end{equation}
# or, alternatively, as
# \begin{equation}
# h_{DT}({\bf x}; {\cal T}) = \argmax_{y \in {\cal Y}} P_{\boldsymbol{\ell}} (y), \nonumber 
# \end{equation}
# where the leaf node $ \boldsymbol{\ell} $ is obtained through the recursion
# \begin{eqnarray}
# \left( \left( f_{\boldsymbol{\eta}}(\cdot), d_{\boldsymbol{\eta}} \right), {\cal T}_{\boldsymbol{\eta} \blacktriangleright 0}, {\cal T}_{\boldsymbol{\eta} \blacktriangleright 1} \right) &\gets& {\cal T}_{\boldsymbol{\eta}} \\
# \boldsymbol{\eta} &\gets& \boldsymbol{\eta} \blacktriangleright f_{\boldsymbol{\eta}}(x_{d_{\boldsymbol{\eta}}})
# \end{eqnarray}
# with $ {\cal T}_{\boldsymbol{\eta} \blacktriangleright 0} \neq \boldsymbol{\varnothing} \wedge {\cal T}_{\boldsymbol{\eta} \blacktriangleright 1} \neq \boldsymbol{\varnothing} $ and $ f_{\boldsymbol{\eta}}(x_{d_{\boldsymbol{\eta}}}) \in \lbrace 0, 1 \rbrace $ corresponding to the decision taken by node $ \boldsymbol{\eta} $. This recursion is initialized with $ \boldsymbol{\eta} \gets \boldsymbol{\varnothing} $ and stops when a leaf is reached, i.e. when the following stop criteria is satisfied
# \begin{equation}
# \left( y_{\boldsymbol{\eta}}, \boldsymbol{\varnothing}, \boldsymbol{\varnothing} \right) \gets {\cal T}_{\boldsymbol{\eta}}
# \end{equation}
# or, for leaves storing distributions over the class values,
# \begin{equation}
# \left( P_{\boldsymbol{\eta}}(\cdot), \boldsymbol{\varnothing}, \boldsymbol{\varnothing} \right) \gets {\cal T}_{\boldsymbol{\eta}}.
# \end{equation}
# In this case, we just make $ \boldsymbol{\ell} \gets \boldsymbol{\eta} $.
# 
# {prf:ref}`dt_classifier` summarizes how the DT classifier performs its prediction. We call it initially as $ h_{DT}( {\bf x} $, $ {\cal T}_{\boldsymbol{\varnothing}}) $ to obtain the predicted label. Note that the tree decision / branch selection at deep $ | \boldsymbol{\eta} | $ is obtained by evaluating the function $ f_{\boldsymbol{\eta}}(\cdot) $ at $ d_{\boldsymbol{\eta}} $-th feature of the input vector $ {\bf x} $. Specifically, we need to check whether $ f_{\boldsymbol{\eta}}(x_{d_{\boldsymbol{\eta}}}) $ is true ($ 1 $) or false ($0$).
# 
# ```{prf:algorithm} DT classifier
# :label: dt_classifier
# 
# **Inputs** The feature vector ${\bf x} $ and a trained DT $ {\cal T}_{\boldsymbol{\eta}} $.
# 
# **Output** The predicted label $ \hat{y}_{\ell} $
# 
# **Function** $ h_{DT}({\bf x}; {\cal T}_{\boldsymbol{\eta}}) $
# 1. $ \left( {\cal C}_{\boldsymbol{\eta}}, {\cal T}_{\boldsymbol{\eta} \blacktriangleright 0}, {\cal T}_{\boldsymbol{\eta} \blacktriangleright 1} \right) \gets {\cal T}_{\boldsymbol{\eta}} $
# 2. **if** $ {\cal T}_{\boldsymbol{\eta} \blacktriangleright 0} \neq \boldsymbol{\varnothing} \wedge {\cal T}_{\boldsymbol{\eta} \blacktriangleright 1} \neq \boldsymbol{\varnothing} $ *(Check if a leaf was reached)*
# 	1. $ \left( f_{\boldsymbol{\eta}}(\cdot), d_{\boldsymbol{\eta}} \right) \gets {\cal C}_{\boldsymbol{\eta}} $
#     2. $ \boldsymbol{\eta}^{\prime} \gets \boldsymbol{\eta} \blacktriangleright f_{\boldsymbol{\eta}}(x_{d_{\boldsymbol{\eta}}}) $
#     3. **return** $h_{DT} ({\bf x}; {\cal T}_{\boldsymbol{\eta}^{\prime}}) $ *(Decision / branch selection)*
# 3. **else**
# 	1. $ P_{\boldsymbol{\ell}} (\cdot) \gets {\cal C}_{\boldsymbol{\eta}} $
#     2. **return** $ \argmax_{y \in {\cal Y}} P_{\boldsymbol{\ell}} (y) $ *(Compute prediction)*
# ```
# 
# ````{margin}
# ```{note}
# In practice, we store categorical distributions at the leaves. In this case, the probability $ P_{\boldsymbol{\ell}} (\hat{y}) $ can be seen as a confidence measure of the prediction $ \hat{y}_{\boldsymbol{\ell}} = h_{DT}(\check{\bf x}, {\cal T}_{\boldsymbol{\varnothing}}) $ obtained from a leaf $ \boldsymbol{\ell} $ with associated p.m.f. $ P_{\boldsymbol{\ell}}(\cdot) $.
# ```
# ````
# 
# ### Decision tree learning
# Let the dataset partition 
# ```{math}
# :label: dataset_recursion
# {\cal D}_{\boldsymbol{\eta} \blacktriangleright a} \triangleq \lbrace \left( {\bf x}', y' \right) \in {\cal D}_{ \boldsymbol{\eta}} \mid f_{{\boldsymbol{\eta}}}({x}'_{d_{\boldsymbol{\eta}}}) = a \rbrace
# ```
# collect all training examples in $ {\cal D}_{ \boldsymbol{\eta}} $ that satisfies the test $  f_{{\boldsymbol{\eta}}}({\bf x}'_{d_{\boldsymbol{\eta}}}) = a $ for some $ a \in \lbrace 0, 1 \rbrace $. Note that we can recursively obtain the dataset partition $ {\cal D}_{\boldsymbol{\ell}} $ at a leaf $ \boldsymbol{\ell} $ from the full training dataset $ {\cal D} $ by first assigning it to the root node, i.e. the recursion must be initialized with $ {\cal D}_{\boldsymbol{\varnothing}} = {\cal D} $, and then by making $ a = \boldsymbol{\ell}[k] $ in {eq}`dataset_recursion` at each recursion iteration $ k \in \lbrace 1, \ldots, | \boldsymbol{\ell} | \rbrace $. Therefore, the first recursion iteration ($ k = 1 $) must computed as
# \begin{equation}
# {\cal D}_{\boldsymbol{\varnothing} \blacktriangleright \boldsymbol{\ell}[1]} = \lbrace \left( {\bf x}', y' \right) \in {\cal D}_{\boldsymbol{\varnothing}} \mid f_{\boldsymbol{\varnothing}}({x}'_{d_{\boldsymbol{\varnothing}}}) = \boldsymbol{\ell}[1] \rbrace. \nonumber
# \end{equation}
# 
# {prf:ref}`dt_training` summarizes the so-called Classification and Regression Tree (CART) algorithm {cite}`breiman1984classification` -- a greed learning algorithm for both classification and regression -- with hyper-parameters $ k_{max} $, $ \delta_{min} $ and $ {\cal F} $. The DT is recursively created by splitting the dataset $ {\cal D} $ according to a cost function. Each split is performed considering feature values over some chosen dimension $ d \in {\cal F} $. The subset $ {\cal F} \subseteq \lbrace 1, \ldots, D \rbrace $ contains the allowed dimensions for splitting. The recursion is stopped when the maximum number of iterations $ k_{max} $ is reached, i.e. the tree depth is bounded by $ k_{max} $, or the cost improvement relative to the previous iteration is below a given threshold $ \delta_{min} $.
# 
# ```{prf:algorithm} Classification and Regression Tree
# :label: dt_training
# 
# **Inputs** The dataset partition ${\cal D}_{\boldsymbol{\eta}} $ assigned to the node $ \boldsymbol{\eta} $, the current cost $c$; maximum number of iterations $ k_{max} $; minimum cost improvement $ \delta_{min} $; allowed features $ {\cal F} $.
# 
# **Output** The decision tree $ {\cal T}_{\boldsymbol{\eta}} $ corresponding to the node $ \boldsymbol{\eta} $'s branch
# 
# **Function** $ CART({\cal D}_{\boldsymbol{\eta}}, c; k_{max}; \delta_{min} ; {\cal F}) $
# 1. $ \left( f_{\boldsymbol{\eta}}(\cdot), d_{\boldsymbol{\eta}}, {\cal D}_{\boldsymbol{\eta} \blacktriangleright 0}, {\cal D}_{\boldsymbol{\eta} \blacktriangleright 1}, c' \right) \gets FindMinCostSplitDecision ({\cal D}_{\boldsymbol{\eta}};  {\cal F}) $
# 2. $ k \gets | \boldsymbol{\eta} | + 1  $ 
# 3. **if** $ k \geq k_{max}$ $ \vee $ $ |c' - c| \leq \delta_{min} $ *(Stop criteria)*
# 	1. $ {\cal D}_{\boldsymbol{\ell}} \gets {\cal D}_{\boldsymbol{\eta}} $
# 	2. $ P_{\boldsymbol{\ell}} (y) \gets \frac{1}{|{\cal D}_{\boldsymbol{\ell}}|} \sum_{\left({\bf x}', y'\right) \in {\cal D}_{\boldsymbol{\ell}}} \left[ y' = y \right]$, for all $ y \in {\cal Y} $ *(Empirical leaf distribution)*
# 	3. $ {\cal C}_{\boldsymbol{\ell}} \gets P_{\boldsymbol{\ell}} (\cdot) $
# 	4. $ {\cal T}_{\boldsymbol{\ell}} \gets \left( {\cal C}_{\boldsymbol{\ell}}, \boldsymbol{\varnothing}, \boldsymbol{\varnothing} \right) $ *(Create prediction node)*
#     5. **return** $ {\cal T}_{\boldsymbol{\ell}} $
# 4. **else**
# 	1. $ {\cal C}_{\boldsymbol{\eta}} \gets \left( f_{\boldsymbol{\eta}}(\cdot), d_{\boldsymbol{\eta}} \right) $
# 	2. $ {\cal T}_{\boldsymbol{\eta} \blacktriangleright 0} \gets CART({\cal D}_{\boldsymbol{\eta} \blacktriangleright 0}, c'; k_{max}; \delta_{min}; {\cal F}) $
# 	3. $ {\cal T}_{\boldsymbol{\eta} \blacktriangleright 1} \gets CART({\cal D}_{\boldsymbol{\eta} \blacktriangleright 1} , c'; k_{max}; \delta_{min}; {\cal F}) $
# 	4. $ {\cal T}_{\boldsymbol{\eta}} \gets \left( {\cal C}_{\boldsymbol{\eta}}, {\cal T}_{\boldsymbol{\eta} \blacktriangleright 0}, {\cal T}_{\boldsymbol{\eta} \blacktriangleright 1} \right) $ *(Create decision node)*
# 	5. **return** $ {\cal T}_{\boldsymbol{\eta}} $
# ```
# 
# The decision tree is built by calling the CART algorithm with the full training dataset $ {\cal D} $ and the initial cost set to infinity, i.e. $$ {\cal T} \gets CART({\cal D}, \infty; k_{max}; \delta_{min}; {\cal F}). $$ 
# 
# Now, let $ {\cal S} $ be some subset of the dataset $ {\cal D} $ and let the set $ {\cal F}_{d}({\cal S}) $ enclose all possible decisions of the type $ f_{d}:{\cal X}_{d} \rightarrow \lbrace 0, 1 \rbrace $ along the $d$-th dimension. Note that $ {\cal F}_{d}({\cal S}) $ is a function of $ {\cal S} $. The cost of further splitting $ {\cal S} \subseteq {\cal D}$ into new disjoint partitions or splits ${\cal S}_{0}$ and ${\cal S}_{1}$ such that $ {\cal S}_{0} \cup {\cal S}_{1} = {\cal S} $ and $ {\cal S}_{0} \cap {\cal S}_{1} = \emptyset $ is then defined as
# ```{math}
# :label: split_cost
# Cost({\cal S}_{0}, {\cal S}_{1}) = \underbrace{\left\lbrace \frac{|{\cal S}_{0}|}{|{\cal S}|} Impurity({\cal S}_{0}) + \frac{|{\cal S}_{1}|}{|{\cal S}|} Impurity({\cal S}_{1}) \right\rbrace}_{\mbox{new combined impurity}} - \underbrace{Impurity({\cal S})}_{\mbox{previous impurity}},
# ```
# in which the function $ Impurity(\cdot) $ is a measure of the impurity of a dataset split based on its labels such that a split containing samples with the same label has minimum impurity and a split in which all labels are distinct has maximum impurity. The minimum cost split of $ {\cal S} $ according to a cost function $ Cost(\cdot) $ can be found as in {prf:ref}`find_min_cost_split_decision`.
# 
# ```{prf:algorithm} Find minimum cost split decision
# :label: find_min_cost_split_decision
# 
# **Inputs** Subset of the dataset $ {\cal S} $, allowed features $ {\cal F} $
# 
# **Output** The result $ {\cal R}_{min} $ containing the minimum cost split of $ {\cal S} $
# 
# **Function** $ FindMinCostSplitDecision({\cal S}; {\cal F}) $
# 1. $ c_{min} \gets \infty $
# 2. $ {\cal R}_{min} \gets \boldsymbol{\varnothing} $ 
# 3. **for** $ d \in {\cal F} \subseteq \lbrace 1, \ldots, D \rbrace $
#     1. **for**  $ f_{d}(\cdot) \in  {\cal F}_{d}({\cal S})$ *(Traverse all possible decisions over $d$-th dimension)*
#     2. $ {\cal S}_{0} \gets \lbrace \left( {\bf x}', y' \right) \in {\cal S}  \mid f_{d}({x}'_{d}) = 0 \rbrace $
#     3. $ {\cal S}_{1} \gets \lbrace \left( {\bf x}', y' \right) \in {\cal S}  \mid f_{d}({x}'_{d}) = 1 \rbrace $
#     4. $ c \gets  Cost({\cal S}_{0}, {\cal S}_{1}) $ *(Compute split cost)*
#     5. **if** $ c < c_{min} $
#         1. $ c_{min} \gets c $
#         2. $ {\cal R}_{min} \gets \left( f_{d}(\cdot), d, {\cal S}_{0}, {\cal S}_{1}, c_{min} \right) $ *(Store intermediate results)*
# 4. **return** $ {\cal R}_{min} $ *(Return the minimum cost result)*
# ```
# 
# The set $ {\cal F}_{d}({\cal S}) $ can be designed in many ways depending on whether the $ d $-th feature takes discrete or continuous values. For example, for continuous features, the data items in $ {\cal S} $ are typically sorted along the $ d $-th dimension and the average of consecutive sorted values are selected as possible decision thresholds. For one particular threshold $ t_{d} $ (out of the $ |{\cal S}| - 1 $ possible thresholds), we write $$ f_{d}(x_{d}) = \left[ x_{d} > t_{d} \right]. $$ Alternatively, let us assume that the $d$-dimension of the feature space $ {\cal X} $ corresponds to a finite discrete alphabet $ {\cal X}_{d} \triangleq \lbrace \xi_{d}^{(1)}, \xi_{d}^{(2)}, \ldots, \xi_{D}^{(|{\cal X}_{d}|)} \rbrace $, i.e. $ x_{d} \in {\cal X}_{d}$. Moreover, let the set $ {\cal S}_{d} \triangleq \lbrace x_{d}^{(1)},  x_{d}^{(2)}, \ldots,  x_{d}^{(|{\cal S}|)} \rbrace $ collect the discrete values along the $d$-th dimension of the data items in $ {\cal S} \subseteq {\cal D} $. Note that the alphabet $ {\cal X}_{d} $ contains unique values, albeit the set $ {\cal S}_{d} $ may contain replicated values. Then, the set $ {\cal S} $ can be split according to one of the following criteria
# 
# * **One-versus-all:** check all binary split decisions of the type 
# $$ {\cal S} \rightarrow \underbrace{\lbrace \left( {\bf x}', y' \right) \in {\cal S} \mid x'_{d} = \xi_d^{(i)} \rbrace}_{{\cal S}_{0}}, \underbrace{\lbrace \left( {\bf x}', y' \right) \in {\cal S} \mid x'_{d} \neq \xi_d^{(i)} \rbrace}_{{\cal S}_{1}}  $$
# for $ i \in \lbrace 1, 2, \ldots, |{\cal X}_{d}| \rbrace $ and $ x'_{d} \triangleq {\bf x}'[d] $.
# * **Complete split:** create a full split decisions of the type
# $$ {\cal S} \rightarrow \underbrace{\lbrace \left( {\bf x}', y' \right) \in {\cal S} \mid x'_{d} = \xi_d^{(1)} \rbrace}_{{\cal S}_{1}}, \underbrace{\lbrace \left( {\bf x}', y' \right) \in {\cal S} \mid x'_{d} = \xi_d^{(2)} \rbrace}_{{\cal S}_{2}},\ldots, \underbrace{\lbrace \left( {\bf x}', y' \right) \in {\cal S} \mid x'_{d} = \xi_d^{(|{\cal X}_{d}|)} \rbrace}_{{\cal S}_{|{\cal X}_{d}|}}  $$
# which leads to a non-binary decision tree; or
# * **Arbitrary splits:** check for arbitrary binary splits decisions such as 
# $$ {\cal S} \rightarrow \underbrace{\lbrace \left( {\bf x}', y' \right) \in {\cal S} \mid x'_{d} = \xi_d^{(1)} \vee x'_{d} = \xi_d^{(2)} \rbrace}_{{\cal S}_{0}}, \underbrace{\lbrace \left( {\bf x}', y' \right) \in {\cal S} \mid x'_{d} \neq \xi_d^{(1)} \wedge x'_{d} \neq \xi_d^{(2)} \rbrace}_{{\cal S}_{1}}. $$
# 
# ````{margin}
# ```{note}
# Keep in mind that the decision tree structure -- available branches from the root to the leaves -- and its content -- the particular set of decisions assigned to decision nodes and distributions assigned to prediction nodes -- are highly dependent on the training dataset $ {\cal D} $.
# ```
# 
# ```{note}
# Slightly different datasets may generate decision trees with significantly different performances, i.e. classification errors, when evaluated with non-training (validating or testing) data.
# ```
# ````
# 
# ````{prf:remark}
# Typically multiple training samples with different labels might reach the same leaf during training, i.e. the dataset split $ {\cal D}_{\boldsymbol{\ell}} $ associated with a leaf $ \boldsymbol{\ell} $ can be impure. Thus, the corresponding sub-space $ {\cal X}_{\boldsymbol{\ell}} $ can not be associated with a single class value in $ {\cal Y} $. In this case, we store the empirical distribution
# ```{math}
# :label: empirical_leaf_distribution
# P_{\boldsymbol{\ell}} (y) \gets \frac{1}{|{\cal D}_{\boldsymbol{\ell}}|} \sum_{\left({\bf x}', y'\right) \in {\cal D}_{\boldsymbol{\ell}}} \left[ y' = y \right], \forall y \in {\cal Y}
# ```
# at the leaf $ \boldsymbol{\ell} $ as indicated in {prf:ref}`dt_training`. 
# ````
# 

# In[1]:


import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.datasets import load_iris
from sklearn.utils import resample
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 200

# Load the IRIS dataset
iris = load_iris()
n_total_samples = len(iris.target)

# Create a DT classifier and fit it to the dataset
X1, y1 = resample(iris.data, iris.target, n_samples=int(0.75*n_total_samples), replace=False, random_state=0)
clf1 = tree.DecisionTreeClassifier(random_state=0)
clf1 = clf1.fit(X1, y1)

# Plot it nicely
ax1 = plt.subplot(2, 1, 1)
tree.plot_tree(clf1, filled=True, ax=ax1)
ax1.set_title('DT1')

# Create another DT classifier and fit it to the shuffled dataset
X2, y2 = resample(iris.data, iris.target, n_samples=int(0.75*n_total_samples), replace=False, random_state=1)
clf2 = tree.DecisionTreeClassifier(random_state=0)
clf2 = clf2.fit(X2, y2)

# Plot it nicely
ax2 = plt.subplot(2, 1, 2)
tree.plot_tree(clf2, filled=True, ax=ax2)
ax2.set_title('DT2')

# Show all subplots
plt.suptitle("Different DT classifiers fitted to 75% random samples from the IRIS dataset")
plt.show()


# The decisions $ \lbrace \left( f_{\boldsymbol{\eta}}(\cdot), d_{\boldsymbol{\eta}} \right) \rbrace $ at the decision nodes $ \lbrace \boldsymbol{\eta} \rbrace $ are designed such that the resulting DT splits the feature space $ {\cal X} $ into multiple disjoint sub-spaces $ \lbrace {\cal X}_{\boldsymbol{\ell}} \rbrace $ associated with the prediction nodes $ \lbrace \boldsymbol{\ell} \rbrace $ such that
# \begin{eqnarray}
# \bigcup_{\boldsymbol{\ell}}  {\cal X}_{\boldsymbol{\ell} }&=& {\cal X} \nonumber \\
# \bigcap_{\boldsymbol{\ell}}  {\cal X}_{\boldsymbol{\ell}} &=& \emptyset. \nonumber
# \end{eqnarray}
# 
# ```{figure} /images/classification/binary_tree2.svg
# ---
# height: 320px
# name: binary_tree2_fig
# align: left
# ---
# Decision tree.
# ```
# 
# ```{figure} /images/classification/feature_space_splitting.svg
# ---
# height: 320px
# name: feature_space_splitting_fig
# align: left
# ---
# Feature space splitting
# ```
# 
# A DT recursively splitting a bi-dimensional feature space $ {\cal X} $ with continuous features $ {\bf x} = \begin{bmatrix} x_{1} & x_{2} \end{bmatrix}^{T} $ into disjoint sub-spaces $ {\cal X}_{\left( 0 \right)} $, $ {\cal X}_{\left( 1, 0, 0 \right)} $, $ {\cal X}_{\left( 1, 0, 1 \right)} $ and $ {\cal X}_{\left( 1, 1 \right)} $ according to selected thresholds $ t_{1} $, $ t_{2} $ and $ t_{3} $. In this example, decision nodes store functions of the type $ f_{d}(x_{d}; t) = \left[ x_{d} > t_{d} \right] $, in which $ d $ and $ t_{d} $ are the selected feature-space dimension and the decision threshold, respectively. Note that the DT assigns different distributions $  P_{\left( 0 \right)}(y) $, $  P_{\left( 1, 0, 0 \right)}(y) $, $  P_{\left( 1, 0, 1 \right)}(y) $ and $  P_{\left( 1, 1 \right)}(y) $, $ y \in {\cal Y} $, over the partitions of the feature space $ {\cal X} $. For a particular observation $ \check{\bf x} \in {\cal X}_{\left( 1, 0, 1 \right)} $, the prediction is computed using the corresponding distribution as $ \hat{y} = \argmax_{y \in {\cal Y}} P_{\left( 1, 0, 1 \right)}(y) $.
# 
# ```{prf:remark}
# Multiple definitions of impurity can be used. Let $ P_{\cal S}(y) $ denote the empirical distribution of the class labels in the dataset split $ {\cal S} $, i.e.
# \begin{equation}
# P_{\cal S} (y) \gets \frac{1}{|{\cal S}|} \sum_{\left({\bf x}', y'\right) \in {\cal S}} \left[ y' = y \right], \forall y \in {\cal Y}. 
# \end{equation}
# We can define $ Impurity(\cdot) $ in {eq}`split_cost` as the Gini's impurity
# \begin{equation}
# G({\cal S}) \triangleq 1 - \sum_{y \in {\cal Y}} P_{\cal S}(y)^{2},
# \end{equation}
# as the entropy
# \begin{equation}
# E({\cal S}) \triangleq - \sum_{y \in {\cal Y}} P_{\cal S}(y) \log P_{\cal S}(y),
# \end{equation}
# or, alternatively, as the self classification error
# \begin{equation}
# C({\cal S}) \triangleq 1 - \max_{y \in {\cal Y}} P_{\cal S}(y).
# \end{equation}
# ```
