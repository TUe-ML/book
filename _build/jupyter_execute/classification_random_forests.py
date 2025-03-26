#!/usr/bin/env python
# coding: utf-8

# ## Random Forests  
# 
# 
# The performance of a decision tree depends on its structure, that is determined by a greedy method, for example with the CART algorithm. The greedy selection of the best split tends to overfit on the training data, as greedy methods quickly get lost in details because they operate only on local views of the data. We can overcome the tendency to overfit by choosing an early stopping criterion, but we can do even better by combining various trees into an ensemble -- called a random forest. The disadvantage is that we lose the interpretability of a single tree, but on the other hand, we obtain one of the strongest classifiers to this date.  
# 
# The theoretical motivation for the random forest is the bias-variance tradeoff for classification (to be discussed in detail in {ref}`class_evaluation`). Similar to the bias-variance tradeoff in regression, a complex model tends to overfit and consequently has a high variance and low bias. In turn, a simple model has a high bias and low variance. If we now train a couple of overfitting decision trees, for example by setting the stopping criteria pretty lax, such that we get big trees, and if we aggregate the results of those trees, then we get an low bias, low variance classifier.
# 
# 
# 
# ### Inference 
# ```{prf:definition} Random Forest
# A Random Forest classifier is an ensemble of $m$ decision trees $f_{dt1},\ldots,f_{dtm}$, aggregating their predictions. Let $\mathbb{1}(\vvec{p})\in\{0,1\}^c$ denote the one-hot encoded argmax function (also called hardmax), such that $\mathbb{1}(\vvec{p})_y=1$ if and only if $y = \argmax_{1\leq l\leq c} p_l$. We define then the random forest classifier as 
# $$ f_{rf}(\vvec{x})= \frac{1}{m}\sum_{j=1}^m \mathbb{1}(f_{dtj}).$$
# As a result, the random forest predicts the class that wins the majority vote of all decision trees.
# \begin{align*}
# \hat{y} & = \argmax_y\ f_{rf}(\vvec{x})_y\\
# &= \mathrm{mode}(\{\hat{y}_{dtj}\mid 1\leq j\leq m\})
# \end{align*}
# ```
# 
# ### Training
# We generate multiple trees from one dataset with a technique called **bootstrapping**.
# 
# * **Bootstrapping:** use several randomized versions or bootstraps $ \lbrace {\cal D}^{(b)} \rbrace $, $ b \in \lbrace 1, \ldots, B \rbrace $, of the training dataset $ {\cal D} $ to build different decision trees $ \lbrace {\cal T}^{(b)} \rbrace $ using the CART algorithm {cite}`breiman2001random`. We can also randomize the CART algorithm parameters. For example, we can select a particular set of features $ {\cal F}^{(b)} \subset \lbrace 1, \ldots, D \rbrace $ to build each tree $ {\cal T}^{(b)} $; and
# 
# That leads to one of the most powerful existing classifiers today to work with *unstructured* data: the random forests (RFs). 
# 
# ```{prf:theorem} expected prediction error bound
# Defining the margin function of a random forest as
# $$m(\vvec{x},y) = p_\omega(f(\vvec{x};\omega)=y)-\max_{l\neq y}p_\omega(f(\vvec{x};\omega)=l),$$
# and the strength of a random forest as the expected margin 
# $\mu = \mathbb{E}_{\vvec{x},y}m(\vvec{x},y)$.
# Assuming that the expected margin is nonnegative $\mu\geq 0$, then the expected prediction error of a random forest is bounded by
# $$EPE = p(y\neq \argmax_l f_{\mathrm{rf}}(\vvec{x})_l) \leq \rho\frac{1-\mu^2}{\mu^2},$$
# where 
# * $\rho = $
# ```
# ```{prf:proof}
# If the margin is negative, then the prediction of the random forest is not correct. Hence, we can write that the expected prediction error is the probability that the margin is negative 
# :::{math}
# :label: eq:epe_margin
# \begin{align*}
# EPE &= p_{\vvec{x},y}(m(\vvec{x},y)<0)\\
# &= p_{\vvec{x},y}(m(\vvec{x},y)-\mu < -\mu)\\
# &= p_{\vvec{x},y}(\mu - m(\vvec{x},y) > \mu)
# \end{align*}
# :::
# We can apply Chebychev's inequality to the probability above. Chebychev's inequality states that 
# $$p(\lvert x-\mu\rvert\geq w)\leq \frac{\sigma^2}{w^2},$$
# where $\mu$ is the expected value and $\sigma^2$ is the variance of random variable $x$.
# We apply this to  Eq. {eq}`eq:epe_margin`
# \begin{align*}
# EPE&\leq  p_{\vvec{x},y}(\lvert\mu - m(\vvec{x},y)\rvert > \mu)
# \leq \frac{\sigma^2}{\mu^2}.
# \end{align*}
# Now we define the second most frequently predicted label of the random forest as
# \begin{align*}
# s(\vvec{x},y) = \argmax_{l\neq y} p_\omega(\hat{y}_\omega(\vvec{x})=l)
# \end{align*}
# We introduce the Kronecker delta that returns one if two integers $i=j$ and zero otherwise 
# $$\delta(i,j) = \begin{cases}
# 1 & \text{ if } i=j\\
# 0 & \text{ if } i\neq j
# \end{cases}.$$
# Using this notation, we can write the margin as
# \begin{align*}
# m(\vvec{x},y)&= p_\omega(\delta(\hat{y}_\omega(\vvec{x}),y)=1) - p_\omega(\delta(\hat{y}_\omega(\vvec{x}), s(\vvec{x},y)=1))\\
# &=\mathbb{E}_\omega[\delta(\hat{y}_\omega(\vvec{x}),y) -\delta(\hat{y}_\omega(\vvec{x}), s(\vvec{x},y)) ]
# \end{align*}
# because of the linearity of the expected value and 
# ```
# The higher the margin, the more confidently a random forest predicts the correct class $y$ for observation $\vvec{x}$. 
# #### Bootstrapping
# 
# Based on the previous discussion regarding the bias-variance trade off, we can improve the performance of an isolated DT classifier by exploring the average behavior -- bias of the prediction error -- of several DT classifiers trained using different datasets. First, let us create $ B $ randomized versions -- bootstraps -- $ \lbrace {\cal D}^{(b)} \rbrace $, $ b \in \lbrace 1, 2, \ldots, B \rbrace $, of the training dataset
# \begin{eqnarray}
# {\cal D} &=& \bigcup_{i=1}^{N} \lbrace \underbrace{\left( {\bf x}_{i}, y_{i} \right)}_{{\cal D}_{i}} \rbrace \nonumber \\
# &=& \lbrace {\cal D}_{1}, {\cal D}_{2}, {\cal D}_{3}, {\cal D}_{4}, {\cal D}_{5}, {\cal D}_{6}, {\cal D}_{7}, \ldots , {\cal D}_{N} \rbrace \nonumber
# \end{eqnarray}
# by sampling pairs $ {\cal D}_{i} \triangleq \left( {\bf x}_{i}, y_{i} \right) $ from $ {\cal D} $ with replacement -- which means that multiple repetitions of the same pair are allowed in each bootstrap. Specifically, let the p.m.f. $ U_{\cal A}(a) $ denote an uniform distribution over the elements of a finite alphabet $ {\cal A} $ such that $ U_{\cal A}(a) = \frac{1}{|{\cal A}|} $, $ \forall a \in {\cal A} $. We build the $b$-th bootstrap by making $$ {\cal D}^{(b)}_{j} \gets {\cal D}_{i_{j}} $$ where the index $ i_{j} $ is drawn from an uniform distribution over the set $ \lbrace 1, \ldots, N \rbrace $, i.e. $$ i_{j} \sim U_{\lbrace 1, \ldots, N \rbrace}(\xi), $$ for each $ j \in \lbrace 1, 2, \ldots N \rbrace $. 
# 
# We repeat this procedure for $ b \in \lbrace 1, \ldots, B \rbrace $ to obtain $ B $ bootstraps as illustrated below
# \begin{eqnarray}
# {\cal D}^{(1)} &=& \underbrace{\lbrace {\cal D}_{9}, {\cal D}_{8}, {\cal D}_{9}, {\cal D}_{6}, {\cal D}_{2}, {\cal D}_{7}, {\cal D}_{1}, \ldots \rbrace}_{N} \nonumber \\
# {\cal D}^{(2)} &=& \underbrace{\lbrace {\cal D}_{11}, {\cal D}_{6}, {\cal D}_{7}, {\cal D}_{7}, {\cal D}_{1}, {\cal D}_{14}, {\cal D}_{20}, \ldots \rbrace}_{N} \nonumber \\
# {\cal D}^{(3)} &=& \underbrace{\lbrace {\cal D}_{3}, {\cal D}_{9}, {\cal D}_{2}, {\cal D}_{4}, {\cal D}_{17}, {\cal D}_{11}, {\cal D}_{16}, \ldots \rbrace}_{N} \nonumber \\
# &\vdots& \nonumber \\
# {\cal D}^{(B)} &=& \underbrace{\lbrace {\cal D}_{5}, {\cal D}_{18}, {\cal D}_{7}, {\cal D}_{5}, {\cal D}_{1}, {\cal D}_{12}, {\cal D}_{7}, \ldots \rbrace}_{N}. \nonumber
# \end{eqnarray}
# 
# ```{prf:remark}
# Note that each pair $ {\cal D}_{i} \in {\cal D} $, $ \forall i \in \lbrace 1, \ldots, N \rbrace $, can appear multiple times in the same bootstrap. For $ B \rightarrow \infty $, only $ 1 - \frac{1}{e} \approx 63.21 $ \% of the original samples will be presented at each bootstrap. Note also that bootstraps follow aleatory sequences with respect to the sequence of pairs $ \lbrace {\cal D}_{1}, {\cal D}_{2}, {\cal D}_{3}, \ldots \rbrace $ in the original dataset $ {\cal D} $.
# ```
# 
# We can follow a similar procedure to create a subset of features $ {\cal F}^{(b)} \subset \lbrace 1, 2, \ldots, D \rbrace $ but restricted to a smaller number of features such that $ | {\cal F}^{(b)} | < D $.
# 
# Finally, we create an ensemble $ {\cal E} = \lbrace {\cal T}^{(b)} \rbrace $, $ b \in \lbrace 1, 2, \ldots, B \rbrace $, of decision trees using the CART algorithm. In particular, the $b$-th decision tree is created by running the CART algorithm $$ {\cal T}^{(b)} \gets CART({\cal D}^{(b)}, \infty; k_{max}; \delta_{min}; {\cal F}^{(b)}) $$ passing the bootstrap $ {\cal D}^{(b)} $ as input and using a randomized subset of features selected by the parameter $ {\cal F}^{(b)} $. Eventually, we can also randomize the stop criteria -- governed by the parameters $ k_{max} $ and $ \delta_{min} $ -- or the set of possible decisions evaluated at each dimension $ d $ of the feature space $ {\cal X} $.
# 
# ````{margin}
# ```{note}
# Typically, we create ensembles containing $ B = 100, 200, 500, \dots $ randomized decision trees and use a third of the features to train each decision tree $ {\cal T}^{(b)} $ such that $ | {\cal F}^{(b)} | = \frac{D}{3} $. 
# ```
# ````
# 
# ### Aggregating
# 
# Let $ \lbrace \hat{y}^{(b)} \rbrace $ collect the predicted labels obtained by the ensemble $ {\cal E} $. We can select the final ensemble prediction $ \hat{y} $ by allowing the individual trees $ \lbrace {\cal T}^{(b)} \rbrace $ to vote. Specifically, for an arbitrary feature vector $ {\bf x} $, the Random Forest (RF) classifier selects the most frequent predicted label 
# \begin{eqnarray}
#  h_{RF}({\bf x}) &=& \argmax_{y \in {\cal Y} } Fr(y; \lbrace \hat{y}^{(b)} \rbrace) \nonumber \\
# \hat{y}^{(b)} &=& h_{DT}({\bf x}; {\cal T}^{(b)}), \,\, \forall b \in \lbrace 1, \ldots, B \rbrace, \nonumber
# \end{eqnarray}
# where the function $ Fr(a; {\cal A}) \triangleq \sum_{a' \in {\cal A}} \left[ a' = a \right] $ counts the frequency of the value $ a $ in a finite set $ {\cal A} $.
# 
# ```{prf:remark}
# Assume that the decision trees in $ {\cal E} $ were designed to perform a regression task instead of a classification task. A decision tree $ {\cal T}^{(b)} \in {\cal E} $ can do that by storing training samples from $ {\cal D}^{(b)} $ at the leaves. In particular, let the training dataset split $$ {\cal D}_{\boldsymbol{\ell}}^{(b)} = \bigcup_{i=1}^{| {\cal D}_{\boldsymbol{\ell}}^{(b)} | } \lbrace \left( {\bf x}_{i}, z_{i} \right) \rbrace $$ collect the data items at the leaf $ \boldsymbol{\ell} $ associated with an input data item $ {\bf x} $. Recap: the binary sequence $ \boldsymbol{\ell} $ indicates which decisions were satisfied ($1$) or not ($0$) by $ {\bf x} $ along the branch from the root to its corresponding leaf and therefore uniquely identifies the leaf itself. We can use e.g. a convex combination $$ h_{DT}({\bf x}; {\cal T}^{(b)}) = \sum_{i=1}^{| {\cal D}_{\boldsymbol{\ell}}^{(b)} |} \alpha_{i}({\bf x}; {\bf x}_{i}) z_{i} $$ to compute the predicted value such that $ \alpha_{i}({\bf x}; {\bf x}_{i}) \propto || {\bf x} - {\bf x}_{i} || $ and $ \sum_{i=1}^{| {\cal D}_{\boldsymbol{\ell}}^{(b)} |} \alpha_{i}({\bf x}; {\bf x}_{i}) = 1 $. In this case, we can combine the predicted values $ \lbrace \hat{z}^{(b)} \rbrace $ obtained by the ensemble $ {\cal E} $ by simply averaging. Thus, for a given feature vector $ {\bf x} $, we can write 
# \begin{eqnarray}
#  h_{RF}({\bf x}) &=& \frac{1}{B} \sum_{b=1}^{B} \hat{z}^{(b)} \nonumber \\
# \hat{z}^{(b)} &=& h_{DT}({\bf x}; {\cal T}^{(b)}), \,\, \forall b \in \lbrace 1, \ldots, B \rbrace. \nonumber
# \end{eqnarray}
# ```
# 
# ```{prf:remark}
# Let $ CE_{test}^{(b)} $ denote the empirical classification error of the DT classifier associated with the $b$-th decision tree $ {\cal T}^{(b)} $ trained based on the boostrap $ {\cal D}^{(b)} $ and evaluated against a testing dataset $ {\cal D}_{test} $. As the testing dataset $ {\cal D}_{test} $ contains realizations of the unknown distribution $ p^{\ast}({\bf x}, y) $, the empirical error $ CE_{test}^{(b)} $ is also a realization of some random variable $ W_{b} \sim p_{b}(w_{b}) $ with variance $ \sigma_{b}^{2} = E_{W_{b} \sim p_{b}} \lbrace (W_{b} - \mu_{b})^{2} \rbrace  $ and mean $ \mu_{b} = E_{W_{b} \sim p_{b}} \lbrace W_{b} \rbrace $. Analogously, we also assume that the empirical classification error $ CE_{test} $ of the ensemble $ {\cal E} $ is a realization of a random variable $ W_{e} \sim p_{e}(w_{e}) $ with variance $ \sigma_{e}^{2} = E_{W_{e} \sim p_{e}} \lbrace (W_{e} - \mu_{e})^{2} \rbrace $ and mean $ \mu_{e} = E_{W_{e} \sim p_{e}} \lbrace W_{e} \rbrace $. Note that, by combining the predicted labels of the ensemble, we expect that the variance $ \sigma_{e}^{2} $ associated with the empirical classification error $ CE_{test} $ of the ensemble will be smaller than any variance $ \lbrace \sigma_{b}^{2} \rbrace $ associated with the classification errors $ \lbrace CE_{test}^{(b)} \rbrace $ of the individual decision trees $ \lbrace {\cal T}^{(b)} \rbrace $. In addition to this, let us assume that in the best case scenario -- for some aggregation procedure -- $ W_{e} $ is given by the average of the random variables $ \lbrace W_{b} \rbrace $, we expect then that -- in this scenario -- the variance $ \sigma_{e}^{2} $ associated with the ensemble $ {\cal E} $ will be the average of the variances $ \lbrace \sigma_{b}^{2} \rbrace $ associated with the individual classifiers. To summarize the discussion, for an arbitrary aggregation procedure, we write $$ \frac{1}{B} \sum_{b=1}^{B} \sigma_{b}^{2} \leq \sigma_{e}^{2} \leq \sigma_{1}^{2}, \sigma_{2}^{2}, \ldots, \sigma_{B}^{2}, $$ i.e. the aggregation procedure reduces the variance associated with the individual classifiers leading to a more robust ensemble classifier in the sense its empirical error $ CE_{test} $ is less sensitive to the employed training and testing datasets: $ {\cal D} $ and $ {\cal D}_{test} $. Finally, it is worth stressing that this discussion aims to motivate the ensemble approach, it does not intend to offer a rigorous proof regarding the ensemble performance.
# ```
# 
# 

# Additive ensembles $f(x) = \sum_{i=1}^M w_i h_i(x)$ wobei $w_i \in \mathbb R$ und $h_i$ sind DTs        
# Theorie: Die Rademacherkomplexität eines additiven Ensembles ist der gewichtete Durchschnitt der Einzelkomplexitäten jedes Baumes ⇒ im Durchschnitt kleine Bäume ⇒ gute Bound  
#  - Beweis für den gewichteten Durchschnitt https://proceedings.mlr.press/v32/cortesb14.html
#  - Abschätzung über die Komplexität von Bäumen https://arxiv.org/abs/2111.04409
# 
# Boosting: Trainiere kleine (in der Theorie stumps, in der Praxis Tiefe <= 5) trees in Runden       
# * AdaBoost: Trainiere Klassifikationsbäume # In der Praxis vermutlich komplett durch XGBoost und Co. ersetzt 
#     \begin{align*}
#             h_{i+1} &= \arg\min_h L(\sum_{i=1}^M w_i h_i(x) + h(x), D) \text{# Finde ein h}\\
#             w_{i+1}  &= \arg\min_h L(\sum_{i=1}^M w_i h_i(x) + wh(x), D) \text{# Liniensuche für das optimale Gewicht}
#     \end{align*}
# 
# * Gradient boosting: Trainiere Regressionsbäume (für ein Klassifikationsproblem)
# $$h_{i+t} = \arg\min_h L(\sum_{i=1}^M w_i h_i(x) + wh(x), D)$$ wobei D der Datensatz ist und w ein festes Gewicht  # Liniensuche wird meistens nicht mehr gemacht
#     * Bekannte Frameworks/Variationen: XGBoost (GPU Support + Regularisierung im Loss), LightGBM (GPU + sampling based DT Algorithmus), CatBoost (GPU + besonderes Loss für kategorische Variable)
# 
# * Random Forest: Trainiere große Klassifikationsbäume parallel (Bagging + weitere Randomisierung beim Lernen der Bäume)
# (!) Das funktioniert gut entgegen der PAC Bound oben      
# Einzelne große Bäume = "Starke Lerner" ⇒ niedriger Bias, hohe Varianz
# Der Generalisierungsfehler eines RF ist die $\rho (1-s^2) / s^2$ wobei $\rho$ die Korrelation zwischen den Bäumen und s^2 den Bias der Bäume misst 
# https://link.springer.com/article/10.1023/A:1010933404324       
# Eine weitere Erklärung für die Robustheit von RF gegenüber Noise in den Daten ist, dass RF effektiv interpolieren: Korrekte Datenpunkte kommen vermutlich in den meisten Bootstrap samples vor, sodass die meisten Bäume diese korrekt identifizieren, wobei noisy Punkte nur vereinzelnd vorkommen. Diese werden in jedem Baum isoliert (da jeder Baum overfittet) https://arxiv.org/abs/1504.07676
