#!/usr/bin/env python
# coding: utf-8

# ## Naïve Bayes   
# 
# In this section, we recap some important concepts from probability theory. In the sequel, for the sake of simplicity, we introduce the Naive Bayes (NB) classifier assuming a space $ {\cal X} $ of discrete-valued features only. However, the results here can be promptly extended to a space $ {\cal X} $ containing continuous-valued features or a mix of continuous and discrete-valued features.
# 
# ### Marginalization
# 
# Let 
# \begin{eqnarray}
# P({\bf a}, {\bf b}) &\equiv& P_{{\bf A},{\bf B}}({\bf a}, {\bf b}) \nonumber \\
# &\triangleq& Pr \lbrace {\bf A} = {\bf a} \wedge {\bf B} = {\bf b} \rbrace \nonumber
# \end{eqnarray}
# denote the joint probability mass function (p.m.f.)[^footnote1] of the random vectors $ {\bf A} $ and $ {\bf B} $, i.e. $ {\bf A}, {\bf B} \sim P({\bf a}, {\bf b}) $. Furthermore, we assume that the realizations $ {\bf a} $ and $ {\bf b} $ of the random vectors $ {\bf A} $ and $ {\bf B} $ take values in the discrete-valued vector spaces $ \Omega_{a} $ and $ \Omega_{b} $, respectively. 
# 
# [^footnote1]: We use the uppercase notation $ P(\cdot) $ to distinguish p.m.f's from continuous-valued and mixed distributions denoted by $ p(\cdot) $.
# 
# ````{prf:definition} Marginalization
# :label: marginalization
# Then, the marginal p.m.f.
# \begin{eqnarray}
# P({\bf a}) &\equiv& P_{{\bf A}}({\bf a}) \nonumber \\
# &\triangleq& Pr \lbrace {\bf A} = {\bf a} \rbrace  \nonumber
# \end{eqnarray}
# of the random vector $ {\bf A} $ can be computed from the joint p.m.f. $ P({\bf a}, {\bf b}) $ as
# \begin{equation}
# P({\bf a}) = \sum_{{\bf b} \in \Omega_{b}} P({\bf a}, {\bf b}).
# \end{equation}
# Analogously, the marginal p.m.f. 
# \begin{eqnarray}
# P({\bf b}) &\equiv& P_{\bf B}({\bf b}) \nonumber \\
# &\triangleq& Pr \lbrace {\bf B} = {\bf b} \rbrace \nonumber
# \end{eqnarray}
# of the random vector $ {\bf B} $ can be obtained by
# \begin{equation}
# P({\bf b}) = \sum_{{\bf a} \in \Omega_{a}} P({\bf a}, {\bf b}).
# \end{equation}
# 
# ````
# 
# ```{prf:remark} Valid distributions
# Recap: a valid p.m.f. $ P({\bf x}) \equiv P_{\bf X}({\bf x}) $, $ {\bf x} \in \Omega_{x} $, must be non-negative and the sum over its support must be equals one, i.e. it must satisfy the following conditions
# \begin{eqnarray}
# && 0 \leq P({\bf x}) \leq 1, \forall {\bf x} \in \Omega_{x} \,\, \mbox{} \nonumber \\
# && \sum_{{\bf x} \in \Omega_{x}} P({\bf x}) = 1. \nonumber \\
# \end{eqnarray}
# ```
# 
# ### Conditional probability distribution
# 
# ````{prf:definition} Conditional probability distributions
# :label: conditional_probability_distribution
# The p.m.f. $ P({\bf a} \mid {\bf b}) $ of the random vector $ {\bf A} $ conditioned on a particular realization $ {\bf b} $ of the random vector $ {\bf B} $ is defined as 
# \begin{eqnarray}
# P({\bf a} \mid {\bf b}) &\equiv& P_{{\bf A} \mid {\bf b}}({\bf a} \mid {\bf b}) \nonumber \\
# &\triangleq& Pr \lbrace {\bf A} = {\bf a} \mid {\bf B} = {\bf b} \rbrace \nonumber
# \end{eqnarray}
# and can be computed from the joint distribution $ P({\bf a}, {\bf b}) $ as
# ```{math}
# :label: pab_cond
# P({\bf a} \mid {\bf b}) = \frac{P({\bf a}, {\bf b})}{P({\bf b})}.
# ```
# ````
# 
# Note that we can write $ {\bf A} \mid {\bf b} \sim P({\bf a} \mid {\bf b}) $ which indicates the distribution of the random vector $ {\bf A} $ given that the realization $ {\bf b} $ of $ {\bf B} $ has occurred. Conversely, the p.m.f. $ P({\bf b} \mid {\bf a}) $ of the random vector $ {\bf B} $ conditioned on a particular realization $ {\bf a} $ of the random vector $ {\bf A} $ is defined as
# \begin{eqnarray}
# P({\bf b} \mid {\bf a}) &\equiv& P_{{\bf B} \mid {\bf a}}({\bf b} \mid {\bf a}) \nonumber \\
# &\triangleq& Pr \lbrace {\bf B} = {\bf b} \mid {\bf A} = {\bf a} \rbrace, \nonumber
# \end{eqnarray}
# i.e. $ {\bf B} \mid {\bf a} \sim P({\bf b} \mid {\bf a}) $, and is obtained from $ P({\bf a}, {\bf b}) $ by
# ```{math}
# :label: pba_cond
# P({\bf b} \mid {\bf a}) = \frac{P({\bf a}, {\bf b})}{P({\bf a})}.
# ```
# Lastly, from Eqs. {eq}`pab_cond` and {eq}`pba_cond`, we can write
# ```{math}
# :label: equiv
# P({\bf a}, {\bf b}) = P({\bf a} \mid {\bf b}) \, P({\bf b}) = P({\bf b} \mid {\bf a}) \, P({\bf a}).
# ```
# 
# ### Bayes rule
# 
# **Goal:** we seek to obtain the so-called *posterior* distribution $ P({\bf a} \mid \check{\bf b}) $ of the random vector $ {\bf A} $ provided that
# * we have observed the realization $ \check{\bf b} $ of $ {\bf B} $;
# * we know the conditional distribution $ P({\bf b} \mid {\bf a}) $, which models how likely it is to observe a given sample $ {\bf b} $ for each possibly value $ {\bf a} \in \Omega_{a} $;
# * we know the *prior* distribution $ P({\bf a}) $ of the random vector $ {\bf A} $.
# 
# Note that we can rewrite Eq. {eq}`equiv` as
# ```{math}
# :label: bayes_rule1
# P({\bf a} \mid {\bf b}) = \frac{P({\bf b} \mid {\bf a}) \, P({\bf a})}{P({\bf b})}.
# ```
# 
# By plugging the particular observation $ \check{\bf b} $ into {eq}`bayes_rule1`, we write
# ```{math}
# :label: bayes_rule2a
# P({\bf a} \mid \check{\bf b}) = \frac{P(\check{\bf b} \mid {\bf a}) \, P({\bf a})}{P(\check{\bf b})}
# ```
# and them
# ```{math}
# :label: bayes_rule2b
# P({\bf a} \mid \check{\bf b}) \propto  P(\check{\bf b} \mid {\bf a}) \, P({\bf a}),
# ```
# in which the denominator $ P(\check{\bf b}) $ on the right-hand side of {eq}`bayes_rule2a` was omitted in {eq}`bayes_rule2b` since it is a constant -- it is the p.m.f. $ P({\bf b}) $ evaluated at the particular sample $ \check{\bf b} $.
# 
# ````{prf:definition} The glorious Bayes rule
# :label: bayes_rule
# Eq. {eq}`bayes_rule1`, i.e. $$ P({\bf a} \mid {\bf b}) = \frac{P({\bf b} \mid {\bf a}) \, P({\bf a})}{P({\bf b})}, $$ is called the Bayes rule -- a.k.a. the probability inversion rule -- and it is the basis for Bayesian inference. It tells us how the information contained in an observed sample $ \check{\bf b} $ can be assimilated through a model $ P({\bf b} \mid {\bf a}) $ to update a *prior* distribution $ P({\bf a}) $ and compute a *posterior* distribution $ P({\bf a} \mid \check{\bf b}) $ which incorporates the observed data $ \check{\bf b} $.
# ````
# 
# ```{prf:remark} Evidence
# The probability $ P(\check{\bf b}) $ of a sample $ \check{\bf b} $ being observed -- independently of $ {\bf A} $ -- is also called the evidence in {eq}`bayes_rule1` and can be easily computed from the known distributions $ P({\bf a} \mid {\bf b}) $ and $ P({\bf a}) $ as
# \begin{eqnarray}
# P(\check{\bf b}) &=& \sum_{{\bf a'} \in \Omega_{a}} P({\bf a}', \check{\bf b}) \nonumber \\
# &\equiv& \sum_{{\bf a'} \in \Omega_{a}} P({\bf a}' \mid \check{\bf b}) P({\bf a}'), \nonumber
# \end{eqnarray}
# in which the likelihood $ P({\bf a} \mid {\bf b}) $ is evaluated at $ \check{\bf b} $.
# ```
# 
# ````{margin}
# ```{note}
# In general, the *uncertainty* in the *prior* distribution $ P({\bf a}) $ is incrementally decreased in the *posterior* distribution $ P({\bf a} \mid \check{\bf b}_{1}, \check{\bf b}_{2}, \ldots) $ as we assimilate more and more data $ \check{\bf b}_{1}, \check{\bf b}_{2}, \ldots $ by means of the Bayes rule.
# ```
# ````
# 
# ```{prf:remark} Uncertainty as a measure of dispersion
# Uncertainty can be thought of as a measure of the dispersion of a distribution. Let $ | \Omega_{x} | $ denote the cardinality -- i.e. the number of elements -- of a vector space $ \Omega_{x} $ with discrete-valued space dimensions. An uniform distribution over $ \Omega_{x} $
# \begin{equation}
# U({\bf x}) = \frac{1}{|\Omega_{x}|}, \forall {\bf x} \in \Omega_{x} \nonumber
# \end{equation}
# is called *non-informative* p.m.f. since it does not provide any information on the most or less probable values of $ {\bf X} $, i.e. all values are equiprobable. On the other hand, we can think of a deterministic vector $ {\bf x}' $ as being the realization of a random vector $ {\bf X} $ distributed according to a p.m.f.
# \begin{equation}
# P_{\bf X}({\bf x}) = \left[ {\bf x} = {\bf x}' \right] \nonumber
# \end{equation}
# with no uncertainty at all, i.e. all values have null probability except for the most probable value.
# ```
# 
# ### Back to the classification problem
# Let us assume that the $ {\cal X} $ is a $ D $-dimensional space comprising discrete-valued features only, i.e.
# \begin{equation}
# {\cal X} = {\cal X}_{1} \times {\cal X}_{2} \times \ldots {\cal X}_{D},
# \end{equation}
# where $ \lbrace {\cal X}_{d} \rbrace $, $ d \in \lbrace 1, \ldots, D \rbrace $, are finite sets of discrete values or alphabets.
# 
# The feature vector $ {\bf x} \in {\cal X} $ can be written in turn as 
# \begin{equation}
# {\bf x} = \begin{bmatrix} x_{1} & x_{2} & \ldots & x_{D} \end{bmatrix}^{T}
# \end{equation}
# with $ x_{d} \in {\cal X}_{d} $, $ \forall d \in \lbrace 1, \ldots, D \rbrace $.
# 
# Remember that the optimal classifier is given by
# ```{math}
# :label: nb_estimate1a
# h^{\ast}({\bf x}) = \argmax_{y \in {\cal Y}} P^{\ast}(y \mid {\bf x}),
# ```
# where the *unknown* p.m.f. $ P^{\ast}(y \mid {\bf x}) $ can be rewritten using the Bayes rule as
# ```{math}
# :label: bayes_rule3
# P^{\ast}(y \mid {\bf x}) \propto P^{\ast}({\bf x} \mid y) P^{\ast}(y).
# ```
# 
# Thus, by plugging {eq}`bayes_rule3` into {eq}`nb_estimate1a`, we can write
# ```{math}
# :label: nb_estimate1b
# h^{\ast}({\bf x}) = \argmax_{y \in {\cal Y}} P^{\ast}({\bf x} \mid y) P^{\ast}(y),
# ```
# since the underlying proportionality constant in {eq}`bayes_rule3` -- i.e. the reciprocal of $ P^{\ast}({\bf x}) $ -- does not depend on $ y $.
# 
# Note that the prior p.m.f. $ P^{\ast}(y) $ can be approximated by computing the frequencies of each label in the training dataset $ {\cal D} $, i.e.
# ```{math}
# :label: freq_approx1
# P^{\ast}(y) &=& Pr \lbrace Y = y \rbrace \nonumber \\
# &\approx& \frac{1}{N} \sum_{ \left( {\bf x}', y' \right) \in {\cal D}} \left[ y' = y \right]. \nonumber \\
# &\triangleq P(y)
# ```
# 
# On the other hand, finding a suitable approximation for the true conditional p.m.f. $ P^{\ast}({\bf x} \mid y) $ -- without further assumptions on its structure -- can be a really trick task since $ {\bf x} $ resides in such high-dimensional feature space $ {\cal X} $.
# 
# ### Naive Bayes assumption
# 
# ````{prf:definition} Naive Bayes assumption
# Let us further assume that all features $ \lbrace x_{d} \rbrace $, $ d \in \lbrace 1, 2, \ldots, D \rbrace $, are conditionally independent. Thus, conditioned on a class value $ y $, we can write
# ```{math}
# :label: naive_assumption
# P^{\ast}({\bf x} \mid y) = \prod_{d=1}^{D} P^{\ast}(x_{d} \mid y).
# ```
# ````
# 
# Let the dataset partition $ {\cal D}_{y} \triangleq \lbrace \left( {\bf x}', y' \right) \in {\cal D} \mid  y' = y \rbrace $ collect all training examples from the dataset $ {\cal D} $ with class value $ y $. In this case, the true-class conditional distributions $ P^{\ast}(x_{d} \mid y) $ can be approximated as
# ```{math}
# :label: freq_approx2
# P^{\ast}(x_{d} \mid y) &\approx& \frac{1}{|{\cal D}_{y}|} \sum_{\left( {\bf x}', y' \right) \in {\cal D}_{y}} \left[ x_{d}' = x_{d} \right] \nonumber \\
# &\triangleq& P(x_{d} \mid y)
# ```
# with $ {\bf x}' = \begin{bmatrix} x_{1}' & \ldots & x_{d}' & \ldots & x_{D} \end{bmatrix}^{T} $.
# 
# ````{prf:remark} Laplace smoothing
# Note that $ \sum_{y \in {\cal Y}} |{\cal D}_{y}| = |{\cal D}| = N $, however, some of the classes might be under represented in the training dataset $ {\cal D} $. Thus, to avoid hard $ 0 $ or $ 1 $ probabilities, we use Laplace smoothing and rewrite {eq}`freq_approx2` as
# ```{math}
# :label: freq_approx3
# P(x_{d} \mid y) = \frac{\alpha + \sum_{\left( {\bf x}', y' \right) \in {\cal D}_{y}} \left[ x_{d}' = x_{d} \right]}{\alpha |\Omega_{d}| + |{\cal D}_{y}|},
# ```
# where $ \alpha \geq 0 $ is the smoothing parameter such that the approximated probability $ P(x_{d} \mid y) $ will be between the empirical probability as in {eq}`freq_approx2` ($ \alpha = 0 $) and the uniform probability $ \dfrac{1}{|\Omega_{d}|} $ ( $ \alpha \rightarrow \infty $).
# ````
# 
# ````{prf:remark} The naive assumption is quite strong
# For example, let us suppose that the feature vector $$ {\bf x} \in {\cal X} = {\cal X}_{1} \times \ldots \times {\cal X}_{D}, $$ with $ {\cal X}_{d} \triangleq \lbrace 0, 1, \ldots, 255 \rbrace $, $ \forall d \in \lbrace 1, \ldots, D \rbrace $, collect the $8$-bit pixel values of a $ 28 \times 28 $ input image containing pictures of handwritten numbers $ y \in {\cal Y} = \lbrace 0, 1, 2, \ldots, 9 \rbrace $. The training dataset has the type
# ```{figure} /images/classification/nb_assumption.svg
# ---
# height: 320px
# name: nb_assumption_fig
# align: left
# ---
# An image dataset with intrinsically spatially-correlated features. Note that nearby image pixels (features) are highly correlated by virtue of the underlying data generation process (handwriting).
# ```
# 
# Note first that the number of dimensions is quite large $ D = 28^{2} = 784 $. As the number of possible values along each dimension $ d $ is $ |{\cal X}_{d}| = 2^{8} = 256 $, there are $ |{\cal X}| = |{\cal X}_{1}| \times |{\cal X}_{2}| \times \ldots \times |{\cal X}_{784}| = 256^{784} $ possible images. This is larger than the number of atoms in the universe $ \approx 10^{82} $. On the other hand, the subset $ {\cal X}' \subset {\cal X} $ containing valid images of numerals is much smaller.
# 
# Finally, as images of handwritten numbers, nearby pixels are intrinsically correlated -- for instance, an image containing the handwriting of the number two still corresponding to the class label $ y = 2 $ after being rotated or slightly translated and scaled such that the handwritten number is still within image bounds. Thus, the conditionally independence assumption is not directly applicable to the feature space $ {\cal X} $ as defined here. That is, conditioned on a class value $ y $, the image pixels $ \lbrace x_{d} \rbrace $, $ d \in \lbrace 1, \ldots, D \rbrace $, are spatially correlated.
# ````
# 
# ### Naive Bayes classifier
# 
# ````{prf:definition} NB classifier
# :label: nb_classifier
# The NB classifier is defined as
# ```{math}
# :label: nb_estimate2
# h_{NB}({\bf x}) = \argmax_{y \in {\cal Y}} \left\lbrace P(y) \prod_{d=1}^{D} P(x_{d} \mid y) \right\rbrace,
# ```
# where $ {\bf x} = \begin{bmatrix} x_{1} & \ldots & x_{d} & \ldots & x_{D} \end{bmatrix}^{T} $.
# ````
# 
# ````{toggle}
# ```{prf:proof}
# By plugging {eq}`naive_assumption` into {eq}`nb_estimate1b` and substituting $ P^{\ast}(y) $ and $ P^{\ast}(x_{d} \mid y) $ by their frequentist approximations $ P(y) $ and $ P(x_{d} \mid y) $ given by {eq}`freq_approx1` and {eq}`freq_approx2`, respectively, we obtain the NB classifier as defined in {eq}`nb_estimate2`.
# ```
# ````
# 
# ````{prf:remark} Switching to log odds
# Note that the NB classifier as defined in {eq}`nb_estimate2` is numerically unstable since it requires the computation of the product of several conditional probabilities $ P(x_{d} \mid y) $, $ d \in \lbrace 1, \ldots, D \rbrace $, for a large dimensional feature space $ {\cal X} $. As computers use a limited number of bits to represent real numbers -- using double-precision or single-precision float-point format --, even though a small fraction of these $ D $ probabilities are small, their product might easily collapse to zero. Thus, a typical workaround to avoid numerical instability is to rewrite {eq}`nb_estimate2` as
# ```{math}
# :label: nb_estimate3
# h_{NB}({\bf x}) &=& \argmax_{y \in {\cal Y}} \left\lbrace \log \left( P(y) \prod_{d=1}^{D} P(x_{d} \mid y) \right) \right\rbrace \nonumber \\
# &\equiv& \argmax_{y \in {\cal Y}} \left\lbrace \log P(y) + \sum_{d=1}^{D} \log P(x_{d} \mid y) \right\rbrace,
# ```
# since $ \log(\cdot) $ is a monotonic increasing function. In this case, the NB classifier is selecting the class value $ y $ which maximizes the log-likelihood function with the term $ \log P(y) $ representing the prior evidence of the class value $ y $ and the terms $ \log P(x_{d} \mid y) $ representing the contribution of each observed component $ x_{d} $ given the class value $ y $. The contribution of the log function to numerical stability is two-folded: 
# * the summation in {eq}`nb_estimate3` avoids collapsing the joint likelihood function $ P( {\bf x} \mid y) \ \equiv P(x_{1}, x_{2}, \ldots, x_{D} \mid y) $ by multiplying several possible small likelihoods $ P(x_{d} \mid y) > 0 $ and, 
# * albeit the logarithm of valid probabilities are upper bounded by $ 0 $, near-zero probabilities / likelihoods are severely penalized since $ \log(0^{+}) = -\infty $.
# ````
# 
