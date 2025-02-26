#!/usr/bin/env python
# coding: utf-8

# # Optimization
# 
# In data mining and machine learning, optimization plays a major role at the heart of almost every method. We could say that optimization is what actually defines the "learning" of machines. Whether we are training a simple linear regression model or a deep neural network, the goal remains the same: finding the best possible solution to a given problem within the constraints of the available data and model. This process of finding the "best" solution is what we refer to as optimization.
# 
# At its core, optimization seeks to minimize (or sometimes maximize) an objective function, such as a loss function, which quantifies how well the model is performing. Some loss functions are easier to optimize than others. For example, if there are many local minima, or saddle points of the loss function, finding the global optimum is not achievable in practical scenarios (often also not really desirable). The optimization determines then what kind of local minimum we are going to get. Hence, the power of a machine learning model largely depends on the power of the optimization method in combination with the suitability of the loss.  
# 
# 
# ## Structure of the Chapter
# 
# In this chapter, we discuss the basics of the optimization techniques used in this course. It's good to read this chapter now and to get familiar with the existing methods. Later, you might want to revisit this chapter, when we  apply the presented optimization methods to machine learning. Often, the optimization methods still need some alterations to the general scheme, to make the optimization work well in practice. We will wor out these details in the corresponding chapers then.     
# 
# Now, we discuss finding minima by the analytical solutions of the First and Secondary order necessary condition, as well as numeric optimization methods like coordinate and gradient descent. We discuss properties of optimization objectives, such as convexity and the existence of constraints, and how these properties influence the set of potential minimizers. 
# 
# 
# ## Recommended Literature:     
# If you want to read up further on this topic, the following material is recommended.       
# **Linear Algebra and Optimization for Machine Learning by Charu C. Aggarwal}**     
# Sections 4.1-4.3 build up nicely the aspects of optimization from the one-dimensional case (univariate optimizattion) to higher dimensions (multivariate optimization). Section 4.6 gives an overview over computing gradients subject to vectors and matrices.

# In[ ]:




