---
title: "Bayesian Additive Regression Trees in R"
date: 2023-04-09T18:29:50+02:00
draft: false
katex: true
math: true
---
<span style="font-size:15px;">

## Introduction

Gradient boosting methods are commonly used in the Machine Learning
field. It's a rather straight forward process as it utilized "tree
boosting" optimization methods by combining random forest algorithms
with a learning rate. Gradient boosting algorithms are seeking to
minimize an objective function.

$$E[y-\tilde{y}]=\underbrace{\sum_{i=1}^I \text{loss}(y_i,\tilde{y}_i)}_{\text{error term}}\underbrace{\sum_{j=1}^J\lambda(T_j)}_{\text{regularization term}}$$

$$E[y-\tilde{y}] = \sum_{i=1}^I \text{loss}(y_i,\tilde{y}_i)_{\text{error term}}+\left(x^{\smash{2}}\right)$$

$$\underbrace{\sum_{j=1}^J\lambda(T_j)}_{\text{regularization term}}$$

Most common machine learning algorithms are using a similar basic
objective function which is based on a ***frequentist approach***
towards statistics. The Bayesian approach treats the models in terms of
a probability distribution, instead of giving you an exact output
parameter.

$$
\begin{align}
  \tag{1.1}
E [y - \tilde{y}] = \underbrace{\sum_{i=1}^I \text{loss} (y_i, \tilde{y}_i)} _ { \text{error term}}   & + \quad \sum_{j=1}^J\underbrace{\lambda(T_j)} _ \text{regularization term}
\end{align}
$$

$$
\begin{align}
E [y - \tilde{y}] = \underbrace{\sum_{i=1}^I \text{loss} (y_i, \tilde{y}_i)} _ { \text{error term}}   & + \quad y_t &= g_t(x_t,v_t)
\end{align}
$$

We can use a Bayesian approach to determine the model parameters. This
approach allows us to incorporate our prior beliefs about the shape of
trees and the overall ensemble structure into the model. After that, the
algorithm will update the priors based on the data using an MCMC
back-fitting technique.

**<u>BART</u>** utilizes similar tree boosting methods but
through a Bayesian framework where predictions are drawn from a
posterior distribution, which is a probability distribution of model
parameters given the observed data.

Additive regression trees have splitting nodes that gives you smaller
prediction spaces and by adding them up it gives you get a better
picture of the overall prediction space. You have a bunch of weak
learners to get a clear picture of the whole. You are taking each small
part of the regression tree that outputs a weak learner by itself and
adding them up to get a bigger picture. To address the overfitting
problem, the Bayesian approach here penalize against it through
regularization prior, in the same the regularization term does in the
boosting algorithm. Data goes through each tree, only the residual flows
to the next tree, the regularization prior balances the data in order to
prevent overfitting. 

test

BART uses non-parametric regression models which is commonly used when
relationships between variables are more complex and difficult to
express. Non-parametric can also be useful for exploratory data
analysis, as they can help to identify patterns and relationships that
may not be apparent with simple summary statistics.

## Approach

In this post I will use the R Package "***bartMachine"*** demonstrate
the effectiveness of BART. ***bartMachine*** provides some interesting
diagnostic features which I will describe later. I will have drawn some
sample data from the FRED database and I'm trying to develop a model for
predicting the SPY based on some underlying economic data. Finally, I
will compare the BART algorithm with the most common gradient boosting
algorithm.

```{r snippetName, echo=F}
plot(df$x, df$y)
```



</span>