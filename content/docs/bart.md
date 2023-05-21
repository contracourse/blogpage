---
title: "Bayesian Additive Regression Trees in R"
date: 2023-04-09T18:29:50+02:00
draft: false
katex: true
math: true
---
<span style="font-size:16px;">

## Introduction

Gradient boosting methods are commonly used in the Machine Learning
field. It's a rather straight forward process as it utilized "tree
boosting" optimization methods by combining random forest algorithms
with a learning rate. Gradient boosting algorithms are seeking to
minimize an objective function.

<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <mi>O</mi>
          <msub>
            <mi></mi>
            <mi>ij</mi>
          </msub>
  <mo>=</mo>
  <munder>
    <mrow class="MJX-TeXAtom-OP MJX-fixedlimits">
      <munder>
        <mrow>
          <munderover>
            <mo>&#x2211;<!-- ∑ --></mo>
            <mrow class="MJX-TeXAtom-ORD">
              <mi>i</mi>
              <mo>=</mo>
              <mn>1</mn>
            </mrow>
            <mi>I</mi>
          </munderover>
          <mtext>loss</mtext>
          <mo stretchy="false">(</mo>
          <msub>
            <mi>y</mi>
            <mi>i</mi>
          </msub>
          <mo>,</mo>
          <msub>
            <mrow class="MJX-TeXAtom-ORD">
              <mover>
                <mi>y</mi>
                <mo stretchy="false">&#x007E;<!-- ~ --></mo>
              </mover>
            </mrow>
            <mi>i</mi>
          </msub>
          <mo stretchy="false">)</mo>
        </mrow>
        <mo>&#x23DF;<!-- ⏟ --></mo>
      </munder>
    </mrow>
    <mrow class="MJX-TeXAtom-ORD">
      <mtext>error term</mtext>
    </mrow>
  </munder>
  <mspace width="1em" />
  <mo>+</mo>
  <munder>
    <mrow class="MJX-TeXAtom-OP MJX-fixedlimits">
      <munder>
        <mrow>
          <munderover>
            <mo>&#x2211;<!-- ∑ --></mo>
            <mrow class="MJX-TeXAtom-ORD">
              <mi>j</mi>
              <mo>=</mo>
              <mn>1</mn>
            </mrow>
            <mi>J</mi>
          </munderover>
          <mi mathvariant="normal">&#x03BB;<!-- λ --></mi>
          <mo stretchy="false">(</mo>
          <msub>
            <mi>T</mi>
            <mi>j</mi>
          </msub>
          <mo stretchy="false">)</mo>
        </mrow>
        <mo>&#x23DF;<!-- ⏟ --></mo>
      </munder>
    </mrow>
    <mrow class="MJX-TeXAtom-ORD">
      <mtext>regularization term</mtext>
    </mrow>
  </munder>
</math>

Most common machine learning algorithms are using a similar basic
objective function which is based on a ***frequentist approach***
towards statistics. The Bayesian approach treats the models in terms of
a probability distribution, instead of giving you an exact output
parameter.

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
> str(data)
Classes ‘data.table’ and 'data.frame':  264 obs. of  8 variables:
 $ T10Y3M            : num  -0.1243 0.0937 0.35 1.175 1.6882 ...
 $ EFFR              : num  6.02 5.51 5.32 4.81 4.23 ...
 $ UNRATE            : num  4.2 4.2 4.3 4.4 4.3 4.5 4.6 4.9 5 5.3 ...
 $ STLFSI4           : num  0.453 0.396 0.627 0.723 0.325 ...
 $ CPIAUCSL_PC1      : num  3.72 3.53 2.98 3.22 3.56 ...
 $ SPY               : num  90.4 81.8 77 83.8 83.3 ...
 $ 2y_expec_Inflation: num  0.0265 0.0263 0.0236 0.0258 0.0273 ...
 $ 1y_real_rate      : num  0.0313 0.0283 0.0362 0.024 0.0186 ...
```



</span>