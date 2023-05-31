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
back-fitting technique. Monte Carlo Markov chain (MCMC) is commonly used in
Bayesian statistics to approximate complex distributions in order to estimate
the posterior distribution of model parameters. For further details see
[here](https://towardsdatascience.com/a-zero-math-introduction-to-markov-chain-monte-carlo-methods-dcba889e0c50)

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

In this post I will use the R Package “bartMachine” to demonstrate the
effectiveness of BART. bartMachine provides some interesting diagnostic features
which I will describe later. I have gathered monthly economic data from FRED in
order to forecast the SPY. I’m trying to develop a model for predicting the SPY
based on some underlying economic data. My dataset can be
seen as below starting from 2001 till end of 2022; the SPY is the predictive
variable \\(\hat{y}\\) with the remaining independent variables \\(x\\).

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

I split the data into training and testing using the R package “caret”,
20% of the data will go into the test set, while the remaining 80% will go into
the training set.

```{r snippetName, echo=F}
library(caret)
y <- data$SPY
df <- within(data, rm(SPY))
set.seed(42) 
test_inds = createDataPartition(y = 1:length(y), p = 0.2, list = F)

df_test = df[test_inds, ]
y_test = y[test_inds]
df_train = df[-test_inds, ]
y_train = y[-test_inds]
```
Now running bartMachine on the training data ``bart_machine =
bartMachine(df_train, y_train)``. The default settings uses a burn-in rate of
250 and 1000 iteration with 50 trees. All of those parameters can be specified
manually. <br>
BART uses L1 & L2 regularization to reduce overfitting, introduce penalties and
reduce complexity especially with high dimensional data. The Pseudo-Rsq is for
non-linear models, it has the same interpretability as a normal R-squared. 

The p-value of the shapiro-wilk test tells us about
the data distribution. If the p-value is less than or equal to the significance
level (usually 0.05), then we reject the null hypothesis and conclude that the
data is not normally distributed.

```
> summary(bart_machine)
bartMachine v1.3.3.1 for regression

training data size: n = 208 and p = 7 
built in 0.9 secs on 3 cores, 50 trees, 250 burn-in and 1000 post. samples

sigsq est for y beforehand: 2046.61 
avg sigsq estimate after burn-in: 78.62889 

in-sample statistics:
 L1 = 666.8 
 L2 = 3483.97 
 rmse = 4.09 
 Pseudo-Rsq = 0.9984
p-val for shapiro-wilk test of normality of residuals: 0.77388 
p-val for zero-mean noise: 0.97604
```

We can use the “rmse_by_num_trees” function to find the optimum number of trees for the model.
I’ve given it a sequence from 15 to 75 trees by 5 increments with 3 number of
replicant trees. The RMSE tree chart is used in order to illustrate the predictive capacity of
our model. With additional hyperparameter optimization we can build a better
bartmachine model in the future.
<br>
```
rmse_by_num_trees(bart_machine, 
                  tree_list=c(seq(15, 75, by=5)),
                  num_replicates=3)
```

![RMSE
Tree-Plot](https://raw.githubusercontent.com/contracourse/blogpage/3fa4f00bccebc3d6a3ae39b57fb4db12bdbb24c9/static/images/rmse_by_num_trees.svg)

As you can see it shows us the path of the trees with its respective RMSE. Then
we can use the trees with the minimum RMSE and run the `bartmachine` again. <br>
``bart_machine <- bartMachine(df_train, y_train, num_trees=35)``

Using the “plot_convergence_diagnostics” function we can see how the MCMC
performs. Overall, the tree nodes perform relatively constant. On the top left,
the Siqsq estimate converges after ~200 interactions inside the interval.
The three subsequent plots separated by gray lines are the post-burn-in
iterations from each of the three computing cores employed during model.

![Plot-Diagnostics](https://raw.githubusercontent.com/contracourse/blogpage/3fa4f00bccebc3d6a3ae39b57fb4db12bdbb24c9/static/images/plot_convergence_diagnostics.svg)

Next up, the “check_bart_error_assumptions” show us the error normality
distribution using QQ-plots. We can see the residuals are normally distributed,
no need of any adjustment. 

![QQ-Plot](https://raw.githubusercontent.com/contracourse/blogpage/16e435c51f0931e54363c456a474fb7952860670/static/images/check_bart_error_assumptions.svg)

Lastly, we will see how well our model performs in-sample and out-of-sample.

Bayesian statistics uses Credible intervals instead of Confidence intervals.
Credible intervals provide a range of values where we can be certain that the
true parameter lies within, given the available data and the assumptions of the
model. The width of a credible interval reflects the amount of uncertainty in
the prediction, with wider intervals indicating higher uncertainty. We can
manually adjust the interval (default is 95%), a low interval means a wider
grey bar of the uncertainty in the predictor. The out of sample chart is more
important since it gives us an impression how well the model performs on the
testing data. For the out-of-sample chart we can now provide a predictive
range of each data point which is about 96% accurate given a 95% preditive interval.

The Prediction intervals are drawn from the posterior distribution of the MCMC
process described earlier, they are wider than the Credible intervals since they
reflecting the uncertainty of the error term. Prediction interval tells us about
the precision of our individual predictions, a Credible interval gives us
information about the likely range of true parameter values. 

![plot-y vs y-hat](https://raw.githubusercontent.com/contracourse/blogpage/16e435c51f0931e54363c456a474fb7952860670/static/images/plot_y_vs_yhat_2.svg)

## Conclusion

Overall, BART is an interesting algorithm with some unique capabilities, but
their suitability will depend on the complexity of the dataset and the task at
hand. Sometimes a Bayesian approach is preferred since it does not find the
single best value, rather a range of possible values determined by the posterior
distribution. <br>

BART can be compared to gradient algorithms like XGBoost or Catboost. BART may
be more appropriate when dealing with complex nonlinear relationships, while
XGBoost may be better suited for simpler problems where speed and scalability
are important.



<!-- ```
rmse <- function(x, y) sqrt(mean((x - y)^2))
rsq <- function(x, y) summary(lm(y~x))$r.squared
y_pred <- predict(bart_machine, df_test)
paste('r2:', rsq(y_test, y_pred)) # the R-squared y-test fit with predicted 
paste('rmse:', rmse(y_test, y_pred))
cor.test(y_test, y_pred, method=c("pearson"))
``` -->
<!-- Output
```
[1] "r2: 0.938910704998679"
[1] "rmse: 27.8567148747163"

        Pearson's product-moment correlation

data:  y_test and y_pred
t = 28.809, df = 54, p-value < 2.2e-16
alternative hypothesis: true correlation is not equal to 0
95 percent confidence interval:
 0.9474239 0.9817738
sample estimates:
     cor 
0.968974
``` -->

<h1 id='references'>References</h1>

Coqueret, G., & Guida, T. (2022, October 18). *Machine Learning for
Factor Investing*. http://www.mlfactor.com/bayes.html

Kapelner, A., & Bleich, J. (2016). **bartMachine**: Machine Learning
with Bayesian Additive Regression Trees. *Journal of Statistical
Software*, *70*(4). https://doi.org/10.18637/jss.v070.i04

Koehrsen, W. (2018, April 14). *Introduction to Bayesian Linear
Regression*.
https://towardsdatascience.com/introduction-to-bayesian-linear-regression-e66e60791ea7

Mamun, O. (2021, May 3). *A Primer to Bayesian Additive Regression Tree
with R*.
https://towardsdatascience.com/a-primer-to-bayesian-additive-regression-tree-with-r-b9d0dbf704d

</span>