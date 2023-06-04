---
title: "Bayesian Additive Regression Trees in R"
date: 2023-04-09T18:29:50+02:00
draft: false
katex: true
math: true
---
<span style="font-size:16px;">

*Code in this example can be found on my [github](https://github.com/contracourse/Bayesian_Regression)*
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

Bayesian Additive Regression Trees *<u>(BART)</u>* utilizes similar tree
boosting methods but through a Bayesian framework where predictions are drawn
from a posterior distribution, which is a probability distribution of model
parameters given the observed data. The BART model can be expressed as follows: 

$$
\begin{equation}
Y = f(X) + E ≈ T^M_1
(X) + T^M_2
(X) + . . . + T^M_m (X) + \varepsilon, 
\hspace{0.5cm}
\varepsilon ∼ N_n
(0, σ^2I_n)
\end{equation}
$$

\\(Y\\) represent a \\(n * 1 \\)  vector of the response variable and \\(X\\) is the \\(n * p \\) matrix of
predictor columns. The \\(\epsilon\\) is the error vector. The \\(m\\) is the number of regression
trees composed from the tree structure denoted as \\(T\\), and the terminal nodes
(leaves) denoted by \\(M\\), representing together an entire tree as \\(T^M\\) with the
structure and leaf parameter.

Additive regression trees have splitting nodes that gives you smaller
prediction spaces by adding them up leaving a better
picture of the overall prediction space. You have a bunch of weak
learners to get a clear picture of the whole. You are taking each small
part of the regression tree that outputs a weak learner by itself and
adding them up to get a bigger picture. To address the overfitting
problem, the Bayesian approach penalizes against it through
regularization priors. In the same way the regularization term does for the
boosting algorithms. Data goes through each tree and only the residual flows
to the next tree, the regularization prior balances the data in order to
prevent overfitting.

BART uses non-parametric regression models which is commonly used when
relationships between variables are more complex and difficult to
express. Non-parametric regression can also be useful for exploratory data
analysis, as they can help to identify patterns and relationships that
may not be apparent with simple summary statistics.

## Approach

I’ve gathered some economic data from the FRED site (the “fredr” R-package 
allows you to access the Fred database via an API) and I’m trying to develop a
BART algorithm to forecast the unemployment rate for America.

My dataset can be
seen below. Observation starting from 2004 until the end of 2022; the UNRATE is the predictive
variable \\(\hat{y}\\) with the remaining independent variables \\(x\\).

```{r snippetName, echo=F}
> str(data_frame)
tibble [228 × 8] (S3: tbl_df/tbl/data.frame)
 $ date                                                       : Date[1:228], format: "2004-01-01" "2004-02-01" ...
 $ UNRATE             (unemployment rate)                     : num [1:228] 5.7 5.6 5.8 5.6 5.6 5.6 5.5 5.4 5.4 5.5 ...
 $ T10Y3M             (10Y-3M spread)                         : num [1:228] 3.25 3.14 2.87 3.39 3.68 3.45 3.14 2.78 2.44 2.3 ...
 $ STLFSI4            (Fed St. Louis Stress Index)            : num [1:228] -0.42 -0.494 -0.446 -0.597 -0.501 ...
 $ EFFR               (Effective Federal Funds Rate)          : num [1:228] 1 1.01 1 1.01 1 1.03 1.27 1.43 1.62 1.76 ...
 $ T5YIFR             (5y5y Forward Infl. Expectations)       : num [1:228] 2.49 2.43 2.45 2.52 2.74 2.65 2.55 2.49 2.4 2.36 ...
 $ REAINTRATREARAT1YE (1-year Real Interest Rate)             : num [1:228] -0.1377 -0.4014 -0.0643 -0.4123 -0.5069 ...
 $ CPIAUCSL           (CPI-Index)                             : num [1:228] 2.03 1.69 1.74 2.29 2.9 ...
```

I split the data into training and testing using the R package “caret”,
20% of the data will go into the test set, while the remaining 80% will go into
the training set.

```{r snippetName, echo=F}
library(caret)
y <- data$UNRATE
df <- within(data, rm(UNRATE))
set.seed(42) 
test_inds = createDataPartition(y = 1:length(y), p = 0.2, list = F)

df_test = df[test_inds, ]
y_test = y[test_inds]
df_train = df[-test_inds, ]
y_train = y[-test_inds]
```
Now running bartMachine on the training data ``bart_machine =
bartMachine(df_train, y_train)``. The default MCMC uses a burn-in rate of
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

training data size: n = 180 and p = 6 
built in 1.1 secs on 3 cores, 50 trees, 250 burn-in and 1000 post. samples

sigsq est for y beforehand: 2.141 
avg sigsq estimate after burn-in: 0.06986 

in-sample statistics:
 L1 = 17.18 
 L2 = 2.65 
 rmse = 0.12 
 Pseudo-Rsq = 0.9967
p-val for shapiro-wilk test of normality of residuals: 0.81473 
p-val for zero-mean noise: 0.91896 
```

We can use the “rmse_by_num_trees” function to find the optimum number of trees
for the model. I’ve given it a sequence from 15 to 75 trees by 5 increments with
3 number of replicant trees. The RMSE tree chart is used in order to illustrate
the out-of-sample predictive capacity of our model. With additional
hyperparameter optimization we can build a better bartmachine model in the
future. <br>

![RMSE
Tree-Plot](https://raw.githubusercontent.com/contracourse/blogpage/1bf4db9d5b37636a0c5e4e1001ce7d1fb206fc2d/static/images/rmse_by_num_trees.svg)

As you can see it shows us the path of the trees with its respective RMSE. Then
we can use the trees with the minimum RMSE and run the `bartmachine` again. The
tree looks pretty static, an increase in the number of trees did not particularly
perform better. <br>
``bart_machine <- bartMachine(df_train, y_train, num_trees=20)``

Using the “plot_convergence_diagnostics” function we can see how the MCMC
performs. Overall, the tree nodes perform relatively constant. On the top left,
the Siqsq estimate converges after ~200 interactions inside the interval.
The three subsequent plots separated by gray lines are the post-burn-in
iterations from each of the three computing cores employed during the model.

![Plot-Diagnostics](https://raw.githubusercontent.com/contracourse/blogpage/d8166cedb681f34c95e01273ca188e3694ed9d93/static/images/plot_convergence_diagnostics.svg)

Next up, the “check_bart_error_assumptions” chart show us the error normality
distribution using QQ-plots. We can see the residuals are normally distributed,
no need of any adjustment. 

![QQ-Plot](https://raw.githubusercontent.com/contracourse/blogpage/d8166cedb681f34c95e01273ca188e3694ed9d93/static/images/check_bart_error_assumptions.svg)

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
range of each data point which is about 85% accurate given a 90% credible interval.

The Prediction intervals are drawn from the posterior distribution of the MCMC
process described earlier, they are wider than the Credible intervals since they
reflecting the uncertainty of the error term. Prediction intervals tells us
about the precision of our individual predictions, a Credible interval gives us
information about the likely range of true parameter values. As you can see
below, the posterior of each prediction is getting larger as the unemployment
rate increases. This reflects the uncertainty around the prediction value on the
independent variable.  If the distribution is wide or has high variance, then
the posterior for that prediction will be larger. This is common in models with
high levels of noise or when the data points are widely spread out. Therefore,
it is important to consider the overall distribution of the target values and
the model's accuracy when interpreting the size of the posterior. <br> I've
chosen a CI of 90% since the posterior distributions are getting larger with a
higher unemployment rates, reflecting the high degree of uncertainty around
these values. A lower CI results in less predictive values that are being
captured by the model and resulting in a lower boundary or limit of the
prediction range.

![plot-y vs y-hat](https://raw.githubusercontent.com/contracourse/blogpage/985cdbc3d4c208bb1341c45f020214089e3eab1a/static/images/plot_y_vs_yhat_2.svg)

Lastly, we can calculate some metrics to compare our bartmachine precitions with the testset.
I've inserted the out-of-sample prediction into a dataframe and then calculated the ratios below. <br>
``y_pred <- predict(bart_machine, df_test)``

```
summary(lm(y_test~y_pred))$r.squared
sqrt(mean((y_test - y_pred)^2))
cor.test(y_test, y_pred, method=c("pearson"))   # Pearson R-squared  

[1] 0.9364641 # R-squared 
[1] 0.5188173 # RMSE

        Pearson's product-moment correlation

data:  y_test and y_pred
t = 26.038, df = 46, p-value < 2.2e-16
alternative hypothesis: true correlation is not equal to 0
95 percent confidence interval:
 0.9428118 0.9818702
sample estimates:
      cor 
0.9677108 
```


### Additional Diagnostic Charts 
In addition, bartmachine allows us to see the same kind of variable importance
plot as xgboost. Running the “investigate_var_importance” function to see which
variable has the most significant influence on the model's predictions.
![var_importance](https://raw.githubusercontent.com/contracourse/blogpage/2ca75f5a98ad698ad2e4cef0b281ba7b3179cca7/static/images/importance%20plot.svg)

Then we can choose the most important variable (in this case the EFFR-rate) and
look at the partial dependence plot. The PD-Plot shows us how the target variable's
predicted values change as the selected input variable varies. The
PD is plotted in black and a default 95% credible intervals plotted in blue for
the other  variables in the dataset. Points plotted are at the 5%ile, 10%ile,
20%ile, . . . , 60%ile and 75%ile of the values of the predictor. We can see as
the EFFR increases the partial effect decreases. Which would be a negative
relationship between the input variable and the predictor variable. In other
words, as the value of X increases, the predicted value decreases. This
indicates that the increased Federal Funds rate is positively associated with
higher unemployment. <br> *Note: It is important to note that interpreting partial
dependence plots requires considering the specific context of the data and the
model.*


![pd-plot](https://raw.githubusercontent.com/contracourse/blogpage/2ca75f5a98ad698ad2e4cef0b281ba7b3179cca7/static/images/pd_plot.svg)


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