import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
import statsmodels.api as sm
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
np.random.seed(1)

# -- 5.1.1 Intro to Statistical Learning
"""
Supervised Learning - Give algorithm inputs and outputs
Unsupervised Learning - Algorithm only given inputs, learn relationships and structures

Bunch of stuff about regression vs Classification problems (kNN is classification)

What is the difference between regression and classification?
Regression results in continuous outputs, whereas classification results in categorical outputs.

What is the difference between least squares loss and  0−1  loss?
Least squares loss is used to estimate the expected value of outputs, whereas  0−1  loss is used to
    estimate the probability of outputs.
"""

# -- 5.1.2 Generating Example Regression Data

n = 100
beta0 = 5
beta1 = 2
x = 10 * ss.uniform.rvs(size=n)
y = beta0 + beta1 * x + ss.norm.rvs(loc=0, scale=1, size=n)
"""
# plt.figure()
# plt.plot(x, y, "o", ms=5)
# xx = np.array([0, 10])
# plt.plot(xx, beta0 + beta1 * xx)
# plt.xlabel('x')
# plt.ylabel('y')
# plt.show()
"""

# -- 5.1.3 Simple Linear Regression
"""
Hats on Variables: indicate that these are parameter estimates, 
    meaning that their parameter values that have been estimated using data
Residual: Denoted as 'e' ->  e_i = y_i - y(hat)_i  [e sub i here is the difference between the i-th
                                                    observed response value and the i-th response 
                                                    value predicted by the model]
Residual Sum of Squares (RSS): RSS = e1^2 + e2^2 + ... + en^2
Least squares estimates of beta 0 and beta 1: minimize the RSS criterion

What is the difference between  Y  (capital letter) and  y  (lowercase letter)?
Y  is a random variable, whereas  y  is a particular value.

def compute_rss(y_estimate, y):
  return sum(np.power(y-y_estimate, 2))
def estimate_y(x, b_0, b_1):
  return b_0 + b_1 * x
rss = compute_rss(estimate_y(x, beta0, beta1), y)
# print(rss)
"""
# -- 5.1.4 Least Squares Estimation in Code
"""
In the following, we're going to assume that we know the true value of beta 0,
and our goal is to estimate the value of beta 1, the slope of the line, from data.
"""
rss = []
slopes = np.arange(-10, 15, 0.01)
for slope in slopes:
    rss.append(np.sum((y -beta0 - (slope * x))**2))

indMin = np.argmin(rss)
# print("Estimate for the slope:", slopes[indMin])  # 1.9999999999997442

# Plot Figure
# plt.figure()
# plt.plot(slopes, rss)
# plt.xlabel("Slopes")
# plt.ylabel("RSS")
# plt.show()

# -- 5.1.5 Simple Linear Regression in Code
mod = sm.OLS(y, x)
est = mod.fit()
# print(est.summary())
"""
                                 OLS Regression Results                                
=======================================================================================
Dep. Variable:                      y   R-squared (uncentered):                   0.968
Model:                            OLS   Adj. R-squared (uncentered):              0.967
Method:                 Least Squares   F-statistic:                              2974.
Date:                Sun, 07 Jun 2020   Prob (F-statistic):                    1.14e-75
Time:                        21:22:28   Log-Likelihood:                         -246.89
No. Observations:                 100   AIC:                                      495.8
Df Residuals:                      99   BIC:                                      498.4
Df Model:                           1                                                  
Covariance Type:            nonrobust                                                  
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
x1             2.7569      0.051     54.538      0.000       2.657       2.857
==============================================================================
Omnibus:                        7.901   Durbin-Watson:                   1.579
Prob(Omnibus):                  0.019   Jarque-Bera (JB):                3.386
Skew:                           0.139   Prob(JB):                        0.184
Kurtosis:                       2.143   Cond. No.                         1.00
==============================================================================

Fitted a model with a single variable, x1. And the estimated coefficient is 2.7. 
That seems a little bit high, given that we know that the true value should be about 2.0
Turns out we fitted a slightly different model, one with a slop but no intercept, forcing the line through 0
"""
X = sm.add_constant(x)  # add a constant to the previous model
mod = sm.OLS(y, X)
est = mod.fit()
# print(est.summary())  # this model is not required to go through the origin, reflected in the slope
"""
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                      y   R-squared:                       0.977
Model:                            OLS   Adj. R-squared:                  0.977
Method:                 Least Squares   F-statistic:                     4115.
Date:                Sun, 07 Jun 2020   Prob (F-statistic):           7.47e-82
Time:                        21:28:33   Log-Likelihood:                -130.72
No. Observations:                 100   AIC:                             265.4
Df Residuals:                      98   BIC:                             270.7
Df Model:                           1                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
const          5.2370      0.174     30.041      0.000       4.891       5.583
x1             1.9685      0.031     64.151      0.000       1.908       2.029
==============================================================================
Omnibus:                        2.308   Durbin-Watson:                   2.206
Prob(Omnibus):                  0.315   Jarque-Bera (JB):                1.753
Skew:                          -0.189   Prob(JB):                        0.416
Kurtosis:                       3.528   Cond. No.                         11.2
==============================================================================
Intercept 5.2 is the value of the outcome y when all predictors are set to 0 (only x1 here)
Increase in x1 by 1 -> 1.9685 increase in y

If the true intercept were negative but the regression model did not include an 
    intercept term, what would that imply for the estimated slope?
The estimated slope would likely be lower than the true slope.

What does an estimated intercept term correspond to?
The estimated outcome when the input is set to zero

You could create several datasets using different seed values and estimate the slope from each. 
    These parameters will follow some distribution. What is the name used for this distribution?
The sampling distribution of the parameter estimates

If the  R^2  value is high, this indicates:
a good fit: the residual sum of squares is low compared to the total sum of squares.
"""

# -- 5.1.6 Multiple Linear Regression
"""
In multiple linear regression, the goal is to predict a quantitative or a scalar valued
response, Y, on the basis of several predictor variables

y^=β^0+x1β^1+x2β^2 
β1 and β2 have been estimated from data. If we assume that β^1=1, and β^2=3.

What is the interpretation of β^1?
The change in the predicted outcome if  x1  is increased by 1, holding  x2  constant.

For a given expected output prediction  y^ , what would be the expected change in the 
    prediction value if you increased  x1  by 1, and decreased  x2  by 3?
-8
"""

# -- 5.1.7 scikit-learn for Linear Regression
n = 500
beta0 = 5
beta1 = 2
beta2 = -1
x1 = 10 * ss.uniform.rvs(size=n)
x2 = 10 * ss.uniform.rvs(size=n)
y = beta0 + (beta1 * x1) + (beta2 * x2) + ss.norm.rvs(loc=0, scale=1, size=n)

X = np.stack([x1, x2], axis=1)

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(X[:, 0], X[:, 1], y, c=y)
# ax.set_xlabel('$x1$')
# ax.set_ylabel('$x2$')
# ax.set_zlabel('$y$')
# plt.show()

lm = LinearRegression(fit_intercept=True)
lm.fit(X, y)
# print(lm.intercept_)  # 5.020202136041929 <- estimated beta0 value
# print(lm.coef_)  # [ 2.0079527  -0.99659908] <- can be indexed
# lm.coef_ [0] is beta1, lm.coef_ [1] is beta2
X0 = np.array([2, 4])  # x1 =2, x2 = 4
# print(lm.predict(X0))  # Gives error: "ValueError: Expected 2D array, got 1D array instead:"
# print(lm.predict(X0.reshape(1, -1)))  # [5.04971121]
# print(lm.score(X, y))  # 0.9770545503305776

# -- 5.1.8 Assessing Model Accuracy
"""
In regression, most common way to compare predictions is Mean Squared Error (MSE)
MSE given by averaging over n data points
    Our indexing variable is i, which goes from 1 to n.
    We take yi, which is the observed outcome.
    From that, we subtract our prediction at the corresponding value xi,
    and we square the difference
Divide into training and test data

Overfit: Model starts to follow the noise in the data too closely
    Extreme -> Model can memorize the data points rather learn the structure of the data
Underfit: Model is not sufficiently flexible to learn the structure in the data

"""

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.5, random_state=1)
lm = LinearRegression(fit_intercept=True)
lm.fit(X_train, y_train)
# print(lm.score(X_test, y_test))  # 0.9783861176911517
