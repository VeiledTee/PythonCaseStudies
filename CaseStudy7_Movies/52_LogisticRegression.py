import matplotlib
import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

np.random.seed(1)

t = 1
stanDev = 1
n = 50

def genData(num, h, sd1, sd2):
    x_1 = ss.norm.rvs(-h, sd1, num)
    y_1 = ss.norm.rvs(0, sd1, num)
    x_2 = ss.norm.rvs(h, sd2, num)
    y_2 = ss.norm.rvs(0, sd2, num)
    return x_1, y_1, x_2, y_2

# generate data
# (x1, y1, x2, y2) = genData(50, 1, 1, 1.5)
(x1, y1, x2, y2) = genData(1000, 1, 2, 2.5)

# plot generated data
def plotData(x_1, y_1, x_2, y_2):
    plt.figure()
    plt.plot(x_1, y_1, "o", ms=2)
    plt.plot(x_2, y_2, "o", ms=2)
    plt.xlabel("$X_1$")
    plt.ylabel("$X_2$")

# plotData(x1, y1, x2, y2)
# plt.show()

# -- 5.2.2 Logistic Regression
"""
p(x) = p(y = 1 | x)
logistic regression is a linear model that models probabilities on a non-linear scale

What is one of the problems with using linear regression to predict probabilities?
Linear regression may predict values outside of the interval between 0 and 1.
"""
def prob_to_odds(p):
    if p <= 0 or p >= 1:
        print("Probabilities must be between 0 and 1.")
    return p / (1-p)

# -- 5.2.3 Logistic Regression in Code
n = 1000
clf = LogisticRegression()
# print(np.vstack((x1, y1)).T.shape)  # (1000, 2) 1000 rows, 2 columns
X = np.vstack((np.vstack((x1, y1)).T, np.vstack((x2, y2)).T))  # generate X matrix
# print(X.shape)  # (2000, 2)
y = np.hstack((np.repeat(1, n), np.repeat(2, n)))  # vector y
# print(y.shape)  # (2000, ) 2000 rows

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.5, random_state=1)
# shape of X objects -> (1000, 2)
# shape of y objects -> (1000, )

clf.fit(X_train, y_train)  # creates LogisticRegression object, lots of parameters, need to google them all
# print(clf.score(X_test, y_test))  # 0.665

# Compute estimated class probabilities
# print(clf.predict_proba(np.array([-2, 0]).reshape(1, -1)))  # [[0.65760323 0.34239677]]
# print(clf.predict(np.array([-2, 0]).reshape(1, -1)))  # [1]  predicts the point (-2, 0) is in class 1

# -- 5.2.4 Computing Predictive Probabilities Across the Grid
def plot_probs(axis, classification, class_no):
    xx1, xx2 = np.meshgrid(np.arange(-5, 5, 0.1), np.arange(-5, 5, 0.1))
    probs = classification.predict_proba(np.stack((xx1.ravel(), xx2.ravel()), axis=1))
    Z = probs[:, class_no]
    Z = Z.reshape(xx1.shape)
    CS = axis.contourf(xx1, xx2, Z)
    cbar = plt.colorbar(CS)
    plt.xlabel("$X_1$")
    plt.ylabel("$X_2$")



plt.figure(figsize=(5,8))
ax = plt.subplot(211)
plot_probs(ax, clf, 0)
plt.title("Pred. prob for class 1")
ax = plt.subplot(212)
plot_probs(ax, clf, 1)
plt.title("Pred. prob for class 2");
plt.show()








