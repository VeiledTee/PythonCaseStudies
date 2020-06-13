import pandas as pd
import numpy as np

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score

import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

# EDIT THIS CODE TO LOAD THE SAVED DF FROM THE LAST HOMEWORK
df = pd.read_csv('movies_clean.csv')

# ----- Exercise 1 ----- #
"""
-> Instantiate LinearRegression(), LogisticRegression(), RandomForestRegressor(), 
and RandomForestClassifier() objects, and assign them to linear_regression, 
logistic_regression, forest_regression, and forest_classifier, respectively.
-> For the random forests models, specify max_depth=4 and random_state=0.
"""
# Define all covariates and outcomes from `df`.
regression_target = 'revenue'
classification_target = 'profitable'
all_covariates = ['budget', 'popularity', 'runtime', 'vote_count', 'vote_average', 'Action', 'Adventure', 'Fantasy',
                  'Science Fiction', 'Crime', 'Drama', 'Thriller', 'Animation', 'Family', 'Western', 'Comedy', 'Romance',
                  'Horror', 'Mystery', 'War', 'History', 'Music', 'Documentary', 'TV Movie', 'Foreign']

regression_outcome = df[regression_target]
classification_outcome = df[classification_target]
covariates = df[all_covariates]

# Instantiate all regression models and classifiers.
linear_regression = LinearRegression()
logistic_regression = LogisticRegression()
forest_regression = RandomForestRegressor(max_depth=4, random_state=0)
forest_classifier = RandomForestClassifier(max_depth=4, random_state=0)
# ----- Exercise 2 ----- #
"""
-> Define a function called correlation with arguments estimator, X, and y. 
    The function should compute the correlation between the observed outcome y and 
    the outcome predicted by the model.
    -> To obtain predictions, the function should first use the fit method 
        of estimator and then use the predict method from the fitted object.
    -> The function should return the first argument from r2_score comparing 
        predictions and y.
-> Define a function called accuracy with the same arguments and code, 
    substituting accuracy_score for r2_score.
"""
def correlation(estimator, X, y):
    predictions = estimator.fit(X, y).predict(X)
    r2Score = r2_score(y, predictions)
    return r2Score


def accuracy(estimator, X, y):
    predictions = estimator.fit(X, y).predict(X)
    accuracyScore = accuracy_score(y, predictions)
    return accuracyScore


# ----- Exercise 3 ----- #
"""
-> Call cross_val_score using linear_regression and forest_regression as models. 
    Store the output as linear_regression_scores and forest_regression_scores, 
    respectively.
-> Set the parameters cv=10 to use 10-fold cross-validation and scoring=correlation 
    to use the correlation function defined in Exercise 2.
-> Plotting code has been provided to compare the performance of the two models. Use 
    plt.show() to plot the correlation between actual and predicted revenue for each 
    cross-validation fold using the linear and random forest regression models.
-> Consider which of the two models exhibits a better fit.
Random Forest
"""
# Determine the cross-validated correlation for linear and random forest models.
linear_regression_scores = cross_val_score(linear_regression, covariates, regression_outcome, cv=10, scoring=correlation)
forest_regression_scores = cross_val_score(forest_regression, covariates, regression_outcome, cv=10, scoring=correlation)

# Plot Results
plt.axes().set_aspect('equal', 'box')
plt.scatter(linear_regression_scores, forest_regression_scores)
plt.plot((0, 1), (0, 1), 'k-')

plt.xlim(0, 1)
plt.ylim(0, 1)
plt.xlabel("Linear Regression Score")
plt.ylabel("Forest Regression Score")

# plt.show()

# ----- Exercise 4 ----- #
"""
-> Call cross_val_score using logistic_regression and forest_classifier as models. Store the output as 
    logistic_regression_scores and forest_classification_scores, respectively.
-> Set the parameters cv=10 to use 10-fold cross-validation and scoring=accuracy to use the accuracy 
    function defined in Exercise 2.
-> Plotting code has been provided to compare the performance of the two models. Use plt.show() to plot 
    the accuracy of predicted profitability for each cross-validation fold using the logistic and random 
    forest classification models.
-> Consider which of the two models exhibits a better fit.
"""
# Determine the cross-validated accuracy for logistic and random forest models.
logistic_regression_scores = cross_val_score(logistic_regression, covariates, classification_outcome, cv=10, scoring=accuracy)
forest_classification_scores = cross_val_score(forest_classifier, covariates, classification_outcome, cv=10, scoring=accuracy)


# Plot Results
plt.axes().set_aspect('equal', 'box')
plt.scatter(logistic_regression_scores, forest_classification_scores)
plt.plot((0, 1), (0, 1), 'k-')

plt.xlim(0, 1)
plt.ylim(0, 1)
plt.xlabel("Linear Classification Score")
plt.ylabel("Forest Classification Score")

# plt.show()

# ----- Exercise 5 ----- #
"""
-> Define positive_revenue_df as the subset of movies in df with revenue greater than zero.
-> Code is provided below that creates new instances of model objects. Replace all instances of df 
with positive_revenue_df, and run the given code.
"""

positive_revenue_df = df[df.revenue > 0]

# Replace the dataframe in the following code, and run.
regression_outcome = positive_revenue_df[regression_target]
classification_outcome = positive_revenue_df[classification_target]
covariates = positive_revenue_df[all_covariates]

# Reinstantiate all regression models and classifiers.
linear_regression = LinearRegression()
logistic_regression = LogisticRegression()
forest_regression = RandomForestRegressor(max_depth=4, random_state=0)
forest_classifier = RandomForestClassifier(max_depth=4, random_state=0)
linear_regression_scores = cross_val_score(linear_regression, covariates, regression_outcome, cv=10, scoring=correlation)
forest_regression_scores = cross_val_score(forest_regression, covariates, regression_outcome, cv=10, scoring=correlation)
logistic_regression_scores = cross_val_score(logistic_regression, covariates, classification_outcome, cv=10, scoring=accuracy)
forest_classification_scores = cross_val_score(forest_classifier, covariates, classification_outcome, cv=10, scoring=accuracy)

# print(np.mean(forest_classification_scores))
# print(df.columns)

# ----- Exercise 6 ----- #
"""
Doesn't work, can't figure out why
"""

linear_regression_scores = cross_val_score(linear_regression, covariates, regression_outcome, cv=10, scoring=correlation)
forest_regression_scores = cross_val_score(forest_regression, covariates, regression_outcome, cv=10, scoring=correlation)

plt.axes().set_aspect('equal', 'box')
plt.scatter(linear_regression_scores, forest_regression_scores)
plt.plot((0, 1), (0, 1), 'k-')
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.xlabel("Linear Regression Score")
plt.ylabel("Forest Regression Score")

plt.show()

forest_regression.fit(covariates, regression_outcome)
sorted(list(zip(all_covariates, forest_regression.feature_importances_)), key=lambda tup: tup[1])
print(all_covariates[0:3])


# ----- Exercise 7 ----- #
"""
Doesn't work, can't figure out why
"""

# Determine the cross-validated accuracy for logistic and random forest models.
logistic_regression_scores = cross_val_score(logistic_regression, covariates, classification_outcome, cv=10, scoring=accuracy)
forest_classification_scores = cross_val_score(forest_classifier, covariates, classification_outcome, cv=10, scoring=accuracy)

plt.axes().set_aspect('equal', 'box')
plt.scatter(logistic_regression_scores, forest_classification_scores)
plt.plot((0, 1), (0, 1), 'k-')
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.xlabel("Linear Classification Score")
plt.ylabel("Forest Classification Score")

plt.show();

forest_classifier.fit(covariates, classification_outcome)
sorted(list(zip(all_covariates, forest_classifier.feature_importances_)), key=lambda tup: tup[1])
print(all_covariates[0:3])



