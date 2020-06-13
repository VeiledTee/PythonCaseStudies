from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier

# -- 5.3.1 Tree-Based Methods for Regression and Classification
"""
The goal of a tree-based method is typically to split up the predictor or feature space such that:
    data within each region are as similar as possible.

For classification, how does a decision tree make a prediction for a new data point?
    It returns the mode of the outcomes of the training data points in the predictor space where
    the new data point falls.
For regression, how does a decision tree make a prediction for a new data point?
    It returns the mean of the outcomes of the training data points in the predictor space where
    the new data point falls.
"""

# -- 5.3.2 Random Forest Predictions
"""
How is randomness at the data level introduced?
    Each tree gets a bootstrapped random sample of training data.
How is randomness at the predictor level introduced?
    Each split only uses a subset of predictors.
In a classification setting, how does a random forest make predictions?
    Each tree makes a prediction and the mode of these predictions is the prediction of the forest.
In a regression setting, how does a random forest make predictions?
    Each tree makes a prediction and the mean of these predictions is the prediction of the forest.
"""







































































































