import numpy as np
import random
import scipy.stats as ss
import pandas as pd
import sklearn.preprocessing as sp
import sklearn.decomposition as sd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.neighbors import KNeighborsClassifier

random.seed(123)


def majority_vote_fast(votes):
    mode, count = ss.mstats.mode(votes)
    return mode

def distance(p1, p2):
    return np.sqrt(np.sum(np.power(p2 - p1, 2)))

def find_nearest_neighbors(p, points, k=5):
    distances = np.zeros(points.shape[0])
    for i in range(len(distances)):
        distances[i] = distance(p, points[i])
    ind = np.argsort(distances)
    return ind[:k]

def knn_predict(p, points, outcomes, k=5):
    ind = find_nearest_neighbors(p, points, k)
    return majority_vote_fast(outcomes[ind])[0]

# ----- Exercise 1 ----- #
"""
Our first step is to import the dataset.

Taking a look at the first 5 rows of the data set, 
how many wines in those 5 rows are considered high quality?
1
"""
data = pd.read_csv("wine.csv")
# print(wine)

# ----- Exercise 2 ---- #
"""
In order to get all numeric data, we will change the color column to an is_red column.

If color == 'red', we will encode a 1 for is_red.
If color == 'white', we will encode a 0 for is_red.
Create this new column, is_red. Drop the color column as well as quality and high_quality. 
We will predict the quality of wine using the numeric data in a later exercise

Store this all numeric data in a pandas dataframe called numeric_data.

How many red wines are in the dataset?
1599
"""

cols = list(pd.read_csv("wine.csv"))
wine = pd.read_csv("wine.csv", usecols=[i for i in cols if i != 'quality' and i != 'high_quality'])
numeric_data = wine.rename(columns={'color': 'is_red'})

totalRed = 0
for i, row in numeric_data.iterrows():
    if row['is_red'] == 'red':
        numeric_data.at[i, 'is_red'] = 1
        totalRed += 1
    else:
        numeric_data.at[i, 'is_red'] = 0

# print(totalRed)

# ----- Exercise 3 ----- #
"""
-> Scale the data using the sklearn.preprocessing function scale() on numeric_data.
-> Convert this to a pandas dataframe, and store it as numeric_data.
-> Include the numeric variable names using the parameter columns = numeric_data.columns.
-> Use the sklearn.decomposition module PCA() and store it as pca.
-> Use the fit_transform() function to extract the first two principal components from the data, 
    and store them as principal_components.
    
New Dataset = principal_components
shape: (6497, 2)
"""
scaled_data = sp.scale(numeric_data)
numeric_data = pd.DataFrame(scaled_data, columns=numeric_data.columns)

pca = sd.PCA(n_components=2)
principal_components = pca.fit_transform(numeric_data)
# print(principal_components.shape)

# ----- Exercise 4 ----- #
"""
The first two principal components can be accessed using principal_components[:,0]
and principal_components[:,1]. Store these as x and y respectively, and make a 
scatter plot of these first two principal components.

Consider how well the two groups of wines are separated by the first two principal components.
"""
observation_colormap = ListedColormap(['red', 'blue'])
x = principal_components[:, 0]
y = principal_components[:, 1]

plt.title("Principal Components of Wine")
plt.scatter(x, y, alpha=0.2, c=data['high_quality'], cmap=observation_colormap, edgecolors='none')
plt.xlim(-8, 8)
plt.ylim(-8, 8)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
# plt.show()

# ----- Exercise 5 ----- #
"""
-> Create a function accuracy(predictions, outcomes) that takes two lists 
of the same size as arguments and returns a single number, which is the 
percentage of elements that are equal for the two lists.
-> Use accuracy to compare the percentage of similar elements in the x and y 
numpy arrays defined below.
-> Print your answer.
-> What is the accuracy of the x predictions on the "true" outcomes y?
"""
x = np.random.randint(0, 2, 1000)
y = np.random.randint(0, 2, 1000)
def accuracy(predictions, outcomes):
    return 100 * np.mean(predictions == outcomes)
# print(accuracy(x, y))  # gives 54.400...6 but should give 51.5

# ----- Exercise 6 ----- #
"""
Use accuracy() to calculate how many wines in the dataset are of low quality. 
Do this by using 0 as the first argument, and data["high_quality"] as the second argument.
"""
# print(accuracy(0, data['high_quality']))

# ----- Exercise 7 ----- #
"""
-> Use knn.predict(numeric_data) to predict which wines are high and low quality and 
store the result as library_predictions.
-> Use accuracy to find the accuracy of your predictions, using library_predictions 
as the first argument and data["high_quality"] as the second argument.
-> Print your answer. Is this prediction better than the simple classifier in Exercise 6?
84.20809604432816
"""
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(numeric_data, data['high_quality'])
library_predictions = knn.predict(numeric_data)
# print(accuracy(library_predictions, data['high_quality']))

# ----- Exercise 8 ----- #
"""
Fix the random generator using random.seed(123), and select 
10 rows from the dataset using random.sample(range(n_rows), 10). Store this selection as selection.

What is the 10th random row selected?
"""
n_rows = data.shape[0]
selection = random.sample(range(n_rows), 10)
# print(selection)  # 4392

# ----- Exercise 9 ----- #
"""
-> For each predictor in predictors[selection], use 
knn_predict(p, predictors[training_indices,:], outcomes[training_indices], k=5) 
to predict the quality of each wine in the prediction set, and store these predictions 
as a np.array called my_predictions. Note that knn_predict is defined as in the Case 3 videos 
(and as given at the beginning of these exercises).
-> Using the accuracy function, compare these results to the selected rows from 
the high_quality variable in data using my_predictions as the first argument and 
data.high_quality.iloc[selection] as the second argument. Store these results as percentage.
-> Print your answer.
"""

predictors = np.array(numeric_data)
training_indices = [i for i in range(len(predictors)) if i not in selection]
outcomes = np.array(data["high_quality"])

my_predictions = [knn_predict(p, predictors[training_indices,:], outcomes[training_indices], k=5) for p in predictors[selection]]
percentage = accuracy(my_predictions, data.high_quality.iloc[selection])
print(percentage)