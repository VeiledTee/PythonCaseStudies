import numpy as np
import scipy.stats as ss
import random
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier


# -- 3.3.1 Intro to kNN
"""
# Statistical learning: collection of mathematical and computational data
# in supervised learning, the goal is to estimate/predict an output based on one or more inputs
# inputs can also be: predictors, independent variables, features, and variables being called common
# outputs are often called: response or dependent variables
## if response is quantitative (a number), those problems are called REGRESSION PROBLEMS
## if response is qualitative (yes/no or blue/green), those problems are called CLASSIFICATION PROBLEMS
# k - nearest neighbors is a classifier!!
"""

# -- 3.3.2 Find Distance Between 2 Points
# d^2 = (|x1-x2|)^2 + (|y1-y2|)^2
"""
p1 = np.array([1, 1])
p2 = np.array([4, 4])
print(p2 - p1)  # [3 3]
print(np.power(p2 - p1, 2))  # [9 9]
print(np.sqrt(np.sum(np.power(p2 - p1, 2))))  # 4.242640687119285
# turn into function
"""
def distance(point1, point2):
    """
    Finds the distance between point1 and point2. REQUIRES NUMPY PACKAGE
    :param point1: a point on the cartesian plane
    :param point2: a point on the cartesian plane
    :return: the distance between point1 and point2
    """
    return np.sqrt(np.sum(np.power(point2 - point1, 2)))

# -- 3.3.3 Majority Vote
"""
Note that while this method is commonly called "majority vote," 
what is actually determined is the plurality vote, because the most common vote 
does not need to represent a majority of votes. We have used the standard naming
convention of majority vote here.
"""
"""
Draft 1
def majorityVote(votes):
    voteCounts = dict()
    for vote in votes:
        if vote in voteCounts:
            voteCounts[vote] += 1
        else:
            voteCounts[vote] = 1
    return voteCounts

voting = [1, 2, 3, 1, 2, 3, 1, 2, 2, 3, 1, 1, 2, 3, 2]
# print(majorityVote(voting))  # {1: 5, 2: 6, 3: 4}
voteCount = majorityVote(voting)

# both lines of code yield same result
max(voteCount)  # 3
max(voteCount.keys())  # 3

# find largest value
maxCount = max(voteCount.values())  # 6
"""
"""
Draft 1 of selecting most common key
function is below
winners = []
for vote, count in voteCount.items():
    if count == maxCount:
        winners.append(vote)

print(winners)  # 2
"""
def majorityVote(votes):
    """
    Selects 1 winner based on input (randomly chosen in the event of a tie)
    :param votes: a list of votes
    :return: one of the votes that is the most common (even in the even of a tie)
    """
    voteCounts = dict()
    for vote in votes:
        if vote in voteCounts:
            voteCounts[vote] += 1
        else:
            voteCounts[vote] = 1
    winners = []
    maxCount = max(voteCounts.values())
    for vote, count in voteCounts.items():
        if count == maxCount:
            winners.append(vote)
    return random.choice(winners)
"""
voting = [1, 2, 3, 3, 3, 3, 1, 2, 3, 1, 1, 1]
print(majorityVote(voting))  # either 1 or 3

# could use MODE
"""
def majorityVoteShort(votes):
    """
    Selects 1 winner based on input (randomly chosen in the event of a tie)
    :param votes: a list of votes
    :return: most common element in 'votes'
    """
    mode, count = ss.mstats.mode(votes)
    return random.choice(mode)
"""
print(majorityVoteShort(voting))  # always returns 1
# We are going to stick with majorityVote() because
# it chooses elements at random, not just the smallest
"""

# -- 3.3.4 Finding Nearest Neighbors
"""
loop over all points
    computer distance between point p and all other points
sort distances and return those k points that are nearest to point p
"""
"""
points = np.array([[1, 1], [1, 2], [1, 3], [2, 1], [2, 2], [2, 3], [3, 1], [3, 2], [3, 3]])
pointP = np.array([2.5, 2])
"""
"""
distances = np.zeros(points.shape[0])
for i in range(len(distances)):
    distances[i] = distance(pointP, points[i])

# argsort returns indices that would sort the given array
ind = np.argsort(distances)
print(ind)  # [4 7 3 5 6 8 1 0 2]
print(distances[ind])  # [0.5 0.5 1.11803399 1.11803399 1.11803399 1.11803399 1.5 1.80277564 1.80277564]
# notice how the first 2 values of distances[ind] are the lowest
plt.plot(points[:, 0], points[:, 1], "ro")
plt.plot(pointP[0], pointP[1], "bo")
plt.axis([0.5, 3.5, 0.5, 3.5])
# plt.show()
"""
def findNearestNeighbours(p, pArrray, k=5):
    """
    find k nearest neighbours of point p and return their indices
    :param p: a point
    :param pArrray: a NumPy array of points
    :param k: how many nearest neighbours you want to find
    :return: a NumPy array is sorted indices
    """
    distanceArray = np.zeros(pArrray.shape[0])
    for i in range(len(distanceArray)):
        distanceArray[i] = distance(p, pArrray[i])
    index = np.argsort(distanceArray)
    return index[:k]
"""
ind = findNearestNeighbours(pointP, points, 3)
print(points[ind])  # with k = 2; [[2 2] [3 2]]
print(points[ind])  # with k = 3; [[2 2] [3 2] [2 1]]
"""
def kNNPredict(p, pArray, outcomes, k=5):
    # find kNN
    kNN = findNearestNeighbours(p, pArray, k)
    # predict class of p based on majority vote
    return majorityVote(outcomes[kNN])
"""
outcome = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1])
print(kNNPredict(np.array([2.5, 2.7]), points, outcome, k=2))  # 1 (belongs to class 1)
print(kNNPredict(np.array([1.0, 2.7]), points, outcome, k=2))  # 0 (belongs to class 0)
"""

# -- 3.3.5 Generating Synthetic Data
"""
n = 5

# Base testing & understanding
print(ss.norm(0, 1).rvs((5, 2)))
print(ss.norm(1, 1).rvs((5, 2)))
# concatenate along the rows (axis=0) of the above arrays with NumPy

np.concatenate((ss.norm(0, 1).rvs((5, 2)), ss.norm(1, 1).rvs((5, 2))), axis=0)  # 10 rows, 2 cols

# tweak so we have n observations
np.concatenate((ss.norm(0, 1).rvs((n, 2)), ss.norm(1, 1).rvs((n, 2))), axis=0)
# generate outcomes
outcomes = np.concatenate((np.repeat(0, n), np.repeat(1, n)))
"""
# Create Function
def generateSyntheticData(k=50):
    """
    Create two sets of points from bivariate normal distributions
    :param k: number of points (default is 50)
    :return: tuple containing the points and the outcomes
    """
    # tweak so we have n observations
    points = np.concatenate((ss.norm(0, 1).rvs((k, 2)), ss.norm(1, 1).rvs((k, 2))), axis=0)
    # generate outcomes
    outcomes = np.concatenate((np.repeat(0, k), np.repeat(1, k)))
    return (points, outcomes)
"""
# print(generateSyntheticData(20))
n = 20
pointArray, outcomeArray = generateSyntheticData(n)

plt.figure()
plt.plot(pointArray[:n, 0], pointArray[:n, 1], "ro")
plt.plot(pointArray[n:, 0], pointArray[n:, 1], "bo")
plt.show()
"""

# -- 3.3.6 Making a Prediction Grid
"""
Learn about enumerate and NumPy meshgrid
Task: once we've observed our data, we can examine some part of the predictor space and compute
the class prediction for each point in the grid using knn classifier -> ask how it classifies
all points belonging to a rectangular region of the predictor space
"""
def makePredicitionGrid(predictors, outcomes, limits, h, k):
    """
    Classify each point on the prediction grid
    :param predictors:
    :param outcomes:
    :param limits:
    :param h:
    :param k:
    :return:
    """
    (x_min, x_max, y_min, y_max) = limits
    xs = np.arange(x_min, x_max, h)
    ys = np.arange(y_min, y_max, h)
    xx, yy = np.meshgrid(xs, ys)

    predctionGrid = np.zeros(xx.shape, dtype=int)

    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            p = np.array([x, y])
            predctionGrid[j, i] = kNNPredict(p, predictors, outcomes, k)

    return (xx, yy, predctionGrid)
"""
# looking more into enumerate()
seasons = ["spring", "summer", "fall", "winter"]
print(list(enumerate(seasons)))  # [(0, 'spring'), (1, 'summer'), (2, 'fall'), (3, 'winter')]
for index, season in enumerate(seasons):
    print(index, season)
    # 0 spring
    # 1 summer
    # 2 fall
    # 3 winter
"""

# -- 3.3.7 Plotting Prediction Grid
def plot_prediction_grid(xx, yy, prediction_grid, filename):
    """ Plot KNN predictions for every point on the grid."""
    from matplotlib.colors import ListedColormap
    background_colormap = ListedColormap(["hotpink", "lightskyblue", "yellowgreen"])
    observation_colormap = ListedColormap(["red", "blue", "green"])
    plt.figure(figsize=(10, 10))
    plt.pcolormesh(xx, yy, prediction_grid, cmap=background_colormap, alpha=0.5)
    plt.scatter(predictors[:, 0], predictors[:, 1], c=outcomes, cmap=observation_colormap, s=50)
    plt.xlabel('Variable 1')
    plt.ylabel('Variable 2')
    plt.xticks(())
    plt.yticks(())
    plt.xlim(np.min(xx), np.max(xx))
    plt.ylim(np.min(yy), np.max(yy))
    plt.savefig(filename)
"""
(predictors, outcomes) = generateSyntheticData()

# print(predictors.shape)
k = 5
fileName = "kNN_Synth5.pdf"
limits = (-3, 4, -3, 4)
h = 0.1
(xx, yy, prediction_grid) = makePredicitionGrid(predictors, outcomes, limits, h, k)
plot_prediction_grid(xx, yy, prediction_grid, fileName)

k = 50
fileName = "kNN_Synth50.pdf"
limits = (-3, 4, -3, 4)
h = 0.1
(xx, yy, prediction_grid) = makePredicitionGrid(predictors, outcomes, limits, h, k)
plot_prediction_grid(xx, yy, prediction_grid, fileName)
"""

# -- 3.3.8 Applying the kNN Method
iris = datasets.load_iris()
print(iris)
predictors = iris.data[:, 0:2]
outcomes = iris.target

plt.plot(predictors[outcomes == 0][:, 0], predictors[outcomes == 0][:, 1], "ro")
plt.plot(predictors[outcomes == 1][:, 0], predictors[outcomes == 1][:, 1], "go")
plt.plot(predictors[outcomes == 2][:, 0], predictors[outcomes == 2][:, 1], "bo")
plt.savefig("kNN_irisPlot.pdf")

k = 5
fileName = "kNN_irisGrid.pdf"
limits = (4, 8, 1.5, 4.5)
h = 0.1
(xx, yy, prediction_grid) = makePredicitionGrid(predictors, outcomes, limits, h, k)
plot_prediction_grid(xx, yy, prediction_grid, fileName)

kNN = KNeighborsClassifier(n_neighbors=5)
kNN.fit(predictors, outcomes)
sk_predictions = kNN.predict(predictors)
print(sk_predictions.shape)
myPredictions = np.array([kNNPredict(p, predictors, outcomes, 5) for p in predictors])
print(myPredictions.shape)
# find how often skPredictions == myPredictions
print(100 * np.mean(sk_predictions == myPredictions))  # 96.0

# how often do myPredictions/sk_predictions == outcomes?
print(100 * np.mean(outcomes == myPredictions))  # 84.66666666666667
print(100 * np.mean(sk_predictions == outcomes))  # 83.33333333333334
