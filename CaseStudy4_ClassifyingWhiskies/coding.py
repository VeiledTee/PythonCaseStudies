import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralCoclustering

# -- 4.1.1 Getting Started with Pandas
"""
Series

# x = pd.Series([6, 3, 8, 6])
# print(x)
# 0    6
# 1    3
# 2    8
# 3    6

x = pd.Series([6, 3, 8, 6], index=["q", "w", "e", "r"])
# print(x)
# q    6
# w    3
# e    8
# r    6
# print(x["w"])  # 3
# print(x[["r", "w"]])
# r    6
# w    3

# age = {"Tim": 29, "Jim": 31, "Pam": 27, "Sam": 35}
# x = pd.Series(age)
# print(x)
# Tim    29
# Jim    31
# Pam    27
# Sam    35
# print(x.index)
# Index(['q', 'w', 'e', 'r'], dtype='object')
# print(sorted(x.index))  # returns list of sorted indices
# ['e', 'q', 'r', 'w']
# print(x.reindex(sorted(x.index)))  # can reindex objects to change indices
# e    8
# q    6
# r    6
# w    3
"""
"""
DataFrames

data = {'name': ['Tim', 'Jim', 'Pam', 'Sam'],
        'age': [29, 31, 27, 35],
        'ZIP': ['02115', '02130', '67700', '00100']}
# pd.DataFrame("data we want to use", columns="what columns we use, and their order")
y = pd.DataFrame(data, columns=['name', 'age', 'ZIP'])
# print(y)
#   name  age    ZIP
# 0  Tim   29  02115
# 1  Jim   31  02130
# 2  Pam   27  67700
# 3  Sam   35  00100
# print(y['name'])
# 0    Tim
# 1    Jim
# 2    Pam
# 3    Sam
# print(y.name)
# 0    Tim
# 1    Jim
# 2    Pam
# 3    Sam
"""
""" CAN  PERFORM ARITHMETIC OPERATIONS """
"""
z = pd.Series([6, 3, 8, 6], index=["e", "q", "r", "t"])
# print(x + z)
# e    14.0
# q     9.0
# r    14.0
# t     NaN -> Not a Number
# w     NaN -> Not a Number
"""

# -- 4.1.2 Loading and Inspecting Data
whisky = pd.read_csv("whiskies.txt")
whisky["Region"] = pd.read_csv("regions.txt")
"""
# print(whisky.head(5))
#    RowID Distillery  Body  ...   Latitude   Longitude     Region
# 0      1  Aberfeldy     2  ...     286580      749680  Highlands
# 1      2   Aberlour     3  ...     326340      842570   Speyside
# 2      3     AnCnoc     1  ...     352960      839320  Highlands
# 3      4     Ardbeg     4  ...     141560      646220      Islay
# 4      5    Ardmore     2  ...     355350      829140  Highlands
# print(whisky.tail(5))
#     RowID    Distillery  Body  ...   Latitude   Longitude     Region
# 81     82     Tobermory     1  ...     150450      755070    Islands
# 82     83       Tomatin     2  ...     279120      829630  Highlands
# 83     84     Tomintoul     0  ...     315100      825560   Speyside
# 84     85       Tormore     2  ...     315180      834960   Speyside
# 85     86  Tullibardine     2  ...     289690      708850  Highlands
# print(whisky.iloc[0:10, 0:5])  # look only at first 10 ros and first 5 columns
#    RowID    Distillery  Body  Sweetness  Smoky
# 0      1     Aberfeldy     2          2      2
# 1      2      Aberlour     3          3      1
# 2      3        AnCnoc     1          3      2
# 3      4        Ardbeg     4          1      4
# 4      5       Ardmore     2          2      2
# 5      6   ArranIsleOf     2          3      1
# 6      7  Auchentoshan     0          2      0
# 7      8     Auchroisk     2          3      1
# 8      9      Aultmore     2          2      1
# 9     10      Balblair     2          3      2

# print(whisky.columns)  # print all the names of columns in the data set
"""
flavours = whisky.iloc[:, 2:14]
"""
print(flavours)
#     Body  Sweetness  Smoky  Medicinal  ...  Nutty  Malty  Fruity  Floral
# 0      2          2      2          0  ...      2      2       2       2
# 1      3          3      1          0  ...      2      3       3       2
# 2      1          3      2          0  ...      2      2       3       2
# 3      4          1      4          4  ...      1      2       1       0
# 4      2          2      2          0  ...      2      3       1       1
# ..   ...        ...    ...        ...  ...    ...    ...     ...     ...
# 81     1          1      1          0  ...      1      2       2       2
# 82     2          3      2          0  ...      1      2       0       1
# 83     0          3      1          0  ...      1      2       1       2
# 84     2          2      1          0  ...      2      1       0       0
# 85     2          3      0          0  ...      1      2       2       1
"""

# -- 4.1.3 Exploring Correlations
corrFlavours = pd.DataFrame.corr(flavours)
# print(corrFlavours)
# Body  Sweetness     Smoky  ...     Malty    Fruity    Floral
# Body       1.000000  -0.136518  0.524032  ... -0.116859 -0.013205 -0.461203
# Sweetness -0.136518   1.000000 -0.405897  ... -0.001516  0.019820  0.144987
# Smoky      0.524032  -0.405897  1.000000  ... -0.192875 -0.312970 -0.431663
# Medicinal  0.354050  -0.392017  0.686071  ... -0.258959 -0.330975 -0.511323
# Tobacco    0.168718  -0.147871  0.365501  ... -0.059347 -0.235145 -0.212375
# Honey      0.082031   0.132558 -0.195318  ...  0.310184  0.108822  0.183029
# Spicy      0.188500  -0.054200  0.231745  ...  0.036303  0.144714  0.034663
# Winey      0.408576   0.115727 -0.028190  ...  0.112368  0.090694 -0.126932
# Nutty      0.126323  -0.032493 -0.023132  ...  0.066157  0.071765  0.018302
# Malty     -0.116859  -0.001516 -0.192875  ...  1.000000  0.207288  0.106309
# Fruity    -0.013205   0.019820 -0.312970  ...  0.207288  1.000000  0.262336
# Floral    -0.461203   0.144987 -0.431663  ...  0.106309  0.262336  1.000000

plt.figure(figsize=(10, 10))
plt.pcolor(corrFlavours)
plt.colorbar()
plt.savefig("CorrFlavours.pdf")

corrWhisky = pd.DataFrame.corr(flavours.transpose())
plt.figure(figsize=(10, 10))
plt.pcolor(corrWhisky)
plt.axis("tight")
plt.colorbar()
plt.savefig("CorrWhisky.pdf")

# -- 4.1.4 Clustering Whiskies by Flavour Profile
model = SpectralCoclustering(n_clusters=6, random_state=0)
model.fit(corrWhisky)
# print(model.rows_)  # output -> array with dimensions number of row clusters * number of rows
# ^^ CORRELATION MATRIX ^^
# Each row in this array identifies a cluster, here ranging from 0 to 5
# Each column identifies a row in the correlation matrix, here ranging from 0 to 85
"""
[[False False False False False False False False False False False False
  False False False False False False False  True False False False False
  False False False False False False False False False False False False
  False False False  True False False False False False False False False
  False False False False False False False  True False False False False
  False False False False False False  True False  True False False False
  False False False False False False False False False False False False
  False False]
 [False False False False False  True False False False False False  True
  False  True False False  True False  True False False False False False
  False False False False False False False False  True  True  True False
  False False  True False  True False False False False  True False  True
   True False False False False False  True False False False False  True
  False False False False False False False False False  True False False
   True False False False False False False False  True False False  True
  False  True]
 [ True False False False  True False False False  True False False False
   True False  True  True False  True False False False False False False
  False False False False False False False  True False False False  True
  False False False False False False False False False False  True False
  False False False False False False False False  True False False False
  False  True False  True  True False False  True False False False  True
  False  True False  True False False  True False False False False False
  False False]
 [False False False  True False False False False False False False False
  False False False False False False False False False  True False  True
  False False False False False False False False False False False False
  False False False False False False False False False False False False
  False False False False False False False False False  True  True False
  False False False False False False False False False False False False
  False False False False False  True False False False False False False
  False False]
 [False  True False False False False False  True False False  True False
  False False False False False False False False False False False False
  False False  True  True False  True False False False False False False
   True False False False False False  True  True  True False False False
  False False False  True  True  True False False False False False False
  False False  True False False  True False False False False  True False
  False False  True False False False False False False False  True False
   True False]
 [False False  True False False False  True False False  True False False
  False False False False False False False False  True False  True False
   True  True False False  True False  True False False False False False
  False  True False False False  True False False False False False False
  False  True  True False False False False False False False False False
   True False False False False False False False False False False False
  False False False False  True False False  True False  True False False
  False False]]
"""
# by summing all columns, we can find out how many observations belong to each cluster
# print(np.sum(model.rows_, axis=1))  # axis=1 is columns
"""
[ 5 20 19  6 19 17]
"""
# by summing all rows, we can find out how many clusters belong to each observation
# print(np.sum(model.rows_, axis=0))  # axis=0 is rows
"""
[1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1]
"""
# print(model.row_labels_)  # index 0 = 2 -> 0th observation belongs to cluster 2
"""
[2 4 5 3 2 1 5 4 2 5 4 1 2 1 2 2 1 2 1 0 5 3 5 3 5 5 4 4 5 4 5 2 1 1 1 2 4
 5 1 0 1 5 4 4 4 1 2 1 1 5 5 4 4 4 1 0 2 3 3 1 5 2 4 2 2 4 0 2 0 1 4 2 1 2
 4 2 5 3 2 5 1 5 4 1 4 1]
"""

# -- 4.1.5 Comparing Correlation Matrices
# We first extract the group labels from the model and append them to the whisky table.
# We also specify their index explicitly.
whisky['Group'] = pd.Series(model.row_labels_, index=whisky.index)
# We then reorder the rows in increasing order by group labels.
# These are the group labels that we discovered using spectral co-clustering
whisky = whisky.iloc[np.argsort(model.row_labels_)]
# We reset the index of our DataFrame. We have now reshuffled the rows and columns of the table
whisky = whisky.reset_index(drop=True)
# We have now reshuffled the rows and columns of the table.
# So let's also recalculate the correlation matrix, and turn into a numpy array
correlations = pd.DataFrame.corr(whisky.iloc[:, 2:14].transpose())  # returns dataframe
correlations = np.array(correlations)

plt.figure(figsize=(14, 7))
plt.subplot(121)
plt.pcolor(corrWhisky)
plt.title("Original")
plt.axis("tight")
plt.subplot(122)
plt.pcolor(correlations)
plt.title("Rearranged")
plt.axis("tight")
plt.savefig("correlations.pdf")
