from collections import Counter
import numpy as np
import pandas as pd
import networkx as nx

# ----- Exercise 1 ----- #
"""
-> Create a function marginal_prob that takes a dictionary chars with 
personal IDs as keys and characteristics as values; it should return a 
dictionary with characteristics as keys and their marginal probability 
(frequency of occurence of a characteristic divided by the sum of frequencies 
of each characteristic) as values.
-> Create a function chance_homophily(chars) that takes a dictionary chars 
defined as above and computes the chance homophily (homophily due to chance alone) 
for that characteristic.
-> A sample of three peoples' favorite colors is given in favorite_colors. Use 
your function to compute the chance homophily in this group, and store it as 
color_homophily.
-> Print color_homophily
"""
def marginal_prob(chars):
    frequencies = dict(Counter(chars.values()))
    sum_frequencies = sum(frequencies.values())
    return {char: freq / sum_frequencies for char, freq in frequencies.items()}


def chance_homophily(chars):
    marginal_probs = marginal_prob(chars)
    return np.sum(np.square(list(marginal_probs.values())))


favorite_colors = {
    "ankit": "red",
    "xiaoyu": "blue",
    "mary": "blue"
}

# color_homophily = chance_homophily(favorite_colors)
# print(color_homophily)  # 0.5555555555555556

# ----- Exercise 2 ----- #
"""
-> Note that individual_characteristics.dta contains several characteristics 
for each individual in the dataset such as age, religion, and caste. Use the 
pandas library to read in and store these characteristics as a dataframe called df.
-> Store separate datasets for individuals belonging to Villages 1 and 2 as 
df1 and df2, respectively.
-> Note that some attributes may be missing for some individuals.
-> Use the head method to display the first few entries of df1.
"""
df = pd.read_csv("https://courses.edx.org/asset-v1:HarvardX+PH526x+2T2019+type@asset+block@individual_characteristics.csv", low_memory=False, index_col=0)

df1 = df[df["village"]==1]
df2 = df[df["village"]==2]
# How many people had a resp_gender value of 1 in the first 5 entries of df1
# print(df1['resp_gend'])  # 3

# ----- Exercise 3 ----- #
"""
-> Define dictionaries with personal IDs as keys and a given covariate 
for that individual as values. Complete this for the sex, caste, and 
religion covariates, for Villages 1 and 2.
-> For Village 1, store these dictionaries into variables named sex1, 
caste1, and religion1.
-> For Village 2, store these dictionaries into variables named sex2, 
caste2, and religion2
"""

sex1 = dict(zip(df1['pid'], df1['resp_gend']))
caste1 = dict(zip(df1['pid'], df1['caste']))
religion1 = dict(zip(df1['pid'], df1['religion']))

# Continue for df2 as well.
sex2 = dict(zip(df2['pid'], df2['resp_gend']))
caste2 = dict(zip(df2['pid'], df2['caste']))
religion2 = dict(zip(df2['pid'], df2['religion']))

# What is the caste value for personal ID 202802 in village 2
# print(caste2[202802])  # OBC

# ----- Exercise 4 ----- #
"""
-> Use chance_homophily to compute the chance homophily for sex, caste, and 
religion In Villages 1 and 2. Consider whether the chance homophily for 
any attribute is very high for either village.

Which characteristic has the highest value of chance homophiliy?
"""
# print(chance_homophily(sex1))  # 0.5027299861680701
# print(chance_homophily(sex2))  # 0.5005945303210464
# print(chance_homophily(caste1))  # 0.6741488509791551
# print(chance_homophily(caste2))  # 0.425368244800893
# print(chance_homophily(religion1))  # 0. 9804896988521925
# print(chance_homophily(religion2))  # 1.0

# ----- Excercise 5 ----- #
"""
-> Complete the function homophily(), which takes a network G, a dictionary 
of node characteristics chars, and node IDs IDs. For each node pair, determine 
whether a tie exists between them, as well as whether they share a characteristic. 
The total count of these is num_ties and num_same_ties, respectively, and their 
ratio is the homophily of chars in G. Complete the function by choosing where to 
increment num_same_ties and num_ties
"""

def homophily(G, chars, IDs):
    """
    Given a network G, a dict of characteristics chars for node IDs,
    and dict of node IDs for each node in the network,
    find the homophily of the network.
    """
    num_same_ties = 0
    num_ties = 0
    for n1, n2 in G.edges():
        if IDs[n1] in chars and IDs[n2] in chars:
            if G.has_edge(n1, n2):
                num_ties += 1
                if chars[IDs[n1]] == chars[IDs[n2]]:
                    num_same_ties += 1
    return num_same_ties / num_ties


# ----- Excercise 6 ----- #
"""
-> In this dataset, each individual has a personal ID, or PID, stored in 
key_vilno_1.csv and key_vilno_2.csv for villages 1 and 2, respectively. 
data_filepath1 and data_filepath2 contain the URLs to the datasets used 
in this exercise. Use pd.read_csv to read in and store key_vilno_1.csv 
and key_vilno_2.csv as pid1 and pid2 respectively.
"""
data_filepath1 = "https://courses.edx.org/asset-v1:HarvardX+PH526x+2T2019+type@asset+block@key_vilno_1.csv"
data_filepath2 = "https://courses.edx.org/asset-v1:HarvardX+PH526x+2T2019+type@asset+block@key_vilno_2.csv"

pid1 = pd.read_csv(data_filepath1, index_col=0)
pid2 = pd.read_csv(data_filepath2, index_col=0)

# What is the personal ID of the person at index 100 in village 1?
# print(pid1.iloc[100])  # 102205

# ----- Excercise 7 ----- #
"""
-> Use your homophily() function to compute the observed homophily for sex, 
caste, and religion in Villages 1 and 2. Print all six values.
-> Use chance_homophily() to compare the observed homophily values to the chance 
homophily values. Are observed values higher or lower than those 
expected by chance?
"""

A1 = np.array(pd.read_csv("https://courses.edx.org/asset-v1:HarvardX+PH526x+2T2019+type@asset+block@adj_allVillageRelationships_vilno1.csv", index_col=0))
A2 = np.array(pd.read_csv("https://courses.edx.org/asset-v1:HarvardX+PH526x+2T2019+type@asset+block@adj_allVillageRelationships_vilno2.csv", index_col=0))
G1 = nx.to_networkx_graph(A1)
G2 = nx.to_networkx_graph(A2)

pid1 = pd.read_csv(data_filepath1, dtype=int)['0'].to_dict()
pid2 = pd.read_csv(data_filepath2, dtype=int)['0'].to_dict()

# print("Village 1 observed proportion of same sex:", homophily(G1, sex1, pid1))
# print("Village 1 observed proportion of same caste:", homophily(G1, caste1, pid1))
# print("Village 1 observed proportion of same religion:", homophily(G1, religion1, pid1))
#
# print("Village 2 observed proportion of same sex:", homophily(G2, sex2, pid2))
# print("Village 2 observed proportion of same caste:", homophily(G2, caste2, pid2))
# print("Village 2 observed proportion of same religion:", homophily(G2, religion2, pid2))
#
# print("Village 1 chance of same sex:", chance_homophily(sex1))
# print("Village 1 chance of same caste:", chance_homophily(caste1))
# print("Village 1 chance of same religion:", chance_homophily(religion1))
#
# print("Village 2 chance of same sex:", chance_homophily(sex2))
# print("Village 2 chance of same caste:", chance_homophily(caste2))
# print("Village 2 chance of same religion:", chance_homophily(religion2))

"""
Village 1 observed proportion of same sex: 0.5908629441624366
Village 1 observed proportion of same caste: 0.7959390862944162
Village 1 observed proportion of same religion: 0.9908629441624366
Village 2 observed proportion of same sex: 0.5658073270013568
Village 2 observed proportion of same caste: 0.8276797829036635
Village 2 observed proportion of same religion: 1.0
Village 1 chance of same sex: 0.5027299861680701
Village 1 chance of same caste: 0.6741488509791551
Village 1 chance of same religion: 0.9804896988521925
Village 2 chance of same sex: 0.5005945303210464
Village 2 chance of same caste: 0.425368244800893
Village 2 chance of same religion: 1.0
"""