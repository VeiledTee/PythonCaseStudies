import networkx as nx
import matplotlib.pyplot as plt
from scipy.stats import bernoulli
import numpy as np

# -- 4.3.1 Intro to Network Analysis
"""
Nodes: components
Edge: interactions between components
Network: Real world object (road network)
Graph: mathematical interpretation of a network
    Neighbours: Two connected vertices
    Path: Sequence of unique vertices, such that any two vertices are connected by an edge
    Path Length: # edges in path
    Connected Graph: if there is a path from one vertex to any other vertex
    Disconnected Graph: opposite of above
        Component: in a disconnected graph, components are the individual graphs that make up the whole
    Size: Number of nodes
"""

# -- 4.3.2 Basics of NetworkX
G = nx.Graph()
G.add_node(1)
G.add_nodes_from([2, 3])
G.add_nodes_from(["U", "V"])
# print(G.nodes())  # [1, 2, 3, 'U', 'V']
G.add_edge(1, 2)
G.add_edge("U", "V")
G.add_edges_from([(1, 3), (1, 4), (1, 5), (1, 6)])
G.add_edge("U", "W")
# print(G.edges())  # [(1, 2), (1, 3), (1, 4), (1, 5), (1, 6), ('U', 'V'), ('U', 'W')]
G.remove_node(2)
# print(G.nodes())  # [1, 3, 'U', 'V', 4, 5, 6, 'W']
G.remove_nodes_from([4, 5])
# print(G.nodes())  # [1, 3, 'U', 'V', 6, 'W']
G.remove_edge(1, 3)
# print(G.edges())  # [(1, 6), ('U', 'V'), ('U', 'W')]
G.remove_edges_from([(1, 2), ("U", "V")])
# print(G.edges())  # [(1, 6), ('U', 'W')]
# print(G.number_of_nodes())  # 6
# print(G.number_of_edges())  # 2

# -- 4.3.3 Graph Visualization
G = nx.karate_club_graph()
# nx.draw(G, with_labels=True, node_color='lightblue', edge_color='gray')
# plt.savefig("KarateGraph.pdf")
# print(G.degree())  # [(0, 16), (1, 9), (2, 10), etc...]
# output means (node, number of edges) or (karate club member, number of friends)
# print(G.degree()[33])  # 17 -> accesses DegreeView object (similar to dictionary) and finds value
# print(G.degree(33))  # 17 -> search through DegreeView object to find value
# print(G.degree(0) is G.degree()[0])  # True
# plt.show()

# -- 4.3.4 Random Graphs
# print(bernoulli.rvs(p=0.2))  # 0.2 chance to print 1, 0.8 chance to print 0
N = 20
p = 0.2

# create empty graph
G = nx.Graph()
# add all N nodes in graph
G.add_nodes_from(range(N))

# loop over all pairs of nodes
for node1 in G.nodes():  # G.nodes() returns NodeView of tuples, need 2 loops
    for node2 in G.nodes():
        if bernoulli.rvs(p=p):  # first p is keyword arg, second p is value
            # add edge w probability 'p'
            G.add_edge(node1, node2)
# print(G.number_of_nodes())  # 20
# plt.show()
"""
This graph will look too densely connected to be right.
In fact, we have a subtle error in our code.
Now we're considering each pair of nodes twice, not just once, as we should.
Consider running through the two loops, one nested inside the other.
Consider a situation where node 1 is equal to 1 and node 2 is equal to 10.
In this case, we're considering the pair 1,10.
Now if you move forward in that loop, there's going to be a moment where node 1 is equal to 10, and node 2 is equal to 1.
In this case, we are considering the node pair 10,1.
But because our graph is undirected, we should consider each pair of nodes just one time.
For this reason, we need to impose an extra constraint such as:
    node 1 less than node 2 or node 1 greater than node 2.
Either additional constraint will force us to consider each pair of nodes just one time.

Above: Original code
Below: Edited code
"""
def er_graph(N, probability):
    """
    Generate an ER graph
    :param N: Number of nodes
    :param probability: Probability that an edge is added between 2 nodes
    :return: graph G
    """
    g = nx.Graph()
    g.add_nodes_from(range(N))
    for n1 in g.nodes():  # g.nodes() returns NodeView of tuples, need 2 loops
        for n2 in g.nodes():
            if n1 < n2 and bernoulli.rvs(p=probability):
                g.add_edge(n1, n2)
    return g

# nx.draw(er_graph(50, 0.08), node_size=40, node_color='gray')
# plt.savefig("erGraph1.pdf")

# -- 4.3.5 Plotting the Degree Distribution
def plotDegreeDistribution(g):
    degree_sequence = [d for n, d in g.degree()]
    plt.hist(degree_sequence, histtype="step")
    plt.xlabel("Degree $k$")
    plt.ylabel("$P(k)$")
    plt.title("Degree Distribution")

# Plotting the above function
# G = er_graph(50, 0.08)
# plotDegreeDistribution(G)
#
# G = er_graph(50, 0.08)
# plotDegreeDistribution(G)
# plt.savefig("histx2.pdf")

G1 = er_graph(500, 0.08)
plotDegreeDistribution(G1)
G2 = er_graph(500, 0.08)
plotDegreeDistribution(G2)
G3 = er_graph(500, 0.08)
plotDegreeDistribution(G3)
plt.savefig("histx3.pdf")

# -- 4.3.6 Descriptive Statistics of Empirical Social Networks
A1 = np.loadtxt("adj_allVillageRelationships_vilno_1.csv", delimiter=',')
A2 = np.loadtxt("adj_allVillageRelationships_vilno_2.csv", delimiter=',')

G1 = nx.to_networkx_graph(A1)
G2 = nx.to_networkx_graph(A2)

def basicNetStats(g):
    print('Number of nodes: ' + str(g.number_of_nodes()))
    print('Number of edges: ' + str(g.number_of_edges()))
    degree_sequence = [d for n, d in g.degree()]
    print("Average degree: %.2f" % np.mean(degree_sequence))

# basicNetStats(G1)
"""
Number of nodes: 843
Number of edges: 3405
Average degree: 8.08
"""
# basicNetStats(G2)
"""
Number of nodes: 877
Number of edges: 3063
Average degree: 6.99
"""
# plotDegreeDistribution(G1)
# plotDegreeDistribution(G2)
# plt.savefig('villageHist.pdf')

# -- 4.3.7 Finding the Largest Connected Component
gen = (G1.subgraph(c) for c in nx.connected_components(G1))
g = gen.__next__()
# print(g.number_of_nodes())  # 825
# len() of a graph returns the number of nodes
G1_LCC = max((G1.subgraph(c) for c in nx.connected_components(G1)), key=len)
G2_LCC = max((G2.subgraph(c) for c in nx.connected_components(G2)), key=len)
# check the Largest Connected Component (LCC) of each graph
# print(len(G1_LCC))  # 825
# print(len(G2_LCC))  # 810
# print(len(G1_LCC) / G1.number_of_nodes())  # 0.9786476868327402 -> 97% of nodes are in the LCC
# print(len(G2_LCC) / G2.number_of_nodes())  # 0.9236031927023945 -> 92% of nodes are in the LCC

plt.figure()
nx.draw(G1_LCC, node_color='red', edge_color='gray', node_size=20)
plt.savefig("village1.pdf")

plt.figure()
nx.draw(G1_LCC, node_color='green', edge_color='gray', node_size=20)
plt.savefig("village2.pdf")
