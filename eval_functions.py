import networkx as nx
import numpy as np
from organism import Organism


class Evaluation:
    #topological properties
    def connectance(self, network:Organism) -> float:
        return network.numInteractions / network.numNodes**2
    

    def proportion_of_self_loops(self, network:Organism) -> float:
        return sum([1 for i in range(network.numNodes) if network.adjacencyMatrix[i][i] != 0]) / network.numNodes
    

    def diameter(self, network:Organism) -> int:
        shortest_path = dict(nx.shortest_path_length(network.getNetworkxObject()))
        return max([max(shortest_path[i].values()) for i in range(len(shortest_path))])
    

    def in_degree_distribution(self, network:Organism) -> float:
        nn = network.numNodes
        networkx_obj = network.getNetworkxObject()
        a = 2.5
        dist = [x**-a for x in [(y/nn)+1 for y in range(nn)]]
        degree_sequence = sorted([networkx_obj.in_degree(n)/nn for n in networkx_obj.nodes()], reverse=True)
        squares = sum([(dist[i]-degree_sequence[i])**2 for i in range(nn)])
        return squares
    

    def out_degree_distribution(self, network:Organism) -> float:
        nn = network.numNodes
        networkx_obj = network.getNetworkxObject()
        a = 2.5
        dist = [x**-a for x in [(y/nn)+1 for y in range(nn)]]
        degree_sequence = sorted([networkx_obj.out_degree(n)/nn for n in networkx_obj.nodes()], reverse=True)
        squares = sum([(dist[i]-degree_sequence[i])**2 for i in range(nn)])
        return squares
    

    #interaction strength properties
    def positive_interactions_proportion(self, network:Organism) -> float:
        return network.numPositive / network.numInteractions


    def average_positive_interactions_strength(self, network:Organism) -> float:
        return sum([sum([y for y in x if y > 0]) for x in network.adjacencyMatrix]) / network.numPositive


    def proportion_of_self_loops_positive(self, network:Organism) -> float:
        return sum([1 for i in range(network.numNodes) if network.adjacencyMatrix[i][i] > 0]) / network.numNodes


    def number_of_mutualistic_pairs(self, network:Organism) -> int:
        adj = network.adjacencyMatrix
        nn = network.numNodes
        return sum([sum([1 for j in range(i+1, nn) if adj[i][j] > 0 and adj[j][i] > 0]) for i in range(nn)])


    def number_of_competiton_pairs(self, network:Organism) -> int:
        adj = network.adjacencyMatrix
        nn = network.numNodes
        return sum([sum([1 for j in range(i+1, nn) if adj[i][j] < 0 and adj[j][i] < 0]) for i in range(nn)])


    def number_of_parasitism_pairs(self, network:Organism) -> int:
        adj = network.adjacencyMatrix
        nn = network.numNodes
        return sum([sum([1 for j in range(i+1, nn) if (adj[i][j] < 0 and adj[j][i] > 0) or (adj[i][j] > 0 and adj[j][i] < 0)]) for i in range(nn)])