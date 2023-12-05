import networkx as nx

from organism import Organism


class Evaluation:
    #topological properties
    def connectance(self, network:Organism) -> float:
        return network.numInteractions / network.numNodes**2


    #interaction strength properties
    def positive_interactions_proportion(self, network:Organism) -> float:
        return network.numPositive / network.numInteractions


    def average_positive_interactions_strength(self, network:Organism) -> float:
        return sum([sum([y for y in x if y > 0]) for x in network.adjacencyMatrix]) / network.numPositive


    def proportion_of_self_loops_positive(self, network:Organism) -> float:
        return sum([1 for i in range(network.numNodes) if network.adjacencyMatrix[i][i] > 0]) / network.numNodes


    def number_of_mutualistic_pairs(self, network:Organism) -> float:
        adj = network.adjacencyMatrix
        nn = network.numNodes
        return sum([sum([1 for j in range(i+1, nn) if adj[i][j] > 0 and adj[j][i] > 0]) for i in range(nn)])


    def number_of_competiton_pairs(self, network:Organism) -> float:
        adj = network.adjacencyMatrix
        nn = network.numNodes
        return sum([sum([1 for j in range(i+1, nn) if adj[i][j] < 0 and adj[j][i] < 0]) for i in range(nn)])


    def number_of_parasitism_pairs(self, network:Organism) -> float:
        adj = network.adjacencyMatrix
        nn = network.numNodes
        return sum([sum([1 for j in range(i+1, nn) if (adj[i][j] < 0 and adj[j][i] > 0) or (adj[i][j] > 0 and adj[j][i] < 0)]) for i in range(nn)])