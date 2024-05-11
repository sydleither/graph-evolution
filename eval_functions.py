import networkx as nx
import numpy as np


class Evaluation:
    #node-level topological properties
    def in_degree_distribution(self, network) -> list[float]:
        networkx_obj = network.getNetworkxObject()
        num_nodes = network.numNodes
        degree_sequence = list(d for _, d in networkx_obj.in_degree())
        freq = [0]*(num_nodes+1)
        for d in degree_sequence:
            freq[d] += 1/num_nodes
        return freq
    

    def out_degree_distribution(self, network) -> list[float]:
        networkx_obj = network.getNetworkxObject()
        num_nodes = network.numNodes
        degree_sequence = list(d for _, d in networkx_obj.out_degree())
        freq = [0]*(num_nodes+1)
        for d in degree_sequence:
            freq[d] += 1/num_nodes
        return freq


    def avg_shortest_path_length_distribution(self, network) -> list[float]:
        weight = 1/network.numNodes**2
        shortest_path = dict(nx.shortest_path_length(network.getNetworkxObject()))
        avg_shortest = sorted([weight*np.mean(list(shortest_path[i].values())) for i in range(len(shortest_path))], reverse=True)
        return avg_shortest
    

    def betweenness_distribution(self, network) -> list[float]:
        networkx_obj = network.getNetworkxObject()
        return sorted(list(nx.betweenness_centrality(networkx_obj).values()), reverse=True)
    
    
    #topological properties
    def strong_components(self, network) -> int:
        return len(list(nx.strongly_connected_components(network.getNetworkxObject())))


    def weak_components(self, network) -> int:
        return len(list(nx.weakly_connected_components(network.getNetworkxObject())))


    def connectance(self, network) -> float:
        return network.numInteractions / network.numNodes**2


    def clustering_coefficient(self, network) -> float:
        return nx.average_clustering(network.getNetworkxObject())
    

    def transitivity(self, network) -> float:
        return nx.transitivity(network.getNetworkxObject())


    def proportion_of_self_loops(self, network) -> float:
        return sum([1 for i in range(network.numNodes) if network.adjacencyMatrix[i][i] != 0]) / network.numNodes
    

    def diameter(self, network) -> int:
        shortest_path = dict(nx.shortest_path_length(network.getNetworkxObject()))
        return max([max(shortest_path[i].values()) for i in range(len(shortest_path))])
    

    def number_of_modules(self, network) -> int:
        if network.numInteractions > 0:
            return len(nx.community.greedy_modularity_communities(network.getNetworkxObject()))
        else:
            return 0


    #interaction strength properties
    def positive_interactions_proportion(self, network) -> float:
        num_interactions = network.numInteractions
        return network.numPositive / num_interactions if num_interactions > 0 else 0


    def average_positive_interactions_strength(self, network) -> float:
        sum_pos = sum([sum([y for y in x if y > 0]) for x in network.adjacencyMatrix])
        num_pos = network.numPositive
        return  sum_pos / num_pos if num_pos > 0 else 0
    

    def average_negative_interactions_strength(self, network) -> float:
        sum_neg = sum([sum([y for y in x if y < 0]) for x in network.adjacencyMatrix])
        num_neg = network.numNegative
        return  sum_neg / num_neg if num_neg > 0 else 0


    def proportion_of_self_loops_positive(self, network) -> float:
        return sum([1 for i in range(network.numNodes) if network.adjacencyMatrix[i][i] > 0]) / network.numNodes
    

#all callable evaluation functions
functions = {funcName:getattr(Evaluation, funcName) for funcName in dir(Evaluation) 
                if callable(getattr(Evaluation, funcName)) and not funcName.startswith("__")}

#non-distribution callables
properties = {funcName:getattr(Evaluation, funcName) for funcName in dir(Evaluation) 
                if callable(getattr(Evaluation, funcName)) and not funcName.startswith("__") and
                not funcName.endswith("_distribution")}

#distribution callables
distributions = {funcName:getattr(Evaluation, funcName) for funcName in dir(Evaluation) 
                if callable(getattr(Evaluation, funcName)) and not funcName.startswith("__") and
                funcName.endswith("_distribution")}


if __name__ == "__main__":
    e = Evaluation()
    print(e.functions)