import networkx as nx
import numpy as np
from organism import Organism


class Evaluation:
    def __init__(self, config) -> None:
        self.config = config

        dist_dict = {}
        for eval_func_name, eval_func_params in config["eval_funcs"].items():
            if "name" in eval_func_params.keys():
                dist_dict[eval_func_name] = self.__get_distribution__(eval_func_params, config["network_size"])
        self.dist_dict = dist_dict


    def __get_distribution__(self, dist_info:dict, num_nodes:int) -> list[float]:
        if dist_info["name"] == "scale-free":
            gamma = dist_info["gamma"]
            offset = dist_info["offset"]
            return [(x+offset)**-gamma for x in range(num_nodes+1)]


    #node-level topological properties
    def in_degree_distribution(self, network:Organism) -> float:
        dist = self.dist_dict["in_degree_distribution"]
        degree_sequence = network.getDegreeDistribution("in_degree_distribution")
        squares = sum([(dist[i]-degree_sequence[i])**2 for i in range(network.numNodes+1)])
        return squares
    

    def out_degree_distribution(self, network:Organism) -> float:
        dist = self.dist_dict["out_degree_distribution"]
        degree_sequence = network.getDegreeDistribution("out_degree_distribution")
        squares = sum([(dist[i]-degree_sequence[i])**2 for i in range(network.numNodes+1)])
        return squares
    

    #topological properties
    def connectance(self, network:Organism) -> float:
        return network.numInteractions / network.numNodes**2


    def proportion_of_self_loops(self, network:Organism) -> float:
        return sum([1 for i in range(network.numNodes) if network.adjacencyMatrix[i][i] != 0]) / network.numNodes
    

    def diameter(self, network:Organism) -> int:
        shortest_path = dict(nx.shortest_path_length(network.getNetworkxObject()))
        return max([max(shortest_path[i].values()) for i in range(len(shortest_path))])
    

    #node-level interaction strength properties
    # def pos_in_weight_distribution(self, network:Organism) -> float:
    #     nn = network.numNodes
    #     networkx_obj = network.getNetworkxObject()
    #     degree_sequence = sorted([networkx_obj.in_degree(n)/nn for n in networkx_obj.nodes()], reverse=True)
    

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