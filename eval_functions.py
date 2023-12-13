import networkx as nx
import numpy as np
from organism import Organism


class Evaluation:
    def __init__(self, config) -> None:
        self.config = config

        dist_dict = {}
        for eval_func_name, eval_func_params in config["eval_funcs"].items():
            if "name" in eval_func_params.keys():
                dist_dict[eval_func_name] = self.get_distribution(eval_func_params, config["network_size"])
        self.dist_dict = dist_dict


    def get_distribution(self, dist_info:dict, num_nodes:int) -> list[float]:
        if dist_info["name"] == "scale-free":
            gamma = dist_info["gamma"]
            return [x**-gamma for x in [(y/num_nodes)+1 for y in range(num_nodes)]]


    #node-level topological properties
    def in_degree_distribution(self, network:Organism) -> float:
        nn = network.numNodes
        networkx_obj = network.getNetworkxObject()
        dist = self.dist_dict["in_degree_distribution"]
        degree_sequence = sorted([networkx_obj.in_degree(n)/nn for n in networkx_obj.nodes()], reverse=True)
        squares = sum([(dist[i]-degree_sequence[i])**2 for i in range(nn)])
        return squares
    

    def out_degree_distribution(self, network:Organism) -> float:
        nn = network.numNodes
        networkx_obj = network.getNetworkxObject()
        dist = self.dist_dict["out_degree_distribution"]
        degree_sequence = sorted([networkx_obj.out_degree(n)/nn for n in networkx_obj.nodes()], reverse=True)
        squares = sum([(dist[i]-degree_sequence[i])**2 for i in range(nn)])
        return squares


    def powerlaw_weight_distribution(self, network:Organism) -> float:
        nn = network.numNodes
        networkx_obj = network.getNetworkxObject()
        beta = self.config["eval_funcs"]["powerlaw_weight_distribution"]["beta"]
        degree_sequence = sorted([networkx_obj.degree(n)/nn for n in networkx_obj.nodes()], reverse=True)
        dist = [x**beta for x in degree_sequence]
        in_weights = [sum([abs(val) for val in row]) for row in network.adjacencyMatrix]
        out_weights = [sum([abs(row[col]) for row in network.adjacencyMatrix]) for col in range(nn)]
        weight_sequence = sorted([(in_weights[i]+out_weights[i]-network.adjacencyMatrix[i][i])/nn for i in range(nn)], reverse=True)
        squares = sum([(dist[i]-weight_sequence[i])**2 for i in range(nn)])
        return squares
    

    #topological properties
    def connectance(self, network:Organism) -> float:
        return network.numInteractions / network.numNodes**2


    def proportion_of_self_loops(self, network:Organism) -> float:
        return sum([1 for i in range(network.numNodes) if network.adjacencyMatrix[i][i] != 0]) / network.numNodes
    

    def diameter(self, network:Organism) -> int:
        shortest_path = dict(nx.shortest_path_length(network.getNetworkxObject()))
        return max([max(shortest_path[i].values()) for i in range(len(shortest_path))])
    

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