import networkx as nx
import numpy as np
from organism import Organism


class Evaluation:
    def __init__(self, config) -> None:
        self.config = config

        target_dist_dict = {}
        for eval_func_name, eval_func_params in config["eval_funcs"].items():
            if eval_func_name.endswith("distribution"):
                target_dist_dict[eval_func_name] = self.__get_target_distribution__(eval_func_params, config["network_size"])
        self.target_dist_dict = target_dist_dict
        self.functions = {func:getattr(Evaluation, func) for func in dir(Evaluation) 
                          if callable(getattr(Evaluation, func)) and
                          not func.startswith("__")}


    def __get_target_distribution__(self, dist_info:dict, num_nodes:int) -> list[float]:
        if "name" in dist_info.keys():
            if dist_info["name"] == "scale-free":
                gamma = dist_info["gamma"]
                offset = dist_info["offset"]
                return [0]+[(x+offset)**-gamma for x in range(num_nodes)]
        if "target" in dist_info.keys():
            return dist_info["target"]
        print("Invalid distribution config")
        exit()


    #node-level topological properties
    def in_degree_distribution(self, network:Organism) -> list[float]:
        networkx_obj = network.getNetworkxObject()
        num_nodes = network.numNodes
        degree_sequence = list(d for _, d in networkx_obj.in_degree())
        freq = [0]*(num_nodes+1)
        for d in degree_sequence:
            freq[d] += 1/num_nodes
        return freq
    

    def out_degree_distribution(self, network:Organism) -> list[float]:
        networkx_obj = network.getNetworkxObject()
        num_nodes = network.numNodes
        degree_sequence = list(d for _, d in networkx_obj.out_degree())
        freq = [0]*(num_nodes+1)
        for d in degree_sequence:
            freq[d] += 1/num_nodes
        return freq


    def avg_shortest_path_length_distribution(self, network:Organism) -> list[float]:
        weight = 1/network.numNodes**2
        shortest_path = dict(nx.shortest_path_length(network.getNetworkxObject()))
        avg_shortest = sorted([weight*np.mean(list(shortest_path[i].values())) for i in range(len(shortest_path))], reverse=True)
        return avg_shortest
    

    def betweenness_distribution(self, network:Organism) -> list[float]:
        networkx_obj = network.getNetworkxObject()
        return sorted(list(nx.betweenness_centrality(networkx_obj).values()), reverse=True)
    
    
    #topological properties
    def strong_components(self, network:Organism) -> int:
        return len(list(nx.strongly_connected_components(network.getNetworkxObject())))


    def weak_components(self, network:Organism) -> int:
        return len(list(nx.weakly_connected_components(network.getNetworkxObject())))


    def connectance(self, network:Organism) -> float:
        return network.numInteractions / network.numNodes**2


    def clustering_coefficient(self, network:Organism) -> float:
        return nx.average_clustering(network.getNetworkxObject())


    def proportion_of_self_loops(self, network:Organism) -> float:
        return sum([1 for i in range(network.numNodes) if network.adjacencyMatrix[i][i] != 0]) / network.numNodes
    

    def diameter(self, network:Organism) -> int:
        shortest_path = dict(nx.shortest_path_length(network.getNetworkxObject()))
        return max([max(shortest_path[i].values()) for i in range(len(shortest_path))])


    #node-level interaction strength properties
    def pos_out_weight_distribution(self, network:Organism) -> float:
        networkx_obj = network.getNetworkxObject()
        matrix = network.adjacencyMatrix
        num_nodes = network.numNodes
        degree_sequence = networkx_obj.out_degree()
        count_weights = [0]*(num_nodes+1)
        sum_weights = [0]*(num_nodes+1)
        for n,d in degree_sequence:
            sum_weights[d] += sum([x for x in matrix[n] if x > 0])
            count_weights[d] += len([x for x in matrix[n] if x > 0])
        avg_weights = [sum_weights[i]/count_weights[i] if count_weights[i] > 0 else 0 for i in range(num_nodes+1)]
        return avg_weights


    def pos_in_weight_distribution(self, network:Organism) -> float:
        networkx_obj = network.getNetworkxObject()
        matrix_T = list(zip(*network.adjacencyMatrix))
        num_nodes = network.numNodes
        degree_sequence = networkx_obj.in_degree()
        count_weights = [0]*(num_nodes+1)
        sum_weights = [0]*(num_nodes+1)
        for n,d in degree_sequence:
            sum_weights[d] += sum([x for x in matrix_T[n] if x > 0])
            count_weights[d] += len([x for x in matrix_T[n] if x > 0])
        avg_weights = [sum_weights[i]/count_weights[i] if count_weights[i] > 0 else 0 for i in range(num_nodes+1)]
        return avg_weights


    def neg_out_weight_distribution(self, network:Organism) -> float:
        networkx_obj = network.getNetworkxObject()
        matrix = network.adjacencyMatrix
        num_nodes = network.numNodes
        degree_sequence = networkx_obj.out_degree()
        count_weights = [0]*(num_nodes+1)
        sum_weights = [0]*(num_nodes+1)
        for n,d in degree_sequence:
            sum_weights[d] += sum([x for x in matrix[n] if x < 0])
            count_weights[d] += len([x for x in matrix[n] if x < 0])
        avg_weights = [sum_weights[i]/count_weights[i] if count_weights[i] > 0 else 0 for i in range(num_nodes+1)]
        return avg_weights


    def neg_in_weight_distribution(self, network:Organism) -> float:
        networkx_obj = network.getNetworkxObject()
        matrix_T = list(zip(*network.adjacencyMatrix))
        num_nodes = network.numNodes
        degree_sequence = networkx_obj.in_degree()
        count_weights = [0]*(num_nodes+1)
        sum_weights = [0]*(num_nodes+1)
        for n,d in degree_sequence:
            sum_weights[d] += sum([x for x in matrix_T[n] if x < 0])
            count_weights[d] += len([x for x in matrix_T[n] if x < 0])
        avg_weights = [sum_weights[i]/count_weights[i] if count_weights[i] > 0 else 0 for i in range(num_nodes+1)]
        return avg_weights


    #interaction strength properties
    def positive_interactions_proportion(self, network:Organism) -> float:
        num_interactions = network.numInteractions
        return network.numPositive / num_interactions if num_interactions > 0 else 0


    def average_positive_interactions_strength(self, network:Organism) -> float:
        sum_pos = sum([sum([y for y in x if y > 0]) for x in network.adjacencyMatrix])
        num_pos = network.numPositive
        return  sum_pos / num_pos if num_pos > 0 else 0


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