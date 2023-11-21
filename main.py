from copy import deepcopy
from random import random, shuffle, sample
from typing import Callable

import matplotlib.pyplot as plt
import networkx as nx


class Organism:
    def __init__(self,numNodes:int, sparsity:float = 0.5) -> None:
        #init numNodes x numNodes matrix with 'sparsity' percent of 1s (0.9 -> 90% 1s)
        #TODO: support weighted digraphs using cliff/arend's sparsity function
        self.adjacencyMatrix:list[list[int]] = [[1 if random() <= sparsity else 0 for _ in range(numNodes)] for _ in range(numNodes)]
        #internal size reference
        self.numNodes:int = numNodes
        #evaluation memo, for possible efficiency boosts (do not access directly, use getter)
        self.evaluationScores:list[float] = []


    def makeMutatedCopy(self,mutationRate:float = 0.005):
        #inheritance
        newOrg = Organism(self.numNodes)
        newOrg.adjacencyMatrix = deepcopy(self.adjacencyMatrix)
        #variation
        for i in range(self.numNodes):
            for j in range(self.numNodes):
                if random() <= mutationRate:
                    #toggle bit (1-> 0, or 0->1)
                    newOrg.adjacencyMatrix[i][j] *= -1
                    newOrg.adjacencyMatrix[i][j] += 1
        return newOrg
    

    def getEvaluationScores(self,evaluationFunctions:list[Callable]) -> list[float]:
        #only evaluate fitness once, otherwise return memo value
        if not self.evaluationScores:
            #TODO: don't evaluate objectives unless required (optimization)
            self.evaluationScores = [func(self) for func in evaluationFunctions]
        return self.evaluationScores
        

    def getNetworkxObject(self) -> nx.DiGraph:
        G = nx.DiGraph()
        for i in range(self.numNodes):
            for j in range(self.numNodes):
                #TODO: support weighted digraphs using abs() >= 0
                if self.adjacencyMatrix[i][j] == 1:
                    #TODO: support weighted digraphs using weight=
                    G.add_edge(i,j)
        return G


    def saveGraphFigure(self,path:str):
        G = self.getNetworkxObject()
        ######################
        # grpah layout style #
        ######################
        pos = nx.nx_agraph.graphviz_layout(G)
        # pos = nx.kamada_kawai_layout(G)
        # pos = nx.spectral_layout(G)
        # pos = nx.planar_layout(G)
        # random
        # pos = nx.shell_layout(G)
        # pos = nx.spring_layout(G)
        ######################

        plt.figure(figsize=(20,20))
        plt.title("Network Topology")

        nx.draw_networkx_nodes(G, pos=pos)
        nx.draw_networkx_labels(G, pos, {node:node for node in G.nodes()}, font_size=9, font_color="k")

        nx.draw_networkx_edges(G, pos=pos) #TODO: support weighted digrpahs with weights
        # nx.draw_networkx_edge_labels(G, pos=pos) #optional edge weight labels go here

        plt.savefig(path)
        plt.close()


########################
# evaluation functions #
########################

#function stub for an evaluation function
#TODO: create real evaluation functions, one for each property of the network
#NOTE: the lexicase selection algorithm assumes we are minimizing values, therefore each evaluation function should be either:
#   1) a distance function, e.g. (val - target)**2 or
#   2) a negative value, e.g. -val (for maximization) or
#   3) a positive value, e.g. val (for minimization)
def foo(network:Organism) -> float:
    return random()


################
# helper funcs #
################

#tanspose of a matrix (list-of-list)
def T(LL:list[list]) -> list[list]:
    return list(zip(*LL))


def epsilonLexicase(population:list[Organism], numParents:int, epsilon:float = 0.05) -> list[Organism]:
    global EVAL_FUNCS, POPSIZE

    parents:list[Organism] = []
    objectiveIDs:list[int]= list(range(len(EVAL_FUNCS)))

    for _ in range(POPSIZE):
         #randomize objective evaluation order
        shuffle(objectiveIDs)
        #IDs of organisms that 'make the cut'
        cut:list[int] = [i for i in range(POPSIZE)]
        for objectiveID in objectiveIDs:
            #get best w.r.t. this objective
            minVal = min([population[i].getEvaluationScores(EVAL_FUNCS)[objectiveID] for i in cut])
            #keep only those organisms that are within epsilon of the best organism
            #TODO: instead of epsilon being a fixed offset, it could be a percentage of the best or a percentage of the range of values
            cut = [i for i in cut if population[i].getEvaluationScores(EVAL_FUNCS)[objectiveID] <= minVal+epsilon]
            if len(cut) == 1:
                parents.append(population[cut[0]])
                break
        #if choices remain after all objectives, choose randomly
        parents.append(population[sample(cut,k=1)[0]])

    return parents


if __name__ == "__main__":
    POPSIZE = 100
    MUTATION_RATE = 0.005
    NETWORK_SIZE = 10
    NETWORK_SPARSITY = 0.5
    NUM_GENERATIONS = 500
    EVAL_FUNCS = [foo,]

    population = [Organism(NETWORK_SIZE,NETWORK_SPARSITY) for _ in range(POPSIZE)]

    for gen in range(NUM_GENERATIONS):
        print("Gen",gen)
        parents = epsilonLexicase(population,POPSIZE)
        children = [parent.makeMutatedCopy(MUTATION_RATE) for parent in parents]
        population = children
    print([x.getEvaluationScores() for x in population])