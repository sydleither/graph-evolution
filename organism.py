from copy import deepcopy
from random import randint, random
from typing import Callable

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


# this function is based on:
# Clifford Bohm, Arend Hintze, Jory Schossau; July 24–28, 2023.
# "A Simple Sparsity Function to Promote Evolutionary Search." 
# Proceedings of the ALIFE 2023: Ghost in the Machine: 
# Proceedings of the 2023 Artificial Life Conference. 
# ALIFE 2023: Ghost in the Machine: Proceedings of the 2023 Artificial Life Conference. 
# Online. (pp. 53). ASME. https://doi.org/10.1162/isal_a_00655
def sparsify(x, percentSparse:float = 0.5, outputRange:tuple[float]=(-1,1)):
    assert 0 <= percentSparse <= 1
    assert outputRange[0] <= 0 <= outputRange[1]
    assert outputRange[0] != 0 or outputRange[1] != 0

    percentNonZero = 1-percentSparse
    neg = abs(outputRange[0])
    pos = outputRange[1]
    negPercent = neg/(pos+neg)
    posPercent = pos/(pos+neg)
    a = negPercent*percentNonZero
    b = posPercent*percentNonZero
    t1 = a
    t2 = a + percentSparse

    if x <= 0:
        return -neg
    if 0 < x <= t1:
        return (neg/a)*x- neg
    if t1 < x <= t2:
        return 0
    if t2 < x <= 1:
        return (pos/b)*(x-t2)
    if 1 < x:
        return pos


class Organism:
    def __init__(self, numNodes:int, sparsity:float, weightRange, genome:list[list[float]]=None) -> None:
        if genome is None:
            self.genotypeMatrix:list[list[float]] = [[random() for _ in range(numNodes)] for _ in range(numNodes)]
        else:
            self.genotypeMatrix:list[list[float]] = genome

        self.numNodes:int = numNodes
        self.sparsity = sparsity
        self.weightRange = weightRange

        self.evaluationScores:dict[str:float] = {}

        self.adjacencyMatrix:list[list[float]] = [[sparsify(val,self.sparsity,self.weightRange) for val in row] for row in self.genotypeMatrix]

        #internal number of interactions reference
        self.numInteractions:int = sum([sum([1 for val in row if val != 0]) for row in self.adjacencyMatrix])
        #internal number of positive interactions reference
        self.numPositive:int = sum([sum([1 for val in row if val > 0]) for row in self.adjacencyMatrix])


    def makeMutatedCopy(self, mutationRate:float, mutationOdds:tuple[int]):
        #setup
        mutationThresholds = [sum(mutationOdds[:k+1]) for k in range(len(mutationOdds))]
        #inheritance
        newGenome = deepcopy(self.genotypeMatrix)
        #variation
        for i in range(self.numNodes):
            for j in range(self.numNodes):
                if random() <= mutationRate:
                    mutationType = randint(1,sum(mutationOdds))
                    if mutationType <= mutationThresholds[0]:
                        #point mutation
                        newGenome[i][j] = random()
                    elif mutationType <= mutationThresholds[1]:
                        #offset mutation
                        offset = (random()/4)-(1/8) #-1/8 to 1/8
                        newGenome[i][j] = min(1,max(0,newGenome[i][j] + offset))
                    elif mutationType <= mutationThresholds[2]:
                        newGenome[i][j] = 1-newGenome[i][j]
                    else:
                        print("ERROR: no mutation selected")
                        exit(1)
        return Organism(self.numNodes,self.sparsity,self.weightRange,newGenome)


    def makeCrossedCopyWith(self, other, rateFromOther):
        #for now, crossover occurs on the node level
        #inheritance
        newGenome = deepcopy(self.genotypeMatrix)
        #crossover
        for i in range(self.numNodes):
            if random() <= rateFromOther:
                newGenome[i] = deepcopy(other.genotypeMatrix[i])
        return Organism(self.numNodes,self.sparsity,self.weightRange,newGenome)


    def getEvaluationScores(self, evaluationDict:dict[str:tuple[Callable,float]]) -> dict[str:float]:
        for name, evaluationPack in evaluationDict.items():
            evalFunc, targetValue = evaluationPack
            if name not in self.evaluationScores:
                self.evaluationScores[name] = (evalFunc(self) - targetValue)**2
        return self.evaluationScores


    def getNetworkxObject(self) -> nx.DiGraph:
        G = nx.DiGraph(np.array(self.adjacencyMatrix))
        return G


    def saveGraphFigure(self, path:str):
        G = self.getNetworkxObject()
        ######################
        # grpah layout style #
        ######################
        # pos = nx.nx_agraph.graphviz_layout(G)
        # pos = nx.kamada_kawai_layout(G)
        # pos = nx.spectral_layout(G)
        # pos = nx.planar_layout(G)
        # random
        pos = nx.shell_layout(G)
        # pos = nx.spring_layout(G)
        ######################

        plt.figure(figsize=(20,20))
        plt.title("Network Topology")

        nx.draw_networkx_nodes(G, pos=pos)
        nx.draw_networkx_labels(G, pos,
                                {node:node for node in G.nodes()},
                                font_size=9,
                                font_color="k")
        
        weights = nx.get_edge_attributes(G, 'weight').values()
        nx.draw_networkx_edges(G, pos=pos, edge_color=weights, width=5, edge_cmap=plt.cm.PuOr, edge_vmin=-1, edge_vmax=1)
        nx.draw_networkx_edge_labels(G, pos=pos,
                                     edge_labels={(n1,n2):round(data['weight'],3) for n1,n2,data in G.edges(data=True)},
                                     label_pos=0.8)

        plt.savefig(path)
        plt.close()

    ###########################
    #pareto sorting functions #
    ###########################
    def __gt__(self,other):
        if not isinstance(other,Organism):
            raise TypeError("Invalid comparison of organism to",type(other))
        meInYou = all([myKey in other.evaluationScores.keys() for myKey in self.evaluationScores.keys()])
        youInMe = all([theirKey in self.evaluationScores.keys() for theirKey in other.evaluationScores.keys()])
        if meInYou and youInMe:
            #NOTE: potential confusion, gtr defines 'better' based on having smallest score
            return all([self.evaluationScores[prop] < other.evaluationScores[prop] for prop in self.evaluationScores.keys()])
        else:
            raise Exception("Organisms must be evaluated on the same criteria.",self.evaluationScores.keys(),other.evaluationScores.keys())
        
    def __eq__(self, other):
        return not (self < other or self > other)
