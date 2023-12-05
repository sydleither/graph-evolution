from copy import deepcopy
from random import random, sample, shuffle
from statistics import mean
from typing import Callable

import matplotlib.pyplot as plt
import networkx as nx


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
    def __init__(self, numNodes:int, sparsity:float=0.5, weightRange=(-1,1)) -> None:
        #init numNodes x numNodes matrix with 'sparsity' percent of 0s, and weights in weightRange.
        self.adjacencyMatrix:list[list[int]] = [[sparsify(random(),percentSparse=sparsity, outputRange=weightRange) 
                                                 for _ in range(numNodes)] for _ in range(numNodes)]
        #internal size reference
        self.numNodes:int = numNodes
        #internal number of interactions reference
        self.numInteractions:int = sum([sum([1 for y in x if y != 0]) for x in self.adjacencyMatrix])
        #internal number of positive interactions reference
        self.numPositive:int = sum([sum([1 for y in x if y > 0]) for x in self.adjacencyMatrix])
        #evaluation memo, for possible efficiency boosts (do not access directly, use getter)
        self.evaluationScores:dict[str:float] = {}
        self.sparsity = sparsity
        self.weightRange = weightRange


    def makeMutatedCopy(self, mutationRate:float = 0.005):
        #inheritance
        newOrg = Organism(self.numNodes)
        newOrg.adjacencyMatrix = deepcopy(self.adjacencyMatrix)
        #variation
        for i in range(self.numNodes):
            for j in range(self.numNodes):
                if random() <= mutationRate:
                    newOrg.adjacencyMatrix[i][j] = sparsify(random(), percentSparse=self.sparsity, outputRange=self.weightRange) 
        return newOrg
    

    def getEvaluationScores(self, evaluationDict:dict[str:tuple[Callable,float]]) -> dict[str:float]:
        for name, evaluationPack in evaluationDict.items():
            evalFunc, targetValue = evaluationPack
            if name not in self.evaluationScores:
                self.evaluationScores[name] = (evalFunc(self) - targetValue)**2
        return self.evaluationScores
            
        
    def getNetworkxObject(self) -> nx.DiGraph:
        G = nx.DiGraph()
        for i in range(self.numNodes):
            for j in range(self.numNodes):
                if abs(self.adjacencyMatrix[i][j]) > 0:
                    G.add_edge(i,j,weight=self.adjacencyMatrix[i][j])
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


########################
# evaluation functions #
########################
def connectance(network:Organism) -> float:
    return network.numInteractions / network.numNodes**2


def positive_interactions_proportion(network:Organism) -> float:
    return network.numPositive / network.numInteractions


def average_positive_interactions_strength(network:Organism) -> float:
    return sum([sum([y for y in x if y > 0]) for x in network.adjacencyMatrix]) / network.numPositive


def proportion_of_self_loops_positive(network:Organism) -> float:
    return sum([1 for i in range(network.numNodes) if network.adjacencyMatrix[i][i] > 0]) / network.numNodes


def number_of_mutualistic_pairs(network:Organism) -> float:
    adj = network.adjacencyMatrix
    nn = network.numNodes
    return sum([sum([1 for j in range(i+1, nn) if adj[i][j] > 0 and adj[j][i] > 0]) for i in range(nn)])


def number_of_competiton_pairs(network:Organism) -> float:
    adj = network.adjacencyMatrix
    nn = network.numNodes
    return sum([sum([1 for j in range(i+1, nn) if adj[i][j] < 0 and adj[j][i] < 0]) for i in range(nn)])


def number_of_parasitism_pairs(network:Organism) -> float:
    adj = network.adjacencyMatrix
    nn = network.numNodes
    return sum([sum([1 for j in range(i+1, nn) if (adj[i][j] < 0 and adj[j][i] > 0) or (adj[i][j] > 0 and adj[j][i] < 0)]) for i in range(nn)])


################
# helper funcs #
################

#transpose of a matrix (list-of-list)
def T(LL:list[list]) -> list[list]:
    return list(zip(*LL))


def epsilonLexicase(population:list[Organism], numParents:int, epsilon:float = 0.05) -> list[Organism]:
    global EVAL_FUNCS, POPSIZE

    parents:list[Organism] = []
    objectiveNames:list[str] = list(EVAL_FUNCS.keys())

    for _ in range(numParents):
        shuffle(objectiveNames) #randomize objective evaluation order
        cut:list[int] = [i for i in range(POPSIZE)] #IDs of organisms that 'make the cut'

        for name in objectiveNames:
            minVal = min([population[i].getEvaluationScores({name:EVAL_FUNCS[name]})[name] for i in cut]) #get best w.r.t. this objective
            #keep only those organisms that are within epsilon of the best organism
            cut = [i for i in cut if population[i].getEvaluationScores({name:EVAL_FUNCS[name]})[name] <= minVal*(1+epsilon)]
            if len(cut) == 1:
                parents.append(population[cut[0]])
                break
        parents.append(population[sample(cut,k=1)[0]]) #if choices remain after all objectives, choose randomly

    return parents


if __name__ == '__main__':
    POPSIZE = 100
    MUTATION_RATE = 0.005
    NETWORK_SIZE = 10
    NETWORK_SPARSITY = 0.1
    NUM_GENERATIONS = 100
    EVAL_FUNCS:dict[str:tuple[Callable,float]] = {
        "connectance":(connectance, 0.5), 
        "positive_interactions_proportion":(positive_interactions_proportion, 0.5),
        "average_positive_interactions_strength":(average_positive_interactions_strength, 0.25),
        "number_of_mutualistic_pairs":(number_of_mutualistic_pairs, 4),
        "number_of_competiton_pairs":(number_of_competiton_pairs, 2)
    }
    eval_funcs_names = EVAL_FUNCS.keys()

    population = [Organism(NETWORK_SIZE, NETWORK_SPARSITY) for _ in range(POPSIZE)]
    fitnessLog = {x:[] for x in eval_funcs_names}
    for gen in range(NUM_GENERATIONS):
        print("Gen",gen)
        parents = epsilonLexicase(population,POPSIZE)
        children = [parent.makeMutatedCopy(MUTATION_RATE) for parent in parents]
        for func_name, funcPack in EVAL_FUNCS.items():
            func_fitnesses = [org.getEvaluationScores({func_name:funcPack})[func_name] for org in population]
            fitnessLog[func_name].append(mean(func_fitnesses))
        population = children

    population[0].saveGraphFigure("testFigure.png")

    for func_name in eval_funcs_names:
        plt.plot(fitnessLog[func_name], label=func_name)
    plt.legend()
    plt.xlabel("Generations")
    plt.ylabel("MSE")
    plt.savefig("selectionTest.png")
    plt.close()
