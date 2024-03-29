from collections import deque
from copy import deepcopy
from random import randint, random, sample

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

import eval_functions as ef


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
    nextID = 0

    def __init__(self, numNodes:int, sparsity:float, weightRange, genome:list[list[float]]=None) -> None:
        self.id = Organism.nextID
        Organism.nextID += 1
        self.nsga_rank = None
        self.nsga_distance = None

        if genome is None:
            self.genotypeMatrix:list[list[float]] = [[random() for _ in range(numNodes)] for _ in range(numNodes)]
        else:
            self.genotypeMatrix:list[list[float]] = genome

        self.numNodes:int = numNodes
        self.sparsity = sparsity
        self.weightRange = weightRange

        self.errors:dict[str:float] = {}
        self.properties:dict = {}

        self.adjacencyMatrix:list[list[float]] = [[round(sparsify(val,self.sparsity,self.weightRange), 3) for val in row] for row in self.genotypeMatrix]

        #internal number of interactions reference
        self.numInteractions:int = sum([sum([1 for val in row if val != 0]) for row in self.adjacencyMatrix])
        #internal number of positive interactions reference
        self.numPositive:int = sum([sum([1 for val in row if val > 0]) for row in self.adjacencyMatrix])
        #internal number of negative interactions reference
        self.numNegative:int = sum([sum([1 for val in row if val < 0]) for row in self.adjacencyMatrix])


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
                        #sign flip mutation
                        newGenome[i][j] = 1-newGenome[i][j]
                    elif mutationType <= mutationThresholds[3]:
                        #weight swap mutation (preserves connectance)
                        edge1= newGenome[i][j]
                        iswap = randint(0,self.numNodes-1)
                        jswap = randint(0,self.numNodes-1)
                        newGenome[i][j] = newGenome[iswap][jswap]
                        newGenome[iswap][jswap] = edge1
                    else:
                        print("ERROR: no mutation selected")
                        exit(1)
        # sometimes add random offset to sparsity and clamp to [0,1]
        newSparsity = self.sparsity if random() > mutationRate else min(1,max(0,self.sparsity + (random()/4)-(1/8)))
        return Organism(self.numNodes,newSparsity,self.weightRange,newGenome)

    
    def xover_traversal_helper(self, other, rateFromOther, algorithm):
        N = len(other.genotypeMatrix)
        if algorithm == "DFS":
            visited = []
        elif algorithm == "BFS":
            visited = deque()
        else: raise Exception("invalid traversal algorithm")
        completed = []
        available = [i for i in range(N)]
        while random() <= rateFromOther and len(completed) < N and available:
            currentNode = sample(available,k=1)[0]
            visited.append(currentNode)
            available = [i for i in range(N)
                            if other.adjacencyMatrix[currentNode][i] != 0 and
                            i not in visited and
                            i not in completed]
            while not available and len(completed) < N and visited:
                completed.append(currentNode)
                if algorithm == "DFS":
                    currentNode = visited.pop()
                elif algorithm == "BFS":
                    currentNode = visited.popleft()
                available = [i for i in range(N)
                            if other.adjacencyMatrix[currentNode][i] != 0 and
                            i not in visited and
                            i not in completed]
        return list(visited) + completed


    def makeCrossedCopyWith(self, other, rateFromOther, crossOdds:tuple[int]):
        #setup
        crossoverThresholds = [sum(crossOdds[:k+1]) for k in range(len(crossOdds))]
        crossoverType = randint(1, sum(crossOdds))
        #inheritance
        newGenome = deepcopy(self.genotypeMatrix)
        #crossover
        if crossoverType <= crossoverThresholds[0]:
            #binary node crossover
            for i in range(self.numNodes):
                if random() <= rateFromOther:
                    newGenome[i] = deepcopy(other.genotypeMatrix[i])
        elif crossoverType <= crossoverThresholds[1]:
            #depth-first-traversal crossover
            crossNodes = self.xover_traversal_helper(other, rateFromOther, "DFS")
            for i in crossNodes:
                newGenome[i] = deepcopy(other.genotypeMatrix[i])
        elif crossoverType <= crossoverThresholds[2]:
            #breadth-first-traversal crossover
            crossNodes = self.xover_traversal_helper(other, rateFromOther, "BFS")
            for i in crossNodes:
                newGenome[i] = deepcopy(other.genotypeMatrix[i])
        return Organism(self.numNodes, self.sparsity, self.weightRange, newGenome)


    def getProperty(self, propertyName:str):
        if propertyName not in self.properties:
            self.properties[propertyName] = ef.functions[propertyName](None,self)
        return self.properties[propertyName]

    
    def getError(self, propertyName:str, target) -> float:
        if propertyName not in self.errors:
            if propertyName.endswith("_weight_distribution"):
                dist = self.getProperty(propertyName)
                self.errors[propertyName] = sum([(dist[i]-target[i])**2 for i in range(len(dist)) if dist[i] != 0])
            elif propertyName.endswith("_distribution"):
                dist = self.getProperty(propertyName)
                self.errors[propertyName] = sum([(dist[i]-target[i])**2 for i in range(len(dist))])
            else:
                self.errors[propertyName] = (self.getProperty(propertyName) - target)**2
        return self.errors[propertyName]


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
    def __gt__(self, other):
        if not isinstance(other,Organism):
            raise TypeError("Invalid comparison of organism to",type(other))
        meInYou = all([myKey in other.errors.keys() for myKey in self.errors.keys()])
        youInMe = all([theirKey in self.errors.keys() for theirKey in other.errors.keys()])
        if meInYou and youInMe:
            #NOTE: potential confusion, gtr defines 'better' based on having smallest score
            noSelfWorse = all([self.errors[prop] <= other.errors[prop] for prop in self.errors.keys()])
            someSelfBetter = any([self.errors[prop] < other.errors[prop] for prop in self.errors.keys()])
            return noSelfWorse and someSelfBetter
        else:
            raise Exception("Organisms must be evaluated on the same criteria.",self.errors.keys(),other.errors.keys())


    def __eq__(self, other):
        if other is None:
            return False
        return not (self < other or self > other)
