from collections import Counter
from random import random, sample
from statistics import mean

from organism import Organism


#transpose of a matrix (list-of-list)
def T(LL:list[list]) -> list[list]:
    return list(zip(*LL))


def run(config):
    popsize = config["popsize"]
    objectives = config["eval_funcs"]

    population = [Organism(config["network_size"], random(), config["weight_range"]) for _ in range(popsize)]
    fitnessLog = {funcName:[] for funcName in objectives}
    diversityLog = {"connectance": [], "clustering_coefficient": [], "positive_interactions_proportion":[], "average_positive_interactions_strength":[]}

    #Algorithm from: Deb, Kalyanmoy, et al.
    #"A fast and elitist multiobjective genetic algorithm: NSGA-II."
    #IEEE transactions on evolutionary computation 6.2 (2002): 182-197.

    #eval parents for initial fitness values
    for name, target in objectives.items():
        popFitnesses = [org.getError(name, target) for org in population]
        fitnessLog[name].append(mean(popFitnesses))

    #init some random children
    children = [Organism(config["network_size"], random(), config["weight_range"]) for _ in range(popsize)]

    #evaluate all children
    for name, target in objectives.items():
        _ = [org.getError(name,target) for org in children]

    for gen in range(config["num_generations"]):
        print("Gen:",gen)
        #begin of selection
        R = population+children
        F = fast_non_dominated_sort(R)
        P = []
        i = 1
        while len(P) + len(F[i]) <= popsize:
            nsga_distance_assignment(F[i])
            P.extend(F[i])
            i += 1
        if len(P) < popsize:
            nsga_distance_assignment(F[i])
            F[i].sort(key=lambda org: org.nsga_distance,reverse=True)
            P.extend(F[i][:popsize-len(P)])
        population = P
        parents = nsga_tournament(population,2*popsize)
        #end of selection

        #repoduction
        children = [parents[i].makeCrossedCopyWith(
            parents[i+popsize],config["crossover_rate"], config["crossover_odds"]).makeMutatedCopy(
            config["mutation_rate"], config["mutation_odds"]) for i in range(popsize)]

        #evaluation
        for name, target in objectives.items():
            popFitnesses = [org.getError(name, target) for org in population]
            fitnessLog[name].append(mean(popFitnesses))
            _ = [org.getError(name,target) for org in children]
        connectance_spread = len(Counter([org.getProperty("connectance") 
                                          for org in population]))
        cc_spread = len(Counter([org.getProperty("clustering_coefficient") 
                                  for org in population]))
        pip_spread = len(Counter([org.getProperty("positive_interactions_proportion") 
                                  for org in population]))
        apis_spread = len(Counter([org.getProperty("average_positive_interactions_strength") 
                                  for org in population]))
        diversityLog["connectance"].append(connectance_spread)
        diversityLog["clustering_coefficient"].append(cc_spread)
        diversityLog["positive_interactions_proportion"].append(pip_spread)
        diversityLog["average_positive_interactions_strength"].append(apis_spread)

    return population, fitnessLog, diversityLog


#Algorithm from: Deb, Kalyanmoy, et al.
#"A fast and elitist multiobjective genetic algorithm: NSGA-II."
#IEEE transactions on evolutionary computation 6.2 (2002): 182-197.
def fast_non_dominated_sort(population):
    F = {1:[]}
    S = {}
    n = {}
    for p in population:
        S[p.id] = []
        n[p.id] = 0
        for q in population:
            if p > q:
                S[p.id].append(q)
            elif q > p:
                n[p.id] += 1
        if n[p.id] == 0:
            p.nsga_rank = 1
            F[1].append(p)
    i = 1
    while len(F[i]) > 0:
        Q = []
        for p in F[i]:
            for q in S[p.id]:
                n[q.id] -= 1
                if n[q.id] == 0:
                    q.nsga_rank = i+1
                    Q.append(q)
        i += 1
        F[i] = Q[:]
    return F


#Algorithm from: Deb, Kalyanmoy, et al.
#"A fast and elitist multiobjective genetic algorithm: NSGA-II."
#IEEE transactions on evolutionary computation 6.2 (2002): 182-197.
def nsga_distance_assignment(I:list[Organism]):
    l = len(I)
    if l == 0: return []
    for i in I:
        i.nsga_distance = 0

    num_obj = len(I[0].errors.keys())
    for m in I[0].errors.keys():
        I.sort(key=lambda org: org.errors[m])
        rng = I[-1].errors[m]-I[0].errors[m]
        I[0].nsga_distance = rng/num_obj
        I[-1].nsga_distance = rng/num_obj
        if rng == 0: continue
        for i in range(1,l-2):
            I[i].nsga_distance += (I[i+1].errors[m]-I[i-1].errors[m])/(rng*num_obj)

    # genotype matrix diversity
    num_vals = I[0].numNodes**2
    for j in range(I[0].numNodes):
        for k in range(I[0].numNodes):
            I.sort(key=lambda org: org.genotypeMatrix[j][k])
            rng = I[-1].genotypeMatrix[j][k] - I[0].genotypeMatrix[j][k]
            I[0].nsga_distance = rng/num_vals
            I[-1].nsga_distance = rng/num_vals
            if rng == 0: continue
            for i in range(1,l-2):
                I[i].nsga_distance += (I[i+1].genotypeMatrix[j][k]-I[i-1].genotypeMatrix[j][k])/(rng*num_vals)

    # sparsity diversity
    I.sort(key=lambda org: org.sparsity)
    rng = I[-1].sparsity - I[0].sparsity
    I[0].nsga_distance = rng
    I[-1].nsga_distance = rng
    if rng != 0:
        for i in range(1,l-2):
            I[i].nsga_distance += (I[i+1].sparsity-I[i-1].sparsity)/rng


#Algorithm from: Deb, Kalyanmoy, et al.
#"A fast and elitist multiobjective genetic algorithm: NSGA-II."
#IEEE transactions on evolutionary computation 6.2 (2002): 182-197.
def nsga_tournament(population, numOffspring):
    parents = []
    for _ in range(numOffspring):
        if random() < 0.25:
            choices = sample(population, k=2)
            if choices[0].nsga_rank > choices[1].nsga_rank:
                parents.append(choices[1])
            elif choices[1].nsga_rank > choices[0].nsga_rank:
                parents.append(choices[0])
            elif choices[0].nsga_distance < choices[1].nsga_distance:
                parents.append(choices[1])
            elif choices[1].nsga_distance < choices[0].nsga_distance:
                parents.append(choices[0])
            else:
                parents.append(sample(choices, k=1)[0])
        else:
            parents.append(sample(population, k=1)[0])
    return parents