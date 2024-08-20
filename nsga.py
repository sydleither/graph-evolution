from collections import Counter
from random import random, sample
from statistics import mean

from organism import Organism
from plot_utils import fast_non_dominated_sort


#transpose of a matrix (list-of-list)
def T(LL:list[list]) -> list[list]:
    return list(zip(*LL))


def run(config):
    popsize = config["popsize"]
    objectives = config["eval_funcs"]
    track_diversity_over = config["track_diversity_over"]

    population = [Organism(config["network_size"], random(), config["weight_range"]) for _ in range(popsize)]
    fitnessLog = {funcName:[] for funcName in objectives}
    diversityLog = {o:[] for o in track_diversity_over}

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
        _ = [org.getError(name, target) for org in children]

    for gen in range(config["num_generations"]):
        print("Generation", gen)

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
            F[i].sort(key=lambda org: org.nsga_distance, reverse=True)
            P.extend(F[i][:popsize-len(P)])
        population = P
        parents = nsga_tournament(population, 2*popsize)
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
        for name in track_diversity_over:
            spread = len(Counter([org.getProperty(name) for org in population]))
            diversityLog[name].append(spread)

    return population, fitnessLog, diversityLog


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