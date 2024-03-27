from random import random, sample
from statistics import mean

import numpy as np

from organism import Organism

#multi-objective map-elites based on:
#"Illuminating search spaces by mapping elites" (Mouret & Clune, 2015)
#"Multi-Objective Quality Diversity Optimization" (Pierrot et al., 2023)


def bin_value(bins, value):
    return bins[(np.abs(bins - value)).argmin()]


def pareto_front(population):
    population = [x for x in population if x is not None]
    F = []
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
            F.append(p)
    return F


def crowding_distance_assignment(I:list[Organism]):
    l = len(I)
    if l == 0: return []
    for i in I:
        i.nsga_distance = 0
    for m in I[0].errors.keys():
        I.sort(key=lambda org: org.errors[m])
        I[0].nsga_distance = float("inf")
        I[-1].nsga_distance = float("inf")
        rng = I[-1].errors[m]-I[0].errors[m]
        if rng == 0: continue
        for i in range(1,l-2):
            I[i].nsga_distance += (I[i+1].errors[m]-I[i-1].errors[m])/rng


def run(config):
    objectives = config["eval_funcs"]
    features = config["diversity_funcs"]
    pareto_size = config["pareto_size"]
    mutation_rate = config["mutation_rate"]
    mutation_odds = config["mutation_odds"]
    crossover_rate = config["crossover_rate"]
    crossover_odds = config["crossover_odds"]
    features = {name:np.asarray(bins) for name,bins in features.items()}
    bin_to_cell = {name:{x:i for i,x in enumerate(features[name])} for name in features}

    #initalize elites map
    elites_map = np.ndarray(shape=[len(x) for x in features.values()]+[pareto_size], dtype=Organism)
    elites_map_size = elites_map.size
    fitnessLog = {funcName:[] for funcName in objectives}
    coverage = []

    for gen in range(1, config["num_generations"]+1):
        #list of all organisms in elites map
        elites_map_flattened = elites_map.flatten()
        orgs_in_map = elites_map_flattened[elites_map_flattened != np.array(None)]

        #get new organism to potentially place in map
        if gen < config["initial_popsize"]:
            org = Organism(config["network_size"], random(), config["weight_range"])
        else:
            porg1, porg2 = sample(list(orgs_in_map), 2)
            org = porg1.makeCrossedCopyWith(porg2, crossover_rate, crossover_odds).makeMutatedCopy(mutation_rate, mutation_odds)
        [org.getError(name, target) for name, target in objectives.items()]

        #get the organism's value for each feature, round that value to the nearest bin, convert the bin into its elites map index
        cell_idx = [bin_to_cell[name][bin_value(features[name], org.getProperty(name))] for name in features]
        #calculate pareto front of cell when including the new organism
        curr_cell = elites_map[*cell_idx]
        curr_cell_with_new_org = np.append(curr_cell, org)
        new_front = pareto_front(curr_cell_with_new_org)
        #replace the cell with the new pareto front
        if len(new_front) > pareto_size:
            crowding_distance_assignment(new_front)
            new_front.sort(key=lambda org: org.nsga_distance,reverse=True)
            new_cell = new_front[:pareto_size]
        elif len(new_front) < pareto_size:
            new_cell = np.pad(new_front, (0,pareto_size-len(new_front)), "constant", constant_values=None)
        else:
            new_cell = new_front
        elites_map[*cell_idx] = new_cell

        #statistics over time
        if gen % 10 == 0:
            print("Gen: ", gen)
            for name, target in objectives.items():
                popFitnesses = [org.getError(name, target) for org in orgs_in_map]
                fitnessLog[name].append(mean(popFitnesses))
            coverage.append(len(orgs_in_map)/elites_map_size)

    elites_map_flattened = elites_map.flatten()
    population = elites_map_flattened[elites_map_flattened != np.array(None)]
    return population, fitnessLog, coverage, elites_map