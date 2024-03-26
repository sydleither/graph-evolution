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


def run(config):
    objectives = config["eval_funcs"]
    features = config["diversity_funcs"]
    pareto_size = config["pareto_size"]
    features = {name:np.asarray(bins) for name,bins in features.items()}
    bin_to_cell = {name:{x:i for i,x in enumerate(features[name])} for name in features}

    #initalize elites map and population
    elites_map = np.ndarray(shape=[len(x) for x in features.values()]+[pareto_size], dtype=Organism)
    population = [Organism(config["network_size"], random(), config["weight_range"]) for _ in range(elites_map.size)]
    fitnessLog = {funcName:[] for funcName in objectives}

    #get fitness values
    for name, target in objectives.items():
        popFitnesses = [org.getError(name, target) for org in population]
        fitnessLog[name].append(mean(popFitnesses))

    #insert population into elites map
    for org in population:
        #get the organism's value for each feature, round that value to the nearest bin, convert the bin into its elites map index
        cell_idx = [bin_to_cell[name][bin_value(features[name], org.getProperty(name))] for name in features]
        #calculate pareto front of cell with new organism
        curr_cell = elites_map[*cell_idx]
        curr_cell_with_new_org = np.append(curr_cell, org)
        new_front = pareto_front(curr_cell_with_new_org)
        #replace the cell with the new pareto front
        if len(new_front) > pareto_size:
            new_cell = sample(new_front, pareto_size)
        elif len(new_front) < pareto_size:
            new_cell = np.pad(new_front, (0,pareto_size-len(new_front)), "constant", constant_values=None)
        else:
            new_cell = new_front
        elites_map[*cell_idx] = new_cell
    exit()