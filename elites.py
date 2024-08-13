from hashlib import md5
from itertools import product
from random import random, sample
from statistics import mean
import struct

import numpy as np

from organism import Organism

#multi-objective map-elites based on:
#"Illuminating search spaces by mapping elites" (Mouret & Clune, 2015)
#"Multi-Objective Quality Diversity Optimization" (Pierrot et al., 2023)


def get_features_dict(hash_resolution):
    return {"sparsity":np.linspace(0, 1, 11)}


def get_orgs_in_map(elites_map):
    for cell in elites_map.values():
        for org in cell:
            yield org


def bin_value(bins, value):
    return bins[(np.abs(bins - value)).argmin()]


def first_front(population):
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


def crowding_distance_assignment(I:list[Organism]):
    l = len(I)
    if l == 0: return
    for i in I:
        i.nsga_distance = 0
    for m in I[0].errors.keys():
        I.sort(key=lambda org: org.errors[m])
        I[0].nsga_distance = float("inf")
        I[-1].nsga_distance = float("inf")
        rng = I[-1].errors[m] - I[0].errors[m]
        if rng == 0: continue
        for i in range(1,l-2):
            I[i].nsga_distance += (I[i+1].errors[m]-I[i-1].errors[m])/rng


def run(config):
    #extract relevant config information for efficiency
    objectives = config["eval_funcs"]
    cell_capacity = config["cell_capacity"]
    mutation_rate = config["mutation_rate"]
    mutation_odds = config["mutation_odds"]
    crossover_rate = config["crossover_rate"]
    crossover_odds = config["crossover_odds"]
    hash_resolution = config["hash_resolution"]
    features = get_features_dict(hash_resolution)
    feature_bins = list(features.values())
    #initalize elites map
    elites_map = {x:[] for x in product(*feature_bins)}
    elites_map_max_size = len(elites_map) * cell_capacity
    cells_with_orgs = []
    orgs_to_place = []
    #initalize tracking performance over time
    fitnessLog = {funcName:[] for funcName in objectives}
    coverage = []

    for gen in range(1, config["num_generations"]+1):
        #get new organism to potentially place in map
        if gen < config["initial_popsize"]:
            org = Organism(config["network_size"], random(), config["weight_range"])
        else:
            pcell_idx_1, pcell_idx_2 = sample(cells_with_orgs, 2)
            porg1 = sample(elites_map[pcell_idx_1], 1)[0]
            porg2 = sample(elites_map[pcell_idx_2], 1)[0]
            org = porg1.makeCrossedCopyWith(porg2, crossover_rate, crossover_odds).makeMutatedCopy(mutation_rate, mutation_odds)
        [org.getError(name, target) for name, target in objectives.items()]
        orgs_to_place.append(org)

        if gen % 100 == 0:
            #place orgs in cells
            for org in orgs_to_place:
                cell_idx_0 = bin_value(features["sparsity"], org.sparsity)
                cell_idx = tuple([cell_idx_0])
                cell = elites_map[cell_idx]
                if len(cell) == 0:
                    cells_with_orgs.append(cell_idx)
                cell.append(org)
            if gen >= config["initial_popsize"]:
                #recalculate pareto fronts of each cell
                for cell_idx,cell in elites_map.items():
                    new_fronts = fast_non_dominated_sort(cell)
                    new_cell_size = np.min([cell_capacity, len(cell)])
                    new_cell = []
                    i = 1
                    while len(new_cell) + len(new_fronts[i]) < new_cell_size:
                        crowding_distance_assignment(new_fronts[i])
                        new_cell.extend(new_fronts[i])
                        i += 1
                    if len(new_cell) < new_cell_size:
                        crowding_distance_assignment(new_fronts[i])
                        new_fronts[i].sort(key=lambda org: org.nsga_distance, reverse=True)
                        new_cell.extend(new_fronts[i][:new_cell_size-len(new_cell)])
                    elites_map[cell_idx] = new_cell
            orgs_to_place = []

        #statistics over time
        if gen % 100 == 0:
            print("Gen: ", gen)
            orgs_in_map = list(get_orgs_in_map(elites_map))
            for name, target in objectives.items():
                popFitnesses = [org.getError(name, target) for org in orgs_in_map]
                fitnessLog[name].append(mean(popFitnesses))
            coverage.append(len(orgs_in_map)/elites_map_max_size)

    #get and return final population
    population = list(get_orgs_in_map(elites_map))
    return population, fitnessLog, coverage, elites_map