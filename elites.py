from itertools import product
from random import random, sample
from statistics import mean

import networkx as nx
import numpy as np

from organism import Organism

#multi-objective map-elites based on:
#"Illuminating search spaces by mapping elites" (Mouret & Clune, 2015)
#"Multi-Objective Quality Diversity Optimization" (Pierrot et al., 2023)


def get_features_dict(hash_resolution):
    return {"sparsity":np.round(np.linspace(0, 1, 11), decimals=1), 
            "genome_hash":np.linspace(0, int("F"*12, 16), hash_resolution)}


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


def genome_hash(genotype_matrix, num_nodes):
    G = nx.DiGraph(np.array(genotype_matrix))
    hash_hex = nx.weisfeiler_lehman_graph_hash(G, edge_attr="weight", iterations=num_nodes//10, digest_size=6)
    return int(hash_hex, 16)


def run(config):
    #extract relevant config information for efficiency
    objectives = config["eval_funcs"]
    cell_capacity = config["cell_capacity"]
    mutation_rate = config["mutation_rate"]
    mutation_odds = config["mutation_odds"]
    crossover_rate = config["crossover_rate"]
    crossover_odds = config["crossover_odds"]
    num_nodes = config["network_size"]
    features = get_features_dict(config["hash_resolution"])
    feature_bins = list(features.values())
    #initalize elites map
    elites_map = {x:[] for x in product(*feature_bins)}
    elites_map_max_size = len(elites_map) * cell_capacity
    #initalize tracking performance over time
    fitnessLog = {funcName:[] for funcName in objectives}
    coverage = []

    for gen in range(1, config["num_generations"]+1):
        #list of all organisms in elites map
        orgs_in_map = list(get_orgs_in_map(elites_map))

        #get new organism to potentially place in map
        if gen < config["initial_popsize"]:
            org = Organism(config["network_size"], random(), config["weight_range"])
        else:
            porg1, porg2 = sample(orgs_in_map, 2)
            org = porg1.makeCrossedCopyWith(porg2, crossover_rate, crossover_odds).makeMutatedCopy(mutation_rate, mutation_odds)
        [org.getError(name, target) for name, target in objectives.items()]

        #get the organism's value for each feature, round that value to the nearest bin, convert the bin into its elites map index
        cell_idx_0 = bin_value(features["sparsity"], org.sparsity)
        cell_idx_1 = bin_value(features["genome_hash"], genome_hash(org.genotypeMatrix, num_nodes))
        cell_idx = tuple([cell_idx_0, cell_idx_1])
        #calculate pareto front of cell when including the new organism
        cell = elites_map[cell_idx]
        cell.append(org)
        new_front = first_front(cell)
        #replace the cell with the new pareto front
        if len(new_front) > cell_capacity:
            crowding_distance_assignment(new_front)
            new_front.sort(key=lambda org: org.nsga_distance,reverse=True)
            new_cell = new_front[:cell_capacity]
        else:
            new_cell = new_front
        elites_map[cell_idx] = new_cell

        #statistics over time
        if gen % 10 == 0:
            print("Gen: ", gen)
            for name, target in objectives.items():
                popFitnesses = [org.getError(name, target) for org in orgs_in_map]
                fitnessLog[name].append(mean(popFitnesses))
            coverage.append(len(orgs_in_map)/elites_map_max_size)

    #get and return final population
    population = list(get_orgs_in_map(elites_map))
    return population, fitnessLog, coverage, elites_map