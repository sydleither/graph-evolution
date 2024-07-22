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
    return {"pheno_hash":np.linspace(0, int("1"*32, 2), hash_resolution),
            "unweighted_hash":np.linspace(int("1"*31, 2), int("1"*32, 2), hash_resolution)}


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


def edit_distance(vec):
    sos = 0
    for i in range(len(vec)):
        sos += (vec[i])**2
    return np.sqrt(sos)


# def genome_hash_bin(phenotype_matrix, num_nodes):
#     binary_matrix = [[1 if val !=0 else 0 for val in row] for row in phenotype_matrix]
#     v = [0 for _ in range(num_nodes*2)]
#     for row in binary_matrix:
#         for i in range(num_nodes):
#             if row[i] == 1:
#                 v[i] += 1
#             else:
#                 v[i] -= 1
#     binary_matrix_T = T(binary_matrix)
#     for row in binary_matrix_T:
#         for i in range(num_nodes):
#             if row[i] == 1:
#                 v[i+num_nodes] += 1
#             else:
#                 v[i+num_nodes] -= 1
#     hash_bin = "".join(["0" if v[i] < 0 else "1" for i in range(num_nodes*2)])
#     return int(hash_bin, 2)


def genome_hash_bin(phenotype_matrix, hash_res=32):
    binary_matrix = [[1 if val !=0 else 0 for val in row] for row in phenotype_matrix]
    v = [0 for _ in range(hash_res)]
    for row in binary_matrix:
        row_to_int = int("".join([str(x) for x in row]), 2)
        row_hash_binary = bin(int(md5(struct.pack('>f', row_to_int)).hexdigest(), 16))[2:].zfill(hash_res)
        for i in range(hash_res):
            if row_hash_binary[i] == "1":
                v[i] += 1
            else:
                v[i] -= 1
    hash_bin = "".join(["0" if v[i] < 0 else "1" for i in range(hash_res)])
    return int(hash_bin, 2)


def genome_hash(genotype_matrix, hash_res=32):
    genome_flat = [x for y in genotype_matrix for x in y]
    v = [0 for _ in range(hash_res)]
    for x in genome_flat:
        x_hash_binary = bin(int(md5(struct.pack('>f', x)).hexdigest(), 16))[2:].zfill(hash_res)
        for i in range(hash_res):
            if x_hash_binary[i] == "1":
                v[i] += x
            else:
                v[i] -= x
    hash_bin = "".join(["0" if v[i] < 0 else "1" for i in range(hash_res)])
    return int(hash_bin, 2)


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

        #get the organism's value for each feature, round that value to the nearest bin, convert the bin into its elites map index
        cell_idx_0 = bin_value(features["pheno_hash"], genome_hash(org.adjacencyMatrix))
        cell_idx_1 = bin_value(features["unweighted_hash"], genome_hash_bin(org.adjacencyMatrix))
        cell_idx = tuple([cell_idx_0, cell_idx_1])
        #calculate pareto front of cell when including the new organism
        cell = elites_map[cell_idx]
        if len(cell) == 0:
            cells_with_orgs.append(cell_idx)
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