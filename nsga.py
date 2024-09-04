#Algorithm based on: Deb, Kalyanmoy, et al.
#"A fast and elitist multiobjective genetic algorithm: NSGA-II."
#And: Gregory S. Hornby.
#"ALPS: The Age-Layered Population Structure for Reducing the Problem of Premature Convergence"

from collections import Counter
from random import random, sample
from statistics import mean

from organism import Organism
from plot_utils import fast_non_dominated_sort


def run(config):
    num_generations = config["num_generations"]
    popsize = config["popsize"]
    objectives = config["eval_funcs"]
    track_diversity_over = config["track_diversity_over"]
    tracking_frequency = config["tracking_frequency"]
    network_size = config["network_size"]
    weight_range = config["weight_range"]
    crossover_rate = config["crossover_rate"]
    crossover_odds = config["crossover_odds"]
    mutation_rate = config["mutation_rate"]
    mutation_odds = config["mutation_odds"]
    tournament_probability = config["tournament_probability"]

    fitnessLog = {funcName:[] for funcName in objectives}
    diversityLog = {o:[] for o in track_diversity_over}
    if tracking_frequency == 0:
        tracking_frequency = num_generations

    age_gap = config["age_gap"]
    age_progression = [age_gap*x**2 for x in range(1, 11)]
    age_layers = [[]]
    if num_generations > age_progression[-1]:
        age_progression.append(num_generations)

    for gen in range(num_generations+1):
        #add new age layer if it is time
        if (gen == age_progression[len(age_layers)-1]):
            parents = nsga_tournament(age_layers[-1], 2*popsize, tournament_probability)
            children = [parents[i].makeCrossedCopyWith(
                        parents[i+popsize], crossover_rate, crossover_odds, gen).makeMutatedCopy(
                        mutation_rate, mutation_odds) for i in range(popsize)]
            for name, target in objectives.items():
                _ = [org.getError(name, target) for org in children]
            age_layers.append(children)

        #initialize / reset layer 0
        if gen % age_gap == 0:
            population = [Organism(network_size, random(), weight_range) for _ in range(popsize)]
            for name, target in objectives.items():
                _ = [org.getError(name, target) for org in population]
            F = fast_non_dominated_sort(population)
            _ = [nsga_distance_assignment(F[f]) for f in F]
            age_layers[0] = population

        #iterate over layers from youngest to oldest
        for l in range(len(age_layers)):
            layer_l = age_layers[l]
            max_age = age_progression[l]

            #produce offspring from layer l and l-1 (if l>0)
            if l > 0:
                R = layer_l + age_layers[l-1]
                F = fast_non_dominated_sort(R)
                _ = [nsga_distance_assignment(F[f]) for f in F]
            else:
                R = layer_l
                age_migrants_in = []
            parents = nsga_tournament(R, 2*popsize, tournament_probability)
            children = [parents[i].makeCrossedCopyWith(
                        parents[i+popsize], crossover_rate, crossover_odds, gen).makeMutatedCopy(
                        mutation_rate, mutation_odds) for i in range(popsize)]
            for name, target in objectives.items():
                _ = [org.getError(name, target) for org in children]

            #move organisms that have aged out to next layer
            layer_candidates = layer_l + children
            if l < len(age_layers)-1: #corrected for index 0 index 1 mismatch
                age_migrants_out = [org for org in layer_candidates if org.age > max_age]
                layer_candidates = [org for org in layer_candidates if org.age <= max_age]
            else:
                age_migrants_out = []

            #selection
            R = layer_candidates + age_migrants_in
            if len(R) < popsize:
                padding = [Organism(network_size, random(), weight_range) for _ in range(popsize-len(R))]
                for name, target in objectives.items():
                    _ = [org.getError(name, target) for org in padding]
                R.extend(padding)
            F = fast_non_dominated_sort(R)
            if len(R) == popsize:
                _ = [nsga_distance_assignment(F[f]) for f in F]
                P = R
            else:
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
            
            age_layers[l] = P
            age_migrants_in = age_migrants_out

        #evaluation
        if gen % tracking_frequency == 0:
            print("Generation", gen)
            oldest_layer = age_layers[-1]
            for name, target in objectives.items():
                popFitnesses = [org.getError(name, target) for org in oldest_layer]
                fitnessLog[name].append(mean(popFitnesses))
            for name in track_diversity_over:
                spread = len(Counter([org.getProperty(name) for org in oldest_layer]))
                diversityLog[name].append(spread)

    return oldest_layer, fitnessLog, diversityLog


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


def nsga_tournament(population, numOffspring, tournament_probability):
    parents = []
    for _ in range(numOffspring):
        if random() < tournament_probability:
            choices = sample(population, k=2)
            choice0 = choices[0]
            choice1 = choices[1]
            if choice0.nsga_rank > choice1.nsga_rank:
                parents.append(choice1)
            elif choice1.nsga_rank > choice0.nsga_rank:
                parents.append(choice0)
            elif choice0.nsga_distance < choice1.nsga_distance:
                parents.append(choice1)
            elif choice1.nsga_distance < choice0.nsga_distance:
                parents.append(choice0)
            else:
                parents.append(sample(choices, k=1)[0])
        else:
            parents.append(sample(population, k=1)[0])
    return parents