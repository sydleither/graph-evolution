import json
import os
import pickle
import sys
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np

from elites import get_features_dict, run
from eval_functions import functions, properties
from organism import Organism
from plot_utils import (fast_non_dominated_sort, final_pop_distribution, 
                        final_pop_histogram, get_perfect_pop, plot_elites_map, T)
from random import seed


def plot_fitness(fitness_log, eval_func_names, save_loc, transparent=False):
    figure, axis = plt.subplots(1, 1)
    for func_name in eval_func_names:
        axis.plot(fitness_log[func_name], label=func_name)
    axis.set_yscale("log")
    figure.supxlabel("Generations")
    figure.supylabel("Error")
    figure.legend()
    if transparent:
        figure.patch.set_alpha(0.0)
    plt.savefig("{}/fitness.png".format(save_loc))
    plt.close()


def plot_coverage(coverage, save_loc, transparent=False):
    figure, axis = plt.subplots(1, 1)
    axis.plot(coverage)
    figure.supxlabel("Generations")
    figure.supylabel("Coverage")
    figure.tight_layout()
    if transparent:
        figure.patch.set_alpha(0.0)
    plt.savefig("{}/coverage.png".format(save_loc))
    plt.close()


def plotParetoFront(population, config, save_loc=None, first_front_only=False):
    #sort
    allFronts = fast_non_dominated_sort(population)
    #plot
    funcNames = list(config["eval_funcs"].keys())
    for i, feature1 in enumerate(funcNames):
        for j, feature2 in enumerate(funcNames):
            if j <= i: continue
            for frontNumber in sorted(allFronts.keys()):
                R = sorted(sorted([(org.errors[feature1], org.errors[feature2]) for org in allFronts[frontNumber]],
                                    key=lambda r: r[1], reverse=True), key=lambda r: r[0])
                plt.plot(*T(R), marker="o", linestyle="--",label=frontNumber)
                if first_front_only: break
            plt.title(feature1 + " " + feature2)
            plt.xlabel(feature1 + " Error")
            plt.ylabel(feature2 + " Error")
            plt.legend()
            if save_loc is not None:
                plt.savefig("{}/pareto_{}_{}.png".format(save_loc, feature1, feature2))
                plt.close()
            else:
                plt.show()


def diversity(population:list[Organism], perfect_pop:list[Organism], save_loc_i:str):
    N = len(perfect_pop)
    with open("{}/diversity.csv".format(save_loc_i), 'w') as diversityFile:
        diversityFile.write("property,entropy,uniformity,spread,final_pop_size,optimized_size\n")
        for name in functions:
            typeCounter = Counter([organism.getProperty(name) 
                                   if "distribution" not in name 
                                   else tuple(organism.getProperty(name)) 
                                   for organism in perfect_pop])
            entropy = -sum([(count/N)*np.log2(count/N) for count in typeCounter.values()])
            uniformity = entropy / np.log2(len(typeCounter))
            spread = len(typeCounter)
            final_pop_size = len(population)
            optimized_size = N
            diversityFile.write("{},{},{},{},{},{}\n".format(name, entropy, uniformity, 
                                                             spread, final_pop_size, optimized_size))


def run_rep(i, save_loc, config):
    seed(i)
    save_loc_i = "{}/{}".format(save_loc, i)
    if not os.path.exists(save_loc_i):
        os.makedirs(save_loc_i)

    objectives = config["eval_funcs"]
    drift_properties = [p for p in properties if p not in objectives]
    if len(drift_properties) > 18:
        print("Too many drift properties to hash.")
        exit()
    final_pop, fitness_log, coverage, elites_map = run(config)
    perfect_pop = get_perfect_pop(final_pop, objectives)
    features = get_features_dict(config["hash_resolution"], float("1"+"9"*len(drift_properties)))

    if config["save_data"] == 1:
        with open("{}/final_pop.pkl".format(save_loc_i), "wb") as f:
            pickle.dump(final_pop, f)
        with open("{}/fitness_log.pkl".format(save_loc_i), "wb") as f:
            pickle.dump(fitness_log, f)
        with open("{}/coverage.pkl".format(save_loc_i), "wb") as f:
            pickle.dump(coverage, f)
        with open("{}/elites_map.pkl".format(save_loc_i), "wb") as f:
            pickle.dump(elites_map, f)
        diversity(final_pop, perfect_pop, save_loc_i)

    if config["plot_data"] == 1:
        plot_fitness(fitness_log, objectives.keys(), save_loc_i)
        final_pop_histogram(perfect_pop, objectives, save_loc_i, plot_all=True)
        final_pop_distribution(perfect_pop, objectives, save_loc_i, plot_all=True, with_error=True)
        plot_coverage(coverage, save_loc_i)
        plot_elites_map(elites_map, objectives, features, save_loc_i, transparent=False)


def main(config, rep=None):
    save_loc = "{}/{}".format(config["data_dir"], config["name"])
    if not os.path.exists(save_loc):
        os.makedirs(save_loc)

    config_path = "{}/config.json".format(save_loc)
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)

    if rep: #cmd specified only
        run_rep(rep, save_loc, config)
    else:
        for i in range(config["reps"]):
            run_rep(i, save_loc, config)


if __name__ == "__main__":
    try:
        config_file = sys.argv[1]
        config = json.load(open(config_file))
    except:
        print("Please give a valid config json to read parameters from.")
        exit()
    
    if len(sys.argv) == 2:
        main(config)
    elif len(sys.argv) == 3:
        rep = sys.argv[2]
        main(config, rep)
    else:
        print("Please pass in valid arguments: config and (rep)(optional)")