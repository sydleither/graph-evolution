import json
import os
import pickle
import sys
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np

from elites import run as elitesrun
from eval_functions import functions
from nsga import fast_non_dominated_sort, run as nsgarun
from organism import Organism
from plot_utils import T, final_pop_distribution, final_pop_histogram, plot_elites_map
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


def plotParetoFront(population, config, save_loc=None,firstFrontOnly=False):
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
                if firstFrontOnly: break
            plt.title(feature1+" "+feature2)
            plt.xlabel(feature1 + " Error")
            plt.ylabel(feature2 + " Error")
            plt.legend()
            if save_loc is not None:
                plt.savefig("{}/pareto_{}_{}.png".format(save_loc, feature1, feature2))
                plt.close()
            else:
                plt.show()


def diversity(population:list[Organism], save_loc_i:str) :
    N = len(population)
    with open("{}/diversity.csv".format(save_loc_i), 'w') as diversityFile:
        diversityFile.write("property,entropy,uniformity,spread\n")
        for name in functions:
            typeCounter = Counter([organism.getProperty(name) 
                                   if "distribution" not in name 
                                   else tuple(organism.getProperty(name)) 
                                   for organism in population])
            entropy = -sum([(count/N)*np.log2(count/N) for count in typeCounter.values()])
            uniformity = entropy / len(typeCounter)
            spread = len(typeCounter) / N
            diversityFile.write("{},{},{},{}\n".format(name, entropy, uniformity, spread))


def run_rep(i, save_loc, config, selection_scheme):
    seed(i)
    save_loc_i = "{}/{}".format(save_loc, i)
    if not os.path.exists(save_loc_i):
        os.makedirs(save_loc_i)

    if selection_scheme == "nsga":
        final_pop, fitness_log = nsgarun(config)
    else:
        final_pop, fitness_log, coverage, elites_map = elitesrun(config)

    if config["save_data"] == 1:
        with open("{}/final_pop.pkl".format(save_loc_i), "wb") as f:
            pickle.dump(final_pop, f)
        with open("{}/fitness_log.pkl".format(save_loc_i), "wb") as f:
            pickle.dump(fitness_log, f)
        if selection_scheme == "map-elites":
            with open("{}/coverage.pkl".format(save_loc_i), "wb") as f:
                pickle.dump(coverage, f)
            with open("{}/elites_map.pkl".format(save_loc_i), "wb") as f:
                pickle.dump(elites_map, f)
        diversity(final_pop, save_loc_i)

    if config["plot_data"] == 1:
        plot_fitness(fitness_log, config["eval_funcs"].keys(), save_loc_i)
        final_pop_histogram(final_pop, config["eval_funcs"], save_loc_i, plot_all=True)
        final_pop_distribution(final_pop, config["eval_funcs"], save_loc_i, plot_all=True, with_error=True)
        plotParetoFront(final_pop, config, save_loc_i)
        final_pop[0].saveGraphFigure("{}/graphFigure.png".format(save_loc_i))
        if selection_scheme == "map-elites":
            plot_coverage(coverage, save_loc_i)
            plot_elites_map(elites_map, config["eval_funcs"], config["diversity_funcs"], save_loc_i, transparent=False)


def main(config, rep=None):
    save_loc = "{}/{}".format(config["data_dir"], config["name"])
    if not os.path.exists(save_loc):
        os.makedirs(save_loc)

    config_path = "{}/config.json".format(save_loc)
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)

    if "diversity_funcs" in config:
        selection_scheme = "map-elites"
    else:
        selection_scheme = "nsga"

    if rep: #cmd specified only
        run_rep(rep, save_loc, config, selection_scheme)
    else:
        for i in range(config["reps"]):
            run_rep(i, save_loc, config, selection_scheme)


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