from itertools import combinations
import json
import os
import pickle
from statistics import mean
import sys
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np

from elites import run as elitesrun
from eval_functions import functions
from nsga import fast_non_dominated_sort, run as nsgarun
from organism import Organism
from plot_utils import T, final_pop_distribution, final_pop_histogram
from random import seed


def plot_fitness(fitness_log, eval_func_names, save_loc, transparent=False):
    figure, axis = plt.subplots(1, 1)
    for func_name in eval_func_names:
        axis.plot(fitness_log[func_name], label=func_name)
    axis.set_yscale("log")
    figure.supxlabel("Generations")
    figure.supylabel("MSE")
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
    if transparent:
        figure.patch.set_alpha(0.0)
    plt.savefig("{}/coverage.png".format(save_loc))
    plt.close()


def plot_elites_map(elites_map, eval_funcs, features_dict, save_loc, transparent=False):
    def generate_heatmap(col, row, obj_name, obj_target):
        mean_heatmap = np.empty([len(row), len(col)])
        count_heatmap = np.empty([len(row), len(col)])
        for i in range(len(col)):
            for j in range(len(row)):
                cell = elites_map[(col[i], row[j])] if len(row) > 1 else elites_map[(col[i],)]
                if len(cell) > 0:
                    mean_heatmap[j,i] = round(mean([org.getError(obj_name, obj_target) for org in cell]), 3)
                else:
                    mean_heatmap[j,i] = 9999
                count_heatmap[j,i] = len(cell)
        return mean_heatmap, count_heatmap
    
    def save_heatmaps(mean_heatmap, count_heatmap, col_name, row_name, objective):
        col_labels = features_dict[col_name]
        row_labels = features_dict[row_name] if row_name is not None else [""]
        figure, axis = plt.subplots(1, 2, figsize=(12,7))
        axis[0].imshow(mean_heatmap, cmap="Greens")
        axis[0].set_xticks(np.arange(len(col_labels)), labels=col_labels)
        axis[0].set_yticks(np.arange(len(row_labels)), labels=row_labels)
        axis[0].set_title("Mean Cell {} Error".format(objective))
        axis[1].imshow(count_heatmap, cmap="Greens_r")
        axis[1].set_xticks(np.arange(len(col_labels)), labels=col_labels)
        axis[1].set_yticks(np.arange(len(row_labels)), labels=row_labels)
        axis[1].set_title("Count of Organisms in Each Cell")
        for i in range(len(col_labels)):
            for j in range(len(row_labels)):
                _ = axis[0].text(i, j, mean_heatmap[j, i], ha="center", va="center", color="black")
                _ = axis[1].text(i, j, count_heatmap[j, i], ha="center", va="center", color="black")
        figure.supxlabel(col_name)
        figure.supylabel(row_name)
        figure.tight_layout()
        if transparent:
            figure.patch.set_alpha(0.0)
        plt.savefig("{}/heatmap_{}.png".format(save_loc, objective))
        plt.close()

    feature_names = list(features_dict.keys())
    feature_bins = list(features_dict.values())
    for name,target in eval_funcs.items():
        if len(features_dict) == 1:
            mean_heatmap, count_heatmap = generate_heatmap(feature_bins[0], [None], name, target)
            save_heatmaps(mean_heatmap, count_heatmap, feature_names[0], None, name)
        elif len(features_dict) == 2:
            mean_heatmap, count_heatmap = generate_heatmap(feature_bins[0], feature_bins[1], name, target)
            save_heatmaps(mean_heatmap, count_heatmap, feature_names[0], feature_names[1], name)
        else:
            print("Too many features to plot elites map.")
            return


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
            plt.xlabel(feature1 + " MSE")
            plt.ylabel(feature2 + " MSE")
            plt.legend()
            if save_loc is not None:
                plt.savefig("{}/pareto_{}_{}.png".format(save_loc, feature1, feature2))
                plt.close()
            else:
                plt.show()


def diversity(population:list[Organism], config:dict, save_loc_i:str) :
    N = config["popsize"]
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
        else:
            diversity(final_pop,config,save_loc_i)

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