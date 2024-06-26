import json
import os
import pickle
import sys
from csv import reader
from statistics import mean

import matplotlib.pyplot as plt
import numpy as np

from elites import get_features_dict
from plot_utils import (calculate_confidence_interval, fast_non_dominated_sort, 
                        final_pop_distribution, final_pop_histogram, 
                        get_perfect_pop, plot_elites_map, T)


def plot_fitnesses_error(fitness_logs, eval_func_names, save_loc, transparent=False):
    figure, axis = plt.subplots(1, 1)
    num_replicates = len(fitness_logs)
    for func_name in eval_func_names:
        eval_func_data = [fitness_logs[i][func_name] for i in range(num_replicates)]
        eval_func_data_mean, neg_error, pos_error = calculate_confidence_interval(eval_func_data)
        axis.plot(eval_func_data_mean, label=func_name)
        axis.fill_between(range(len(eval_func_data_mean)), neg_error, pos_error, alpha=0.5)
    axis.set_yscale("log")
    figure.supxlabel("Generations")
    figure.supylabel("Error")
    figure.legend()
    if transparent:
        figure.patch.set_alpha(0.0)
    plt.savefig("{}/fitness_w_error.png".format(save_loc))
    plt.close()


def plot_fitnesses_sep(fitness_logs, eval_func_names, save_loc, transparent=False):
    num_plots = len(eval_func_names)
    num_plots = num_plots if num_plots > 1 else 2
    figure, axis = plt.subplots(num_plots, 1, figsize=(5,3*num_plots))
    sp = 0
    for func_name in eval_func_names:
        for run in range(len(fitness_logs)):
            axis[sp].plot(fitness_logs[run][func_name], alpha=0.5, color='gray')
        eval_func_data = [fitness_logs[i][func_name] for i in range(len(fitness_logs))]
        eval_func_data_mean = [np.mean([eval_func_data[i][j] for i in range(len(eval_func_data))]) for j in range(len(eval_func_data[0]))]
        axis[sp].plot(eval_func_data_mean, color="forestgreen")
        axis[sp].set_title(func_name)
        axis[sp].set_yscale("log")
        sp += 1
    figure.supxlabel("Generations")
    figure.supylabel("Error")
    figure.suptitle("Error Over Time")
    figure.tight_layout(rect=[0, 0.03, 1, 0.95])
    if transparent:
        figure.patch.set_alpha(0.0)
    plt.savefig("{}/fitness.png".format(save_loc))
    plt.close()


def plot_coverage(coverages, save_loc, transparent=False):
    figure, axis = plt.subplots(2, 1)
    coverage_mean, neg_error, pos_error = calculate_confidence_interval(coverages)
    for run in range(len(coverages)):
        axis[0].plot(coverages[run], alpha=0.5, color='gray')
    axis[0].plot(coverage_mean, color="forestgreen")
    axis[0].set_title("All Replicates")
    axis[1].plot(coverage_mean, color="forestgreen")
    axis[1].fill_between(range(len(coverage_mean)), neg_error, pos_error, alpha=0.5, color="forestgreen")
    axis[1].set_title("Replicates Confidence Interval")
    figure.supxlabel("Generations")
    figure.supylabel("Coverage")
    figure.tight_layout()
    if transparent:
        figure.patch.set_alpha(0.0)
    plt.savefig("{}/coverage.png".format(save_loc))
    plt.close()


def combined_pareto_front(final_pops, config, save_loc=None, first_front_only=False):
    #sort
    allOrgs = [org for pop in final_pops for org in pop ]
    newID = 0
    for org in allOrgs:
        org.id = newID
        newID+= 1
    allFronts = fast_non_dominated_sort(allOrgs)
    #plot
    funcNames = list(config["eval_funcs"].keys())
    for i, feature1 in enumerate(funcNames):
        for j, feature2 in enumerate(funcNames):
            if j <= i: continue
            for frontNumber in sorted(allFronts.keys()):
                R = sorted(sorted([(org.getError(feature1,None), org.getError(feature2,None)) for org in allFronts[frontNumber]],
                                    key=lambda r: r[1], reverse=True), key=lambda r: r[0])
                plt.plot(*T(R), marker="o", linestyle="--",label=frontNumber)
                if first_front_only: break
            plt.title(feature1+" "+feature2)
            plt.xlabel(feature1 + " Error")
            plt.ylabel(feature2 + " Error")
            plt.legend()
            if save_loc is not None:
                plt.savefig("{}/pareto_{}_{}.png".format(save_loc, feature1, feature2))
                plt.close()
            else:
                plt.show()


def combined_diversity(logs, data_path):
    scratch = {}
    for log in logs:
        for row in log:
            name, entropy, uniformity, spread, final_pop_size, perfect_pop_size = row
            if name not in scratch:
                scratch[name] = []
            scratch[name].append([float(entropy), float(uniformity), float(spread), 
                                  float(final_pop_size), float(perfect_pop_size)])
    for name in scratch:
        scratch[name] = [mean(x) for x in T(scratch[name])]
    with open("{}/diversity_all.csv".format(data_path), 'w') as entropyFile:
        entropyFile.write("property,entropy,uniformity,spread,final_pop_size,optimized_size\n")
        for name, measures in scratch.items():
            entropyFile.write("{},{},{},{},{},{}\n".format(name, measures[0], measures[1], 
                                                     measures[2], measures[3], measures[4]))


def main(config_dir):
    final_pops = []
    fitness_logs = []
    diversity_logs = []
    elites_maps = []
    coverages = []

    for run_dir in os.listdir(config_dir):
        full_path = "{}/{}".format(config_dir, run_dir)
        if run_dir.endswith(".json"):
            config_file = json.load(open(full_path))
        elif not os.path.isfile(full_path):
            if os.path.exists("{}/final_pop.pkl".format(full_path)):
                with open("{}/final_pop.pkl".format(full_path), "rb") as f:
                    final_pops.append(pickle.load(f))
                with open("{}/fitness_log.pkl".format(full_path), "rb") as f:
                    fitness_logs.append(pickle.load(f))
                with open("{}/diversity.csv".format(full_path), "r") as f:
                    rdr = reader(f)
                    _ = next(f) #remove header
                    diversity_logs.append([line for line in rdr])
                with open("{}/elites_map.pkl".format(full_path), "rb") as f:
                    elites_maps.append(pickle.load(f))
                with open("{}/coverage.pkl".format(full_path), "rb") as f:
                    coverages.append(pickle.load(f))

    data_path = "{}/{}".format(config_file["data_dir"], config_file["name"])
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    eval_funcs = config_file["eval_funcs"]
    perfect_pops = []
    for final_pop in final_pops:
        perfect_pops.append(get_perfect_pop(final_pop, eval_funcs))
    features = get_features_dict(config_file["hash_resolution"])

    final_pop_histogram(perfect_pops, eval_funcs, data_path, plot_all=True)
    final_pop_distribution(perfect_pops, eval_funcs, data_path, plot_all=True)
    plot_fitnesses_sep(fitness_logs, eval_funcs.keys(), data_path)
    plot_fitnesses_error(fitness_logs, eval_funcs.keys(), data_path)
    combined_diversity(diversity_logs, data_path)
    plot_coverage(coverages, data_path)
    combo_elites_map = {}
    map_keys = elites_maps[0].keys()
    for map_key in map_keys:
        all_cells = [elite_map[map_key] for elite_map in elites_maps]
        combo_elites_map[map_key] = [org for cell in all_cells for org in cell]
    plot_elites_map(combo_elites_map, config_file["eval_funcs"], features, data_path)


if __name__ == "__main__":
    if len(sys.argv) == 2:
        main(sys.argv[1])
    else:
        print('Please give a valid data directory.')
