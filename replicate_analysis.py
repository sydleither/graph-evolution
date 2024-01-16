import json
import os
import pickle
import sys
from csv import reader
from statistics import mean

import matplotlib.pyplot as plt
import numpy as np
from eval_functions import Evaluation
from plot_utils import (T, calculate_standard_error, final_pop_distribution,
                        final_pop_histogram)
from ga import fast_non_dominated_sort


def plot_fitnesses_error(fitness_logs, eval_func_names, save_loc, transparent=False):
    figure, axis = plt.subplots(1, 1)
    num_replicates = len(fitness_logs)
    for func_name in eval_func_names:
        eval_func_data = [fitness_logs[i][func_name] for i in range(num_replicates)]
        eval_func_data_mean, neg_error, pos_error = calculate_standard_error(eval_func_data)
        axis.plot(eval_func_data_mean, label=func_name)
        axis.fill_between(range(len(eval_func_data_mean)), neg_error, pos_error, alpha=0.5)
    axis.set_yscale("log")
    figure.supxlabel("Generations")
    figure.supylabel("MSE")
    figure.legend()
    if transparent:
        figure.patch.set_alpha(0.0)
    plt.savefig("{}/fitness_w_error.png".format(save_loc))
    plt.close()


def plot_fitnesses_sep(fitness_logs, eval_func_names, save_loc, transparent=False):
    num_plots = len(eval_func_names)
    num_plots = num_plots if num_plots > 1 else 2
    figure, axis = plt.subplots(num_plots, 1, figsize=(5,3*num_plots))
    i = 0
    for func_name in eval_func_names:
        for run in range(len(fitness_logs)):
            axis[i].plot(fitness_logs[run][func_name], alpha=0.5, color='gray')
        eval_func_data = [fitness_logs[i][func_name] for i in range(len(fitness_logs))]
        eval_func_data_mean = [np.mean([eval_func_data[i][j] for i in range(len(eval_func_data))]) for j in range(len(eval_func_data[0]))]
        axis[i].plot(eval_func_data_mean, color="forestgreen")
        axis[i].set_title(func_name)
        axis[i].set_yscale("log")
        i += 1
    figure.supxlabel("Generations")
    figure.supylabel("MSE")
    figure.tight_layout(rect=[0, 0.03, 1, 0.95])
    if transparent:
        figure.patch.set_alpha(0.0)
    plt.savefig("{}/fitness.png".format(save_loc))
    plt.close()


def combined_pareto_front(final_pops,config,save_loc=None,firstFrontOnly=False):
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
                R = sorted(sorted([(org.evaluationScores[feature1], org.evaluationScores[feature2]) for org in allFronts[frontNumber]],
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


def combined_entropy(logs,data_path):
    scratch = {}
    for log in logs:
        for row in log:
            name,entropy = row
            if name not in scratch:
                scratch[name] = []
            scratch[name].append(float(entropy))
    for name in scratch:
        scratch[name] = mean(scratch[name])
    with open("{}/entropy_all.csv".format(data_path),'w') as entropyFile:
        entropyFile.write("Name,Entropy(bits)\n")
        for name,entropy in scratch.items():
            entropyFile.write("{},{}\n".format(name,entropy))


def main(config_dir):
    final_pops = []
    fitness_logs = []
    entropy_logs = []

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
                with open("{}/entropy.csv".format(full_path), "r") as f:
                    rdr = reader(f)
                    _ = next(f) #remove header
                    entropy_logs.append([line for line in rdr])

    data_path = "{}/{}".format(config_file["data_dir"], config_file["name"])
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    
    eval_obj = Evaluation(config_file)
    eval_funcs = config_file["eval_funcs"]
    final_pop_histogram(eval_obj, final_pops, eval_funcs, data_path, plot_all=True)
    final_pop_histogram(eval_obj, final_pops, eval_funcs, data_path, plot_all=False)
    final_pop_distribution(eval_obj, final_pops, eval_funcs, data_path, plot_all=True)
    final_pop_distribution(eval_obj, final_pops, eval_funcs, data_path, plot_all=False)
    plot_fitnesses_sep(fitness_logs, eval_funcs.keys(), data_path)
    plot_fitnesses_error(fitness_logs, eval_funcs.keys(), data_path)
    combined_pareto_front(final_pops,config_file,data_path)
    combined_entropy(entropy_logs,data_path)


if __name__ == "__main__":
    if len(sys.argv) == 2:
        main(sys.argv[1])
    else:
        print('Please give a valid data directory.')
