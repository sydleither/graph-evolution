import os
import json
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
from eval_functions import Evaluation
from plot_utils import calculate_standard_error, final_pop_distribution, final_pop_histogram


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


def main(config_dir): #TODO: get pareto front from all reps
    final_pops = []
    fitness_logs = []

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


if __name__ == "__main__":
    if len(sys.argv) == 2:
        main(sys.argv[1])
    else:
        print('Please give a valid data directory.')
