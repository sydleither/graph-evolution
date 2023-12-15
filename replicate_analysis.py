import os
import json
from math import ceil
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
from bintools import numBins
from eval_functions import Evaluation
from plot_utils import calculate_standard_error, final_pop_histogram


def final_pop_distributions(eval_obj, final_pops, eval_funcs, save_loc, plot_all=True, transparent=False):
    #get list of properties to plot
    if plot_all:
        distributions = [func for func in dir(Evaluation) if callable(getattr(Evaluation, func)) and func.endswith("distribution")]
    else:
        distributions = [func for func in eval_funcs.keys() if func.endswith("distribution")]
    #dynamically set size of figure
    num_plots = len(distributions)
    if num_plots == 0:
        return
    fig_col_cnt = 1 if num_plots == 1 else 2 if num_plots <= 4 else 4
    fig_row_cnt = ceil(num_plots/fig_col_cnt)
    figure, axis = plt.subplots(fig_row_cnt, fig_col_cnt, figsize=(4*fig_row_cnt, 3*fig_col_cnt), squeeze=False)
    fig_row = 0
    fig_col = 0
    #plot every distribution and if plotting more than the objective distributions, color them differently
    for dist_name in distributions:
        for final_pop in final_pops:
            eval_func = getattr(eval_obj, dist_name)
            org_dists = [eval_func(org) for org in final_pop]
            degree_mean, neg_error, pos_error = calculate_standard_error(org_dists)
            axis[fig_row][fig_col].plot(degree_mean, label=dist_name)
            axis[fig_row][fig_col].fill_between(range(len(degree_mean)), neg_error, pos_error, alpha=0.5)
        is_eval_func = dist_name in eval_funcs.keys()
        if is_eval_func:
            goal_dist = eval_obj.dist_dict[dist_name]
            axis[fig_row][fig_col].plot(goal_dist, color="black", linewidth=2)
        if plot_all:
            color = "forestgreen" if is_eval_func else "sienna"
        else:
            color = "black"
        axis[fig_row][fig_col].set_title(dist_name, color=color)
        fig_row += 1
        if fig_row % fig_row_cnt == 0:
            fig_col += 1
            fig_row = 0
    figure.tight_layout(rect=[0, 0.03, 1, 0.95])
    figure.suptitle('Final Population Distributions')
    fig_name = "distributions_all" if plot_all else "distributions"
    plt.savefig("{}/{}.png".format(save_loc, fig_name), bbox_inches='tight', transparent=transparent)
    plt.close()


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


def main(config_dir):
    final_pops = []
    fitness_logs = []

    for run_dir in os.listdir(config_dir):
        full_path = "{}/{}".format(config_dir, run_dir)
        if run_dir.endswith(".json"):
            config_file = json.load(open(full_path))
        elif not os.path.isfile(full_path):
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
    final_pop_distributions(eval_obj, final_pops, eval_funcs, data_path, plot_all=True)
    final_pop_distributions(eval_obj, final_pops, eval_funcs, data_path, plot_all=False)
    plot_fitnesses_sep(fitness_logs, eval_funcs.keys(), data_path)
    plot_fitnesses_error(fitness_logs, eval_funcs.keys(), data_path)


if __name__ == "__main__":
    if len(sys.argv) == 2:
        main(sys.argv[1])
    else:
        print('Please give a valid data directory.')
