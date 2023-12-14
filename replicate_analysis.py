import os
import json
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
from bintools import numBins
from eval_functions import Evaluation


def final_pop_histograms_all(eval, final_pops, eval_funcs, save_loc, transparent=False):
    all_property_names = [func for func in dir(Evaluation) if callable(getattr(Evaluation, func)) and not func.startswith("__")]
    figure, axis = plt.subplots(5, 3, figsize=(12, 15))
    fig_row = 0
    fig_col = 0
    for property_name in all_property_names:
        is_eval_func = property_name in eval_funcs.keys()
        eval_func = getattr(eval, property_name)
        data = [[eval_func(org) for org in final_pops[run]] for run in range(len(final_pops))]
        axis[fig_row][fig_col].hist(data, bins=numBins([d for dd in data for d in dd]), stacked=True)
        if is_eval_func:
            ideal_val = eval_funcs[property_name]["target"] if "target" in eval_funcs[property_name].keys() else 0
            axis[fig_row][fig_col].axvline(ideal_val, color="black", linestyle="--")
        color = "forestgreen" if is_eval_func else "sienna"
        axis[fig_row][fig_col].set_title(property_name, color=color)
        fig_row += 1
        if fig_row % 5 == 0:
            fig_col += 1
            fig_row = 0
    figure.tight_layout(rect=[0, 0.03, 1, 0.95])
    figure.suptitle('Final Population Histograms')
    plt.savefig("{}/histograms_all.png".format(save_loc), transparent=transparent)
    plt.close()



def final_pop_histograms(eval, final_pops, eval_funcs, save_loc, transparent=False):
    num_plots = len(eval_funcs)
    figure, axis = plt.subplots(1, num_plots, figsize=(4*num_plots,5))
    i = 0
    for func_name, func_params in eval_funcs.items():
        eval_func = getattr(eval, func_name)
        ideal_val = func_params["target"] if "target" in func_params.keys() else 0
        data = [[eval_func(org) for org in final_pops[run]] for run in range(len(final_pops))]
        axis[i].hist(data, bins=numBins([d for dd in data for d in dd]), stacked=True)
        axis[i].axvline(ideal_val, color="black", linestyle="--")
        axis[i].set_title(func_name)
        i += 1
    figure.tight_layout(rect=[0, 0.03, 1, 0.95])
    figure.suptitle("Final Population Histograms")
    plt.savefig("{}/histograms.png".format(save_loc), transparent=transparent)
    plt.close()


def plot_fitnesses_error(fitness_logs, eval_func_names, save_loc, transparent=False):
    figure, axis = plt.subplots(1, 1)
    num_replicates = len(fitness_logs)
    for func_name in eval_func_names:
        eval_func_data = [fitness_logs[i][func_name] for i in range(num_replicates)]
        num_generations = len(eval_func_data[0])
        eval_func_data_mean = [np.mean([eval_func_data[i][j] for i in range(num_replicates)]) for j in range(num_generations)]
        eval_func_data_error = [np.std([eval_func_data[i][j] for i in range(num_replicates)])/np.sqrt(num_replicates) for j in range(num_generations)]
        neg_error = [eval_func_data_mean[i]-eval_func_data_error[i] for i in range(num_generations)]
        pos_error = [eval_func_data_mean[i]+eval_func_data_error[i] for i in range(num_generations)]
        axis.plot(eval_func_data_mean, label=func_name)
        axis.fill_between(range(num_generations), neg_error, pos_error, alpha=0.5)
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
    final_pop_histograms(eval_obj, final_pops, config_file["eval_funcs"], data_path)
    final_pop_histograms_all(eval_obj, final_pops, config_file["eval_funcs"], data_path)
    plot_fitnesses_sep(fitness_logs, config_file["eval_funcs"].keys(), data_path)
    plot_fitnesses_error(fitness_logs, config_file["eval_funcs"].keys(), data_path)


if __name__ == "__main__":
    if len(sys.argv) == 2:
        main(sys.argv[1])
    else:
        print('Please give a valid data directory.')
