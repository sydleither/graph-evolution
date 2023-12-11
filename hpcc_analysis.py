import os
import json
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
from bintools import numBins
from eval_functions import Evaluation


def final_pop_histograms(final_pops, eval_funcs, file_name, transparent=False):
    eval = Evaluation()
    num_plots = len(eval_funcs)
    figure, axis = plt.subplots(1, num_plots, figsize=(3*num_plots,5))
    i = 0
    for func_name, ideal_val in eval_funcs.items():
        eval_func = getattr(eval, func_name)
        for run in range(len(final_pops)):
            func_fitnesses = [eval_func(org) for org in final_pops[run]]
            axis[i].hist(func_fitnesses, bins=numBins(func_fitnesses), alpha=0.5)
        axis[i].axvline(ideal_val, color="black", linestyle="--")
        axis[i].set_title(func_name)
        i += 1
    figure.tight_layout(rect=[0, 0.03, 1, 0.95])
    figure.suptitle('Final Population Histograms')
    plt.savefig(f'{file_name}/histograms.png', transparent=transparent)
    plt.close()


def plot_fitnesses_error(fitness_logs, eval_func_names, file_name, transparent=False):
    figure, axis = plt.subplots(1, 1)
    for func_name in eval_func_names:
        eval_func_data = [fitness_logs[i][func_name] for i in range(len(fitness_logs))]
        eval_func_data_mean = [np.mean([eval_func_data[i][j] for i in range(len(eval_func_data))]) for j in range(len(eval_func_data[0]))]
        eval_func_data_error = [np.std([eval_func_data[i][j] for i in range(len(eval_func_data))])/np.sqrt(len(eval_func_data)) for j in range(len(eval_func_data[0]))]
        neg_error = [eval_func_data_mean[i]-eval_func_data_error[i] for i in range(len(eval_func_data_mean))]
        pos_error = [eval_func_data_mean[i]+eval_func_data_error[i] for i in range(len(eval_func_data_mean))]
        axis.plot(eval_func_data_mean, label=func_name)
        axis.fill_between(range(len(eval_func_data_mean)), neg_error, pos_error, alpha=0.5)
    axis.set_yscale("log")
    figure.supxlabel("Generations")
    figure.supylabel("MSE")
    figure.legend()
    if transparent:
        figure.patch.set_alpha(0.0)
    plt.savefig(f'{file_name}/fitness1.png')
    plt.close()


def plot_fitnesses_sep(fitness_logs, eval_func_names, file_name, transparent=False):
    num_plots = len(eval_func_names)
    figure, axis = plt.subplots(num_plots, 1, figsize=(5,3*num_plots))
    i = 0
    for func_name in eval_func_names:
        eval_func_data = [fitness_logs[i][func_name] for i in range(len(fitness_logs))]
        eval_func_data_mean = [np.mean([eval_func_data[i][j] for i in range(len(eval_func_data))]) for j in range(len(eval_func_data[0]))]
        axis[i].plot(eval_func_data_mean)
        for run in range(len(fitness_logs)):
            axis[i].plot(fitness_logs[run][func_name], alpha=0.5, color='gray')
        axis[i].set_title(func_name)
        axis[i].set_yscale("log")
        i += 1
    figure.supxlabel("Generations")
    figure.supylabel("MSE")
    figure.tight_layout(rect=[0, 0.03, 1, 0.95])
    if transparent:
        figure.patch.set_alpha(0.0)
    plt.savefig(f'{file_name}/fitness2.png')
    plt.close()


def main(config_name):
    final_pops = []
    fitness_logs = []

    config_dir = SCRATCH_LOC+config_name
    for run_dir in os.listdir(config_dir):
        with open(f'{config_dir}/{run_dir}/final_pop.pkl', 'rb') as f:
            final_pops.append(pickle.load(f))
        with open(f'{config_dir}/{run_dir}/fitness_log.pkl', 'rb') as f:
            fitness_logs.append(pickle.load(f))
    config_file = json.load(open(f'{config_dir}/0/{config_name}.json'))

    data_path = f'data/{config_name}'
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    
    final_pop_histograms(final_pops, config_file["eval_funcs"], data_path)
    plot_fitnesses_sep(fitness_logs, config_file["eval_funcs"].keys(), data_path)


if __name__ == "__main__":
    SCRATCH_LOC = '/mnt/gs21/scratch/leithers/graph-evolution/'
    if len(sys.argv) == 2:
        main(sys.argv[1])
    else:
        print('Please give a valid run directory located in scratch.')
    