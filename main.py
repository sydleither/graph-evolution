import json
import sys

import matplotlib.pyplot as plt
from bintools import numBins
from eval_functions import Evaluation
from ga import run


def final_pop_histogram(final_pop, eval_funcs, transparent=False):
    eval = Evaluation()
    num_plots = len(eval_funcs)
    figure, axis = plt.subplots(1, num_plots, figsize=(3*num_plots,5)) #TODO: dynamically add new rows when columns are full
    i = 0
    for func_name, ideal_val in eval_funcs.items():
        eval_func = getattr(eval, func_name)
        func_fitnesses = [eval_func(org) for org in final_pop]
        axis[i].hist(func_fitnesses, bins=numBins(func_fitnesses), color="forestgreen")
        axis[i].axvline(ideal_val, color="black", linestyle="--")
        axis[i].set_title(func_name)
        i += 1
    figure.tight_layout(rect=[0, 0.03, 1, 0.95])
    figure.suptitle('Final Population Histograms')
    plt.savefig('histograms.png', transparent=transparent)
    plt.close()


def plot_fitness(fitness_log, eval_func_names, transparent=False):
    figure, axis = plt.subplots(1, 1)
    for func_name in eval_func_names:
        axis.plot(fitness_log[func_name], label=func_name)
    axis.set_yscale("log")
    figure.supxlabel("Generations")
    figure.supylabel("MSE")
    figure.legend()
    if transparent:
        figure.patch.set_alpha(0.0)
    plt.savefig("fitness.png")
    plt.close()


def main(config):
    final_pop, fitness_log = run(config)
    
    plot_fitness(fitness_log, config["eval_funcs"].keys())
    final_pop_histogram(final_pop, config["eval_funcs"])
    final_pop[0].saveGraphFigure("./graphFigure.png")


if __name__ == "__main__":
    try:
        config_file = sys.argv[1]
        config = json.load(open(config_file))
    except:
        print("Please give a valid config json to read parameters from.")
        exit()
    main(config)