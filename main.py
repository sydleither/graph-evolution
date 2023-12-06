import json
import sys

import matplotlib.pyplot as plt

from ga import run


def plot_fitness(fitness_log, eval_func_names):
    figure, axis = plt.subplots(1, 2, figsize=(10,6))
    for func_name in eval_func_names:
        axis[0].plot(fitness_log[func_name], label=func_name)
        axis[1].plot(fitness_log[func_name])
    axis[1].set_yscale("log")
    figure.supxlabel("Generations")
    figure.supylabel("MSE")
    figure.legend()
    plt.savefig("fitness.png")
    plt.close()


def main(config):
    final_pop, fitness_log = run(config)
    plot_fitness(fitness_log, config["eval_funcs"].keys())


if __name__ == "__main__":
    try:
        config_file = sys.argv[1]
        config = json.load(open(config_file))
    except:
        print("Please give a valid config json to read parameters from.")
        exit()
    main(config)