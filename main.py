import json
import os
import pickle
import sys

import matplotlib.pyplot as plt
from bintools import numBins
from eval_functions import Evaluation
from ga import run


#transpose a matrix (list of list)
def T(LL:list[list])->list[list]:
    return list(zip(*LL))


def distributions(): #TODO
    return


def final_pop_histogram_all(eval, final_pop, eval_funcs, save_loc, transparent=False):
    all_property_names = [func for func in dir(Evaluation) if callable(getattr(Evaluation, func)) and not func.startswith("__")]
    figure, axis = plt.subplots(5, 3, figsize=(12, 15))
    fig_row = 0
    fig_col = 0
    for property_name in all_property_names:
        is_eval_func = property_name in eval_funcs.keys()
        eval_func = getattr(eval, property_name)
        func_fitnesses = [eval_func(org) for org in final_pop]
        color = "forestgreen" if is_eval_func else "sienna"
        axis[fig_row][fig_col].hist(func_fitnesses, bins=numBins(func_fitnesses), color=color)
        if is_eval_func:
            ideal_val = eval_funcs[property_name]["target"] if "target" in eval_funcs[property_name].keys() else 0
            axis[fig_row][fig_col].axvline(ideal_val, color="black", linestyle="--")
        axis[fig_row][fig_col].set_title(property_name)
        fig_row += 1
        if fig_row % 5 == 0:
            fig_col += 1
            fig_row = 0
    figure.tight_layout(rect=[0, 0.03, 1, 0.95])
    figure.suptitle('Final Population Histograms')
    plt.savefig("{}/histograms_all.png".format(save_loc), transparent=transparent)
    plt.close()


def final_pop_histogram(eval, final_pop, eval_funcs, save_loc, transparent=False):
    num_plots = len(eval_funcs)
    figure, axis = plt.subplots(1, num_plots, figsize=(4*num_plots,5)) #TODO: dynamically add new rows when columns are full
    i = 0
    for func_name, func_params in eval_funcs.items():
        eval_func = getattr(eval, func_name)
        func_fitnesses = [eval_func(org) for org in final_pop]
        ideal_val = func_params["target"] if "target" in func_params.keys() else 0
        axis[i].hist(func_fitnesses, bins=numBins(func_fitnesses), color="forestgreen")
        axis[i].axvline(ideal_val, color="black", linestyle="--")
        axis[i].set_title(func_name)
        i += 1
    figure.tight_layout(rect=[0, 0.03, 1, 0.95])
    figure.suptitle('Final Population Histograms')
    plt.savefig("{}/histograms.png".format(save_loc), transparent=transparent)
    plt.close()


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


def plotParetoFront(population, config, save_loc=None):
    paretoFront = []
    for i in range(config["popsize"]):
        if not any([population[j] > population[i] for j in range(config["popsize"]) if j != i]):
            paretoFront.append(population[i])
    funcNames = list(config["eval_funcs"].keys())
    for feature1, feature2 in zip(funcNames[:-1], funcNames[1:]):
        R = sorted(sorted([(org.evaluationScores[feature1], org.evaluationScores[feature2]) for org in paretoFront], key=lambda r: r[1], reverse=True), key=lambda r: r[0])
        plt.plot(*T(R), marker="o", linestyle="--")
        plt.title(feature1+" "+feature2)
        plt.xlabel(feature1 + " MSE")
        plt.ylabel(feature2 + " MSE")
        if save_loc:
            plt.savefig("{}/pareto_{}_{}.png".format(save_loc, feature1, feature2))
            plt.close()
        else:
            plt.show()


def run_rep(i, config):
    save_loc = "{}/{}/{}".format(config["data_dir"], config["name"], i)
    try:
        os.makedirs(save_loc)
    except:
        print("{} already exists. Please delete or rename experiment.".format(save_loc))
        exit()

    final_pop, fitness_log = run(config)

    if config["save_data"] == 1:
        with open("{}/final_pop.pkl".format(save_loc), "wb") as f:
            pickle.dump(final_pop, f)
        with open("{}/fitness_log.pkl".format(save_loc), "wb") as f:
            pickle.dump(fitness_log, f)

    if config["plot_data"] == 1:
        plot_fitness(fitness_log, config["eval_funcs"].keys(), save_loc)
        final_pop_histogram(Evaluation(config), final_pop, config["eval_funcs"], save_loc)
        final_pop_histogram_all(Evaluation(config), final_pop, config["eval_funcs"], save_loc)
        plotParetoFront(final_pop, config, save_loc)
        final_pop[0].saveGraphFigure("{}/graphFigure.png".format(save_loc))


def main(config, rep=None):
    if rep:
        run_rep(rep, config)
    else:
        for i in range(config["reps"]):
            run_rep(i, config)

    config_path = "{}/{}/config.json".format(config["data_dir"], config["name"])
    if not os.path.exists(config_path):
        with open(config_path, "w") as f:
            json.dump(config, f, indent=4)


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