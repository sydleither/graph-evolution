import json
import os
import pickle
import sys
from collections import Counter

import matplotlib.pyplot as plt
from eval_functions import Evaluation
from ga import run
from numpy import log2
from organism import Organism
from plot_utils import T, final_pop_distribution, final_pop_histogram


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
    for i, feature1 in enumerate(funcNames):
        for j, feature2 in enumerate(funcNames):
            if j <= i: continue
            R = sorted(sorted([(org.evaluationScores[feature1], org.evaluationScores[feature2]) for org in paretoFront], key=lambda r: r[1], reverse=True), key=lambda r: r[0])
            plt.plot(*T(R), marker="o", linestyle="--")
            plt.title(feature1+" "+feature2)
            plt.xlabel(feature1 + " MSE")
            plt.ylabel(feature2 + " MSE")
            if save_loc is not None:
                plt.savefig("{}/pareto_{}_{}.png".format(save_loc, feature1, feature2))
                plt.close()
            else:
                plt.show()


def diversity(population:list[Organism],config:dict,save_loc_i:str) :
    global eval_obj
    N = config["popsize"]
    with open("{}/entropy.csv".format(save_loc_i),'w') as diversityFile:
        diversityFile.write("Name,Entropy(bits)\n")
        for eval_func_name,eval_func in eval_obj.functions.items():
            typeCounter = Counter([organism.getProperty(eval_func_name,eval_func) if "distribution" not in eval_func_name 
                                else tuple(organism.getProperty(eval_func_name,eval_func)) for organism in population])
            entropy = -sum([(count/N)*log2(count/N) for count in typeCounter.values()])
            diversityFile.write("{},{}\n".format(eval_func_name,entropy))


def run_rep(i, save_loc, config):
    global eval_obj
    save_loc_i = "{}/{}".format(save_loc, i)
    if not os.path.exists(save_loc_i):
        os.makedirs(save_loc_i)

    final_pop, fitness_log = run(config)

    if config["save_data"] == 1:
        with open("{}/final_pop.pkl".format(save_loc_i), "wb") as f:
            pickle.dump(final_pop, f)
        with open("{}/fitness_log.pkl".format(save_loc_i), "wb") as f:
            pickle.dump(fitness_log, f)
        diversity(final_pop,config,save_loc_i)

    if config["plot_data"] == 1:
        plot_fitness(fitness_log, config["eval_funcs"].keys(), save_loc_i)
        final_pop_histogram(eval_obj, final_pop, config["eval_funcs"], save_loc_i, plot_all=True)
        final_pop_histogram(eval_obj, final_pop, config["eval_funcs"], save_loc_i, plot_all=False)
        final_pop_distribution(eval_obj, final_pop, config["eval_funcs"], save_loc_i, plot_all=True, with_error=False)
        final_pop_distribution(eval_obj, final_pop, config["eval_funcs"], save_loc_i, plot_all=True, with_error=True)
        plotParetoFront(final_pop, config, save_loc_i)
        final_pop[0].saveGraphFigure("{}/graphFigure.png".format(save_loc_i))


def main(config, rep=None):
    save_loc = "{}/{}".format(config["data_dir"], config["name"])

    if rep:
        run_rep(rep, save_loc, config)
    else:
        for i in range(config["reps"]):
            run_rep(i, save_loc, config)

    config_path = "{}/config.json".format(save_loc)
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)


if __name__ == "__main__":
    try:
        config_file = sys.argv[1]
        config = json.load(open(config_file))
        eval_obj = Evaluation(config)
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