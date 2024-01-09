from random import random
from organism import Organism
from main import final_pop_histogram,final_pop_distribution,plotParetoFront
from eval_functions import Evaluation
import sys
import json
import os
from collections import Counter
from numpy import log2
from typing import Callable


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


if __name__ == "__main__":
    try:
        config_file = sys.argv[1]
        config = json.load(open(config_file))
        eval_obj = Evaluation(config)
    except:
        print("Please give a valid config json to read parameters from.")
        exit()

    numSamples = config["popsize"]
    NUM_NODES = config["network_size"]
    WEIGHT_RANGE = config["weight_range"]
    E = config["epsilon"]
    save_loc = "{}/{}".format(config["data_dir"], config["name"]+"UNIFORM")

    print("Generating samples... (This may take some time)")
    samples = [Organism(NUM_NODES,random(),WEIGHT_RANGE) for n in range(numSamples)]

    eval_obj = Evaluation(config)

    #build eval dict
    eval_funcs:dict[str:tuple[Callable,float]] = {}
    for eval_func_name, eval_func_params in config["eval_funcs"].items():
        if eval_func_name.endswith("distribution"):
            target = eval_obj.target_dist_dict[eval_func_name]
        else:
            target = eval_func_params["target"] if "target" in eval_func_params.keys() else 0
        eval_funcs[eval_func_name] = (eval_obj.functions[eval_func_name], target)

    #use eval dict to eval all organisms
    for func_name, funcPack in eval_funcs.items():
        func_fitnesses = [org.getEvaluationScores({func_name:funcPack})[func_name] for org in samples]

    #check how many solutions in the sampled set meet the specified targets
    cut:list[int] = [i for i in range(numSamples)] #IDs of organisms that 'make the cut'
    for name, func_params in config["eval_funcs"].items():
        if name.endswith("distribution"):
            T = eval_obj.target_dist_dict[name]
            cut = [i for i in cut if sum([abs(o-t) for o,t in zip(samples[i].getProperty(name,eval_funcs[name]),T)]) <= E] #total absolute error <= epsilon
        else:
            T = func_params["target"]
            cut = [i for i in cut if T*(1-E) <= samples[i].getProperty(name,eval_funcs[name]) <= T*(1+E)]
        if len(cut) == 0:
            break
    print(len(cut), "organisms meet the specified requirements within", config["epsilon"], "%")
    
    #init save location
    if not os.path.exists(save_loc):
        os.makedirs(save_loc)

    print("Plotting data...")

    print("All histograms...")
    final_pop_histogram(eval_obj, samples, config["eval_funcs"], save_loc, plot_all=True)

    print("Distributions with error...")
    final_pop_distribution(eval_obj, samples, config["eval_funcs"], save_loc, plot_all=True, with_error=False)
    
    print("All distributions...")
    final_pop_distribution(eval_obj, samples, config["eval_funcs"], save_loc, plot_all=True, with_error=True)
    
    print("Saving Pareto Front")
    plotParetoFront(samples, config, save_loc)

    print("Saving Entropies...")
    diversity(samples,config,save_loc)

    print("DONE!")