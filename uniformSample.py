from random import random
from organism import Organism
from main import final_pop_histogram,final_pop_distribution,plotParetoFront
from eval_functions import Evaluation
import sys
import json
import os
from collections import Counter
from numpy import log2

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
    save_loc = "{}/{}".format(config["data_dir"], config["name"]+"UNIFORM")

    print("Generating samples... (This may take some time)")
    samples = [Organism(NUM_NODES,random(),WEIGHT_RANGE) for n in range(numSamples)]

    eval_obj = Evaluation(config)
    if not os.path.exists(save_loc):
            os.makedirs(save_loc)
    print("Plotting data...")
    print("All histograms...")
    final_pop_histogram(eval_obj, samples, config["eval_funcs"], save_loc, plot_all=True)
    print("Distributions with error...")
    final_pop_distribution(eval_obj, samples, config["eval_funcs"], save_loc, plot_all=True, with_error=False)
    print("All distributions...")
    final_pop_distribution(eval_obj, samples, config["eval_funcs"], save_loc, plot_all=True, with_error=True)
    
    # plotParetoFront(samples, config, save_loc)
    print("Saving Entropies...")
    diversity(samples,config,save_loc)

    print("DONE!")