import json
import sys

import numpy as np
from scipy.stats import norm, expon


def experiment_config(objectives_name, objectives, target_dicts, network_size):
    exp_name = "{}_{}".format(objectives_name, network_size)

    config = {
        "data_dir": "data",
        "name": exp_name,
        "reps": 1,
        "save_data": 1,
        "plot_data": 0,
        "scheme": "NSGAII",
        "popsize": 100,
        "mutation_rate": 0.005,
        "mutation_odds": [1,2,1],
        "crossover_odds": [1,2,2],
        "crossover_rate": 0.6,
        "weight_range": [-1,1],
        "network_size": network_size,
        "network_sparsity": 0.5,
        "num_generations": 500 if network_size == 10 else 1000,
        "epsilon": 0.025,
        "eval_funcs": {objectives[i]: target_dicts[i] for i in range(len(objectives))}
    }

    config_path = "experiments/configs/{}.json".format(exp_name)
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)

    return exp_name


def one_objective_experiment(objectives):
    config_names = []
    for objective in objectives:
        for network_size in [10, 50, 100]:
            target_dicts = set_target_dicts(objective, network_size)
            for i in range(len(target_dicts)):
                target_dict = target_dicts[i]
                config_names.append(experiment_config("{}_{}".format(objective, i), [objective], [target_dict], network_size))
    return config_names


def set_target_dicts(objective, network_size):
    if objective == "strong_components":
        target_dicts = [{"target": 1}]
    elif objective == "number_of_competiton_pairs":
        target_dicts = [{"target":x} for x in [0, network_size/5]]
    elif objective.endswith("degree_distribution"):
        ns_inv = 1/network_size
        if network_size == 10:
            basically_exp = [ns_inv*np.floor(expon.pdf(x, loc=1, scale=network_size/5)/ns_inv) for x in range(network_size+1)]
        else:
            basically_exp = [ns_inv*np.round(expon.pdf(x, loc=1, scale=network_size/5)/ns_inv) for x in range(network_size+1)]
        basically_norm = [ns_inv*np.round(norm.pdf(x, loc=network_size/4, scale=network_size/10)/ns_inv) for x in range(network_size+1)]
        target_dicts = [{"target": basically_norm}, {"target": basically_exp}]
    elif objective.startswith("neg_"):
        target_dicts = [{"name": "uniform", "value": -0.25},
                        {"name": "linear", "a":-1/network_size, "b": 0},
                        {"target": [-(x%(network_size/5))/(network_size/5) for x in range(network_size)]}]
    elif objective.startswith("pos_"):
        target_dicts = [{"name": "uniform", "value": 0.25},
                        {"name": "linear", "a":1/network_size, "b": 0},
                        {"target": [(x%(network_size/5))/(network_size/5) for x in range(network_size)]}]
    else:
        target_dicts = [{"target": 0.25}, {"target": 0.75}]
    return target_dicts


def generate_scripts(config_names):
    #generate bash script to run all the configs on the hpcc
    with open("experiments/run_experiments_hpcc", "w") as f:
        f.write("cd /mnt/scratch/leithers/graph-evolution\n")
        f.write("cp /mnt/home/leithers/graph_evolution/graph-evolution/main.py .\n"+
                "cp /mnt/home/leithers/graph_evolution/graph-evolution/ga.py .\n"+
                "cp /mnt/home/leithers/graph_evolution/graph-evolution/eval_functions.py .\n"+
                "cp /mnt/home/leithers/graph_evolution/graph-evolution/organism.py .\n"+
                "cp /mnt/home/leithers/graph_evolution/graph-evolution/bintools.py .\n"+
                "cp /mnt/home/leithers/graph_evolution/graph-evolution/plot_utils.py .\n")
        for config_name in config_names:
            f.write("sbatch /mnt/home/leithers/graph_evolution/graph-evolution/experiments/hpcc.sb {}.json\n".format(config_name))
    #generate bash script to analyze the runs when they are all done
    with open("experiments/analyze_experiments", "w") as f:
        f.write("cd /mnt/home/leithers/graph_evolution/graph-evolution\n")
        for config_name in config_names:
            f.write("python3 replicate_analysis.py $SCRATCH/graph-evolution/data/{}\n".format(config_name))


if __name__ == "__main__":
    experiment_name = sys.argv[1]
    generate_script = True if len(sys.argv) == 3 else False
    config_names = []
    if experiment_name == "single_prop":
        objectives = ["connectance", "average_positive_interactions_strength", "number_of_competiton_pairs", 
                      "positive_interactions_proportion", "strong_components", "proportion_of_self_loops"]
        config_names = one_objective_experiment(objectives)
    elif experiment_name == "single_dist":
        objectives = ["in_degree_distribution", "out_degree_distribution", "pos_in_weight_distribution",
                    "pos_out_weight_distribution", "neg_in_weight_distribution", "neg_out_weight_distribution"]
        config_names = one_objective_experiment(objectives)
    else:
        print("Please give a valid experiment name.")
    if generate_script:
        generate_scripts(config_names)