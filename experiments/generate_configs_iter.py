from itertools import combinations
import json

import numpy as np
from scipy.stats import norm, expon


def experiment_config(exp_dir, objectives_name, eval_funcs, network_size):
    exp_name = "{}_{}".format(objectives_name, network_size)

    config = {
        "data_dir": exp_dir,
        "name": exp_name,
        "reps": 1,
        "save_data": 1,
        "plot_data": 0,
        "scheme": "NSGAII",
        "popsize": 200,
        "mutation_rate": 0.005,
        "mutation_odds": [1,2,1],
        "crossover_odds": [1,2,2],
        "crossover_rate": 0.6,
        "weight_range": [-1,1],
        "network_size": network_size,
        "network_sparsity": 0.5,
        "num_generations": 500 if network_size == 10 else 2500,
        "epsilon": 0.025,
        "eval_funcs": eval_funcs
    }

    config_path = "experiments/configs/{}.json".format(exp_name)
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)

    return exp_name


#https://stackoverflow.com/questions/312443/how-do-i-split-a-list-into-equally-sized-chunks
def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def generate_scripts(exp_dir, config_names):
    config_chunks = chunks(config_names, 99)
    for i,chunk in enumerate(config_chunks):
        #generate bash script to run all the configs on the hpcc
        with open(f"experiments/run_experiments_hpcc{i}", "w") as f:
            f.write("cd /mnt/scratch/leithers/graph-evolution\n")
            f.write("cp /mnt/home/leithers/graph_evolution/graph-evolution/main.py .\n"+
                    "cp /mnt/home/leithers/graph_evolution/graph-evolution/ga.py .\n"+
                    "cp /mnt/home/leithers/graph_evolution/graph-evolution/eval_functions.py .\n"+
                    "cp /mnt/home/leithers/graph_evolution/graph-evolution/organism.py .\n"+
                    "cp /mnt/home/leithers/graph_evolution/graph-evolution/bintools.py .\n"+
                    "cp /mnt/home/leithers/graph_evolution/graph-evolution/plot_utils.py .\n")
            for config_name in chunk:
                f.write("sbatch /mnt/home/leithers/graph_evolution/graph-evolution/experiments/hpcc.sb {}.json\n".format(config_name))
        #generate bash script to analyze the runs when they are all done
        with open(f"experiments/analyze_experiments{i}", "w") as f:
            f.write("cd /mnt/home/leithers/graph_evolution/graph-evolution\n")
            for config_name in chunk:
                f.write("python3 replicate_analysis.py $SCRATCH/graph-evolution/{}/{}\n".format(exp_dir, config_name))


def iteration_experiment(exp_dir):
    config_names = []
    for network_size in [10, 50, 100]:
        ns_inv = 1/network_size
        if network_size == 10:
            basically_exp = [ns_inv*np.floor(expon.pdf(x, loc=1, scale=network_size/5)/ns_inv) for x in range(network_size+1)]
        else:
            basically_exp = [ns_inv*np.round(expon.pdf(x, loc=1, scale=network_size/5)/ns_inv) for x in range(network_size+1)]
        eval_funcs = [
        {
            "in_degree_distribution": {"target": basically_exp},
            "out_degree_distribution": {"target": basically_exp},
            "strong_components": {"target": 1},
            "proportion_of_self_loops": {"target": 0},
            #"number_of_competiton_pairs": {"target": network_size/5},
            "positive_interactions_proportion": {"target": 0.75}
        },
        {
            "in_degree_distribution": {"target": basically_exp},
            "out_degree_distribution": {"target": basically_exp},
            "positive_interactions_proportion": {"target": 0.75},
            "average_positive_interactions_strength": {"target": 0.75},
            "average_negative_interactions_strength": {"target": -0.25},
            #"number_of_competiton_pairs": {"target": network_size/5}
        },
        {
            "positive_interactions_proportion": {"target": 0.75},
            "average_negative_interactions_strength": {"target": -0.25},
            "average_positive_interactions_strength": {"target": 0.75},
            "connectance": {"target": 0.25},
            "number_of_competiton_pairs": {"target": network_size/5},
            "proportion_of_self_loops": {"target": 0.1},
        }
        ]
        for i in range(1,7):
            for j,eval_func in enumerate(eval_funcs):
                if j == 1:
                    combo = list(combinations(eval_func, i))
                    for k,c in enumerate(combo):
                        new_eval_func = {c[x]:eval_func[c[x]] for x in range(len(c))}
                        config_name = experiment_config(exp_dir, "{}_{}_{}".format(i, j, k), new_eval_func, network_size)
                        config_names.append(config_name)
    return config_names


if __name__ == "__main__":
    experiment_name = "iter"
    config_names = iteration_experiment(experiment_name)
    print(len(config_names))
    generate_scripts(experiment_name, config_names)