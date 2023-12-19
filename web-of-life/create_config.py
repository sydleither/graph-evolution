import json
import os
import sys
sys.path.append("../")
import numpy as np
from organism import Organism
from eval_functions import Evaluation


def extract_properties(org):
    eval = Evaluation({"eval_funcs":{}})
    eval_funcs = {
        "in_degree_distribution": {"target":eval.in_degree_distribution(org)},
        "out_degree_distribution": {"target":eval.out_degree_distribution(org)},
        "connectance": {"target":eval.connectance(org)},
        "average_positive_interactions_strength": {"target":eval.average_positive_interactions_strength(org)}
    }
    return eval_funcs


def main(network_name, network):
    org = Organism(len(network), 0, (0,1))
    org.adjacencyMatrix = network
    org.numInteractions = sum([sum([1 for val in row if val != 0]) for row in network])
    org.numPositive = sum([sum([1 for val in row if val > 0]) for row in network])

    eval_funcs = extract_properties(org)

    config = {
        "data_dir": "data",
        "name": network_name,
        "reps": 1,
        "save_data": 1,
        "plot_data": 0,
        "popsize": 100,
        "mutation_rate": 0.005,
        "mutation_odds": [1,2,1],
        "crossover_odds": [1,2,2],
        "crossover_rate": 0.6,
        "weight_range": [0,1],
        "network_size": len(network),
        "network_sparsity": 1-eval_funcs["connectance"]["target"],
        "num_generations": 500,
        "epsilon": 0.05,
        "eval_funcs" : eval_funcs
    }

    config_path = "../{}.json".format(network_name)
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)


if __name__ == "__main__":
    try:
        network_file = sys.argv[1]
        network = np.loadtxt(network_file, delimiter=',')
    except:
        print("Please give a valid formatted ecological network to read in.")
        exit()
    main(os.path.basename(network_file)[:-4], network)