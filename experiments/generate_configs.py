import json
import sys


def one_objective_configs(objective, target_dicts):
    exp_names = []
    for scheme in ["lexicase", "NSGAII"]:
        for network_size in [10, 50, 100]:
            for target_val_i in range(len(target_dicts)):
                exp_name = "{}_{}_{}_{}".format(scheme, objective, target_val_i, network_size)

                target_dict = target_dicts[target_val_i]
                config = {
                    "data_dir": "data",
                    "name": exp_name,
                    "reps": 1,
                    "save_data": 1,
                    "plot_data": 0,
                    "scheme": scheme,
                    "popsize": 100,
                    "mutation_rate": 0.005,
                    "mutation_odds": [1,2,1],
                    "crossover_odds": [1,2,2],
                    "crossover_rate": 0.6,
                    "weight_range": [-1,1],
                    "network_size": network_size,
                    "network_sparsity": 0.5,
                    "num_generations": 500,
                    "epsilon": 0.025,
                    "eval_funcs": {objective: target_dict}
                }

                config_path = "experiments/configs/{}.json".format(exp_name)
                with open(config_path, "w") as f:
                    json.dump(config, f, indent=4)

                exp_names.append(exp_name)
    return exp_names


def one_objective_experiment(objectives, generate_script):
    #generate config files
    config_names = []
    for objective in objectives:
        if objective.endswith("distribution"):
            target_dicts = [{"name": "scale-free", "gamma": 2, "offset": 2}] #need to improve this
        else:
            target_dicts = [{"target": x} for x in [0.25, 0.5, 0.75]]
        config_names += one_objective_configs(objective, target_dicts)

    if generate_script:
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
    objectives = ["connectance", "average_positive_interactions_strength", "number_of_competition_pairs", 
                  "positive_interactions_proportion", "strong_components", "proportion_of_self_loops",
                  "in_degree_distribution", "out_degree_distribution", "pos_in_weight_distribution",
                  "pos_out_weight_distribution", "neg_in_weight_distribution", "neg_out_weight_distribution"]

    experiment_name = sys.argv[1]
    generate_script = True if len(sys.argv) == 3 else False
    if experiment_name == "single":
        one_objective_experiment(objectives, generate_script)
    else:
        print("Please give a valid experiment name.")