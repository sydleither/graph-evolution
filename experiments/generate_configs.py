import json
import sys


def connectance_configs(eval_funcs, property_names):
    exp_names = []
    for scheme in ["lexicase", "NSGAII"]:
        for connectance in [0.25, 0.5, 0.75]:
            for size in [10, 50, 100]:
                exp_name = "{}_connectance_{}_{}_{}".format(scheme, property_names, str(connectance).replace('.', ''), size)

                eval_funcs["connectance"] = {"target":connectance}

                config = {
                    "data_dir": "data",
                    "name": exp_name,
                    "reps": 1,
                    "save_data": 1,
                    "plot_data": 0,
                    "scheme": scheme,
                    "popsize": 250,
                    "mutation_rate": 0.005,
                    "mutation_odds": [1,2,1],
                    "crossover_odds": [1,2,2],
                    "crossover_rate": 0.6,
                    "weight_range": [-1,1],
                    "network_size": size,
                    "network_sparsity": 1-connectance,
                    "num_generations": 1000,
                    "epsilon": 0.025,
                    "eval_funcs": eval_funcs
                }

                config_path = "experiments/configs/{}.json".format(exp_name)
                with open(config_path, "w") as f:
                    json.dump(config, f, indent=4)

                exp_names.append(exp_name)
    return exp_names


def connectance_experiment(generate_script):
    #generate config files
    configs = []
    configs += connectance_configs({}, "")

    eval_funcs = {
        "in_degree_distribution": {"name": "scale-free", "gamma": 1.5, "offset": 2},
        "out_degree_distribution": {"name": "scale-free", "gamma": 1.5, "offset": 2},
    }
    configs += connectance_configs(eval_funcs, "DD")

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
            for config_name in configs:
                f.write("sbatch /mnt/home/leithers/graph_evolution/graph-evolution/experiments/hpcc.sb {}.json\n".format(config_name))
        #generate bash script to analyze the runs when they are all done
        with open("experiments/analyze_experiments", "w") as f:
            f.write("cd /mnt/home/leithers/graph_evolution/graph-evolution\n")
            for config_name in configs:
                f.write("python3 replicate_analysis.py $SCRATCH/graph-evolution/data/{}\n".format(config_name))


if __name__ == "__main__":
    experiment_name = sys.argv[1]
    generate_script = True if len(sys.argv) == 3 else False
    if experiment_name == "connectance":
        connectance_experiment(generate_script)
    else:
        print("Please give a valid experiment name.")