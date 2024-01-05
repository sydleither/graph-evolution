import os
import pandas as pd
import pickle
import sys


OBJECTIVES_OF_INTEREST = ["connectance", "average_positive_interactions_strength", "number_of_competiton_pairs", 
                          "positive_interactions_proportion", "strong_components", "proportion_of_self_loops",
                          "in_degree_distribution", "out_degree_distribution", "pos_in_weight_distribution",
                          "pos_out_weight_distribution", "neg_in_weight_distribution", "neg_out_weight_distribution"]


def main(data_dir):
    df_cols = ["experiment_name", "rep", "selection", "objective", "target", "network_size", 
               "MSE", "Entropy", "Drift_Entropy_Avg"]
    df_rows = []

    for experiment_dir in os.listdir(data_dir):
        full_obj_path = "{}/{}".format(data_dir, experiment_dir)
        if not os.path.isfile(full_obj_path):
            for rep_dir in os.listdir(full_obj_path):
                full_rep_path = "{}/{}".format(full_obj_path, rep_dir)
                if not os.path.isfile(full_rep_path):
                    #skip uncompleted experiments
                    if not os.path.exists("{}/final_pop.pkl".format(full_rep_path)):
                        print("Skipped {} rep {}".format(experiment_dir, rep_dir))
                        continue
                    #read in fitness log
                    with open("{}/fitness_log.pkl".format(full_rep_path), "rb") as f:
                        fitness_log = pickle.load(f)
                    #read in entropy csv as a dataframe
                    entropy_df = pd.read_csv("{}/entropy.csv".format(full_rep_path))
                    #subset entropy dataframe to only include objectives of interest under drift
                    drift_objectives = [x for x in OBJECTIVES_OF_INTEREST if x not in list(fitness_log.keys())]
                    drift_entropies = entropy_df.loc[entropy_df["Name"].isin(drift_objectives)]
                    #get mean and std of drift entropies
                    drift_entropy_avg = drift_entropies["Entropy(bits)"].mean() #TODO change this to delta entropy
                    #get details of experiments via experiment directory name
                    #selection_scheme, experiment_name, target, network_size = experiment_dir.split("-")
                    parts_of_experiment_dir_name = experiment_dir.split("_")
                    selection_scheme = parts_of_experiment_dir_name[0]
                    target = parts_of_experiment_dir_name[-2]
                    network_size = parts_of_experiment_dir_name[-1]
                    experiment_name = "_".join(parts_of_experiment_dir_name[1:-2])
                    #add values of interest to a list to turn into a dataframe
                    for objective,fitnesses in fitness_log.items():
                        entropy = entropy_df.loc[entropy_df["Name"] == objective]["Entropy(bits)"].values[0]
                        df_rows.append([experiment_name, rep_dir, selection_scheme, objective, target, network_size, 
                                        fitnesses[-1], entropy, drift_entropy_avg])

    df = pd.DataFrame(data=df_rows, columns=df_cols)
    df_reduced = df[["selection", "experiment_name", "objective", "MSE", "Entropy", "Drift_Entropy_Avg"]]
    print(df_reduced.groupby(["selection", "experiment_name", "objective"]).mean())


if __name__ == "__main__":
    if len(sys.argv) == 2:
        main(sys.argv[1])
    else:
        print('Please give a valid data directory holding experiments.')